/* k3_ops.c - the numeric core of the Kimi K3 engine.
 *
 * Every routine here is gated on a JSON fixture under tests/fixtures/ops/, generated
 * by tools/emit_fixtures.py from the pure-torch reference. Per-op fixtures exist
 * alongside the full-model oracle because the oracle is pass-or-fail: it proves the
 * stack is wrong without indicating which of ~40 kernels is responsible.
 *
 * FLOATING-POINT CONTRACT. This file requires -ffp-contract=off and no -ffast-math;
 * both build systems set them. The kernels are written so that three implementations
 * of the same operation agree exactly:
 *
 *   - the scalar C99 path, which is the reference,
 *   - the OpenMP path (guarded by _OPENMP), which parallelises only over independent
 *     output rows, so it introduces no reduction and changes no arithmetic,
 *   - the AVX2 path (guarded by __AVX2__), which reproduces the scalar code's
 *     four-accumulator partition and reduction tree exactly rather than choosing a
 *     more natural one.
 *
 * That last point is the reason several loops here look hand-unrolled for no visible
 * gain: the unrolling fixes a summation ORDER that the vector path must match. Reduce
 * them to the obvious form and the paths diverge in the last bits, which shows up as
 * a fixture failure on one machine and a pass on another.
 *
 * Accumulators are double throughout. Hidden size is 7168 and expert rows are 2048
 * wide; a float32 accumulator loses precision the reference comparisons can see.
 */
#include "k3.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------------- fatal errors ---- */
/* Several kernels here need a small temporary that cannot be hoisted into caller-owned
 * scratch without changing a published signature. They are hundreds of bytes to a few
 * kilobytes, on a path where the engine has already reserved tens of gigabytes, so a
 * failure means the process is finished either way.
 *
 * What matters is HOW it finishes. These kernels return void, so a failed allocation
 * could only be handled by returning early, which leaves the output buffer holding
 * whatever was in it before, and the caller consumes it as a result. The run then
 * completes and prints a plausible token computed from uninitialised memory. That is the
 * one failure mode this engine treats as unacceptable, so allocation failure aborts
 * loudly instead.
 *
 * This is deliberately NOT how streamed experts are handled: an expert that fails to
 * load is recoverable in principle, so it is counted in k3_expert_drops and the decision
 * is left to the caller. Memory exhaustion inside a kernel is not recoverable. */
static void k3_fatal_oom(const char *what, size_t bytes)
{
    fprintf(stderr,
            "k3: FATAL, could not allocate %zu bytes for %s.\n"
            "    Aborting rather than continuing with an uninitialised buffer, which\n"
            "    would produce plausible-looking but meaningless output.\n",
            bytes, what);
    abort();
}

/* The same rule, for a bound that is exceeded rather than an allocation that fails.
 * k3_mla_cached writes its output only after the capacity check, so returning early
 * would hand the caller back whatever the scratch buffer held from the previous layer,
 * and k3_decoder_layer_inc folds that straight into the residual. The run finishes and
 * prints a token derived from the wrong layer's activations. Abort instead. */
static void k3_fatal_bound(const char *what, long value, long limit)
{
    fprintf(stderr,
            "k3: FATAL, %s is %ld, which exceeds the limit of %ld.\n"
            "    Aborting rather than returning without writing the output buffer,\n"
            "    which would fold the previous layer's values into the residual and\n"
            "    produce plausible-looking but meaningless output.\n"
            "    Shorten the prompt, lower --gen, or drop --incremental.\n",
            what, value, limit);
    abort();
}

/* ------------------------------------------------------------- layer map ---- */
/* The released config lists full_attn_layers ONE-BASED, and
 * configuration_kimi_k3.py:152-156 tests (layer_idx + 1) in kda_layers. Getting this
 * off by one silently swaps KDA and MLA layers throughout the stack. */
int k3_is_mla(const K3Cfg *c, int layer)
{
    for (int i = 0; i < c->n_full_attn; i++)
        if (c->full_attn[i] == layer + 1) return 1;
    return 0;
}
int k3_is_kda(const K3Cfg *c, int layer)   { return !k3_is_mla(c, layer); }
int k3_is_dense(const K3Cfg *c, int layer) { return layer < c->first_dense; }

/* --------------------------------------------------------------- rmsnorm ---- */
void k3_rmsnorm(float *y, const float *x, const float *w, int n, float eps)
{
    /* double accumulator: 7168 squared terms in float32 loses real precision, and
     * every downstream comparison against the reference depends on this. */
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float inv = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
    for (int i = 0; i < n; i++) y[i] = w[i] * x[i] * inv;
}

/* -------------------------------------------------------------- SiTU-GLU ---- */
static inline float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

/* See k3.h. Incremented whenever a streamed expert cannot be fetched. */
long k3_expert_drops = 0;

void k3_situ_glu(float *y, const float *x, int n, float b1, float b2)
{
    const float *gate = x;
    const float *up   = x + n;
    for (int i = 0; i < n; i++) {
        const float g = gate[i];
        /* The sigmoid takes the UNCAPPED gate. Feeding it the capped value instead
         * still yields a bounded, plausible function and is WRONG.
         * modeling_kimi_linear.py:79 */
        const float a = b1 * tanhf(g / b1) * sigmoidf_(g);
        const float u = b2 * tanhf(up[i] / b2);
        y[i] = a * u;
    }
}

/* ------------------------------------------------------------- ShortConv ---- */
/* Causal depthwise conv, SiLU fused, exactly as ShortConvolution(activation='silu').
 * state[c*(k-1) + j] holds the previous inputs for channel c, oldest first.
 * Updated in place so a decode loop can carry it forward. */
void k3_shortconv(float *y, const float *x, const float *w, float *state,
                  int channels, int k, int T)
{
    const int hist = k - 1;
    /* hist can be 0 when k == 1, and malloc(0) is permitted to return NULL. Guard on
     * hist rather than on buf, or a legitimate k == 1 configuration silently skips the
     * whole convolution and leaves y untouched. */
    float *buf = hist ? (float *)malloc((size_t)hist * sizeof(float)) : NULL;
    if (hist && !buf) k3_fatal_oom("ShortConv history", (size_t)hist * sizeof(float));

    for (int c = 0; c < channels; c++) {
        if (hist) {   /* memcpy/memset with a NULL pointer is UB even at length 0 */
            if (state) memcpy(buf, state + (size_t)c * hist, (size_t)hist * sizeof(float));
            else       memset(buf, 0, (size_t)hist * sizeof(float));
        }

        for (int t = 0; t < T; t++) {
            const float cur = x[(size_t)t * channels + c];
            /* taps are ordered oldest..newest, matching conv1d over a left-padded
             * sequence: w[k-1] multiplies the CURRENT input. */
            float acc = w[(size_t)c * k + hist] * cur;
            for (int j = 0; j < hist; j++)
                acc += w[(size_t)c * k + j] * buf[j];

            for (int j = 0; j + 1 < hist; j++) buf[j] = buf[j + 1];
            if (hist > 0) buf[hist - 1] = cur;

            y[(size_t)t * channels + c] = acc * sigmoidf_(acc);   /* SiLU, fused */
        }
        if (state && hist) memcpy(state + (size_t)c * hist, buf, (size_t)hist * sizeof(float));
    }
    free(buf);
}

/* ------------------------------------------------------------ KDA decay ----- */
void k3_kda_decay(float *g, float *alpha, const float *z, const float *A_log,
                  const float *dt_bias, int H, int D, float lb)
{
    for (int h = 0; h < H; h++) {
        /* PER HEAD. The checkpoint stores head_dim floats but only the first H are
         * nonzero. Indexing this per channel is a silent, fatal error. */
        const float a = expf(A_log[h]);
        for (int d = 0; d < D; d++) {
            const int i = h * D + d;
            const float u  = a * (z[i] + dt_bias[i]);
            const float gi = lb * sigmoidf_(u);   /* in (lb, 0] */
            g[i] = gi;
            alpha[i] = expf(gi);                  /* in (e^lb, 1] */
        }
    }
}

/* -------------------------------------------------------- KDA recurrence ---- */
void k3_kda_step(float *S, float *o, const float *q, const float *k,
                 const float *v, const float *alpha, float beta, int dk, int dv)
{
    /* 1. channel-wise decay: scale ROW i of S by alpha[i]. The gate is per key
     *    channel, not a scalar, which is what "channel-wise forget gate" means. */
    for (int i = 0; i < dk; i++) {
        float *row = S + (size_t)i * dv;
        const float a = alpha[i];
        for (int j = 0; j < dv; j++) row[j] *= a;
    }

    /* 2. read the state along k:  u = S^T k */
    /* Allocated AFTER the decay above has already modified S. Returning early here
     * would leave the recurrent state permanently scaled but never updated -- silent,
     * unrecoverable corruption of every subsequent token. */
    float *u = (float *)calloc((size_t)dv, sizeof(float));
    if (!u) k3_fatal_oom("KDA recurrence temporary", (size_t)dv * sizeof(float));
    for (int i = 0; i < dk; i++) {
        const float ki = k[i];
        if (ki == 0.0f) continue;
        const float *row = S + (size_t)i * dv;
        for (int j = 0; j < dv; j++) u[j] += ki * row[j];
    }

    /* 3. rank-one delta write. (v - u) is the prediction error: this is what makes
     *    it a DELTA rule rather than plain accumulation. */
    for (int i = 0; i < dk; i++) {
        const float ki = k[i];
        if (ki == 0.0f) continue;
        float *row = S + (size_t)i * dv;
        for (int j = 0; j < dv; j++) row[j] += ki * beta * (v[j] - u[j]);
    }

    /* 4. output from the ALREADY UPDATED state: o = S^T q */
    for (int j = 0; j < dv; j++) o[j] = 0.0f;
    for (int i = 0; i < dk; i++) {
        const float qi = q[i];
        if (qi == 0.0f) continue;
        const float *row = S + (size_t)i * dv;
        for (int j = 0; j < dv; j++) o[j] += qi * row[j];
    }
    free(u);
}

/* ---------------------------------------------------------------- matmul ---- */
/* The dominant cost in the engine: essentially all compute time is spent here or in
 * k3_matmul_mxfp4. Two properties of this kernel are load bearing.
 *
 *   1. OUTPUT ROWS ARE INDEPENDENT. Parallelising the outer loop introduces no
 *      reduction and no race, and changes no arithmetic at all, each row is summed
 *      by exactly one thread in exactly the order below. Results are therefore
 *      identical at any thread count, which the fixtures rely on.
 *
 *   2. FOUR ACCUMULATORS, PARTITIONED BY i%4, REDUCED AS (a0+a1)+(a2+a3). The split
 *      keeps the FMA pipeline full, a single accumulator serialises on the latency
 *      chain, but it is written out explicitly rather than left to the compiler
 *      because it fixes a summation ORDER. k3_matmul_bf16 and both AVX2 paths
 *      reproduce this exact partition and this exact tree, which is what makes the
 *      three implementations agree bit for bit.
 *
 * Floating-point addition is not associative, so the four-way split is a real change
 * to the arithmetic relative to a sequential sum. Keeping the accumulators in double
 * bounds the difference far below fp32 output precision; making them float would not.
 */
void k3_matmul(float *y, const float *x, const float *W, int in, int out)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) {
        const float *row = W + (size_t)o * in;
        /* Sixteen accumulators, EXPLICITLY fused products. fma() in double is the
         * same IEEE operation as _mm256_fmadd_pd per lane, so the scalar and vector
         * paths stay bit-identical while the dependent-add latency chain that made
         * one accumulator ~10x slower than the machine's floor disappears. The
         * reduction pairs lanes exactly the way the vector path's (v0+v1)+(v2+v3)
         * then cross-lane tree does; change one and you must change the other. */
        double a[16] = {0};
        int i = 0;
        for (; i + 15 < in; i += 16)
            for (int l = 0; l < 16; l++)
                a[l] = fma((double)row[i + l], (double)x[i + l], a[l]);
        double b0 = (a[0] + a[4]) + (a[8]  + a[12]);
        double b1 = (a[1] + a[5]) + (a[9]  + a[13]);
        double b2 = (a[2] + a[6]) + (a[10] + a[14]);
        double b3 = (a[3] + a[7]) + (a[11] + a[15]);
        double acc = (b0 + b1) + (b2 + b3);
        for (; i < in; i++) acc = fma((double)row[i], (double)x[i], acc);
        y[o] = (float)acc;
    }
}

/* ------------------------------------------------------------- Gated MLA ---- */
/* MLA with an optional KV cache, which is what makes incremental decode possible.
 *
 * WHY MLA IS THE ONLY PIECE THAT NEEDS ONE
 *   KDA already carries everything it needs: k3_kda_layer updates its recurrent state
 *   and its ShortConv history in place, so a decode step just declines to clear them.
 *   The attn-res block stack is per token with no cross-token dependency. MLA is the
 *   exception, because softmax attention must see every previous key and value, and
 *   this function recomputes them all from x on every call. That is what makes decode
 *   O(T^2).
 *
 * WHAT IS CACHED, AND WHY NOT THE COMPRESSED LATENT
 *   The obvious saving is to cache the 576-float compressed latent and re-expand it
 *   through kv_b each step, which is 42x smaller. It is also far slower: kv_b is
 *   24576x512, so re-expanding every cached position costs an O(T) sweep of 12.6M-MAC
 *   matmuls per layer per token. The EXPANDED per-head keys and values are cached
 *   instead: 96 heads x 256 floats = 98,304 B per position per layer, and 2.37 MB per
 *   position across the 24 MLA layers. A 64-token generation is therefore 151 MB of
 *   KV cache, small enough not to affect the memory budget at any preset.
 *
 *   The rope slot is cached separately because it is SHARED across heads: 64 values per
 *   position, not per head. Folding it into the per-head block would waste 96x the space
 *   and, worse, invites treating it as per-head somewhere.
 *
 * kvc == NULL selects the self-contained path, which recomputes all keys and values
 * from x and caches nothing. Both paths must produce identical output; the op fixtures
 * gate the uncached path and tests/unit/k3_model.c gates them against each other.
 */
void k3_mla_cached(float *out, const float *x, const K3MlaW *w, const K3Cfg *c,
                   int T, float *scratch,
                   float *kvc, float *ropec, int cached, int cap)
{
    const int E  = c->hidden, H = c->n_heads;
    const int qn = c->qk_nope, qr = c->qk_rope, vh = c->v_head;
    const int qh = qn + qr;                       /* 192: the FULL head width      */
    const int kvw = c->kv_lora + qr;              /* 576: latent + shared rope slot */
    const int kvd = qn + vh;                      /* 256: cached width per head    */
    const float scale = 1.0f / sqrtf((float)qh);  /* :359, over qh not qn           */
    if (!kvc) cached = 0;
    const int last = cached + T - 1;              /* highest absolute position      */
    if (kvc && last >= cap)
        k3_fatal_bound("MLA KV cache position", (long)last, (long)cap - 1);

    /* Scratch layout. Every region below is DISJOINT and must stay so. Overlapping
     * any two of them can appear to work, aliasing the gate buffer onto q, say, is
     * safe only while H*vh < H*qh holds, but that is an accident of the released
     * dimensions, not an invariant, and it breaks silently the moment v_head grows.
     * Size the buffer with k3_mla_scratch_cached(); do not compute it by hand. */
    float *q    = scratch;                          /* [T][H][qh]     */
    float *ct   = q    + (size_t)T * H * qh;        /* [kvw] transient, one token */
    float *ql   = ct   + (size_t)kvw;               /* [q_lora]       */
    float *acc  = ql   + (size_t)c->q_lora;         /* [H][vh]        */
    float *gbuf = acc  + (size_t)H * vh;            /* [H][vh] gate   */
    float *sc   = gbuf + (size_t)H * vh;            /* [last+1] scores */
    /* Without a cache the keys/values live in scratch and cover only this call. */
    float *kvs  = sc   + (size_t)(last + 1);        /* [T][H][kvd]    */
    float *rps  = kvs  + (kvc ? 0 : (size_t)T * H * kvd);   /* [T][qr] */

    #define K3_KV_AT(p)   (kvc   ? kvc   + (size_t)(p) * H * kvd : kvs + (size_t)(p) * H * kvd)
    #define K3_ROPE_AT(p) (ropec ? ropec + (size_t)(p) * qr      : rps + (size_t)(p) * qr)

    /* ---- per-token projections ---- */
    for (int t = 0; t < T; t++) {
        const int p = cached + t;
        const float *xt = x + (size_t)t * E;
        k3_mmw(ql, xt, w->q_a, w->wdt, E, c->q_lora);
        k3_rmsnorm(ql, ql, w->q_a_norm, c->q_lora, c->rms_eps);
        k3_mmw(q + (size_t)t * H * qh, ql, w->q_b, w->wdt, c->q_lora, H * qh);

        /* ONE projection emits the compressed latent AND the shared rope slot */
        k3_mmw(ct, xt, w->kv_a, w->wdt, E, kvw);
        /* the norm covers the latent only, never the rope slot */
        k3_rmsnorm(ct, ct, w->kv_a_norm, c->kv_lora, c->rms_eps);
        memcpy(K3_ROPE_AT(p), ct + c->kv_lora, (size_t)qr * sizeof(float));
        k3_mmw(K3_KV_AT(p), ct, w->kv_b, w->wdt, c->kv_lora, H * kvd);
    }

    /* ---- attention, per head, causal ---- */
    for (int t = 0; t < T; t++) {
        const int p = cached + t;
        for (int h = 0; h < H; h++) {
            const float *qt = q + ((size_t)t * H + h) * qh;
            float m = -INFINITY;
            for (int s = 0; s <= p; s++) {                 /* causal: s <= p */
                const float *ks = K3_KV_AT(s) + (size_t)h * kvd;
                const float *kr = K3_ROPE_AT(s);           /* shared slot */
                double d = 0.0;
                for (int i = 0; i < qn; i++) d += (double)qt[i] * (double)ks[i];
                /* the rope slot is UNROTATED but still scored, and the SAME 64
                 * values serve every head. Dropping this term is the silent bug. */
                for (int i = 0; i < qr; i++) d += (double)qt[qn + i] * (double)kr[i];
                sc[s] = (float)d * scale;
                if (sc[s] > m) m = sc[s];
            }
            double z = 0.0;
            for (int s = 0; s <= p; s++) { sc[s] = expf(sc[s] - m); z += sc[s]; }

            float *o = acc + (size_t)h * vh;
            for (int j = 0; j < vh; j++) o[j] = 0.0f;
            for (int s = 0; s <= p; s++) {
                const float pr = (float)(sc[s] / z);
                const float *vs = K3_KV_AT(s) + (size_t)h * kvd + qn;
                for (int j = 0; j < vh; j++) o[j] += pr * vs[j];
            }
        }

        /* ---- output gate then projection. Gate BEFORE o_proj, and no norm on it,
         * unlike KDA which norms first. :470-473 ---- */
        if (w->g) {
            k3_mmw(gbuf, x + (size_t)t * E, w->g, w->wdt, E, H * vh);
            for (int i = 0; i < H * vh; i++)
                acc[i] *= 1.0f / (1.0f + expf(-gbuf[i]));
        }
        k3_mmw(out + (size_t)t * E, acc, w->o, w->wdt, H * vh, E);
    }
    #undef K3_KV_AT
    #undef K3_ROPE_AT
}

void k3_mla(float *out, const float *x, const K3MlaW *w, const K3Cfg *c,
            int T, float *scratch)
{
    k3_mla_cached(out, x, w, c, T, scratch, NULL, NULL, 0, 0);
}

/* ---------------------------------------------------------------- router ---- */
void k3_router(int *idx, float *w, const float *x, const float *W,
               const float *bias, int hidden, int n_experts, int topk,
               int renorm, float routed_scale)
{
    /* Returning early here would leave idx[] and w[] untouched, and k3_moe forms
     * `w->w1 + idx[j]*I*L` from them one line later -- an arbitrary pointer built from
     * uninitialised stack. */
    float *score  = (float *)malloc((size_t)n_experts * sizeof(float));
    float *choice = (float *)malloc((size_t)n_experts * sizeof(float));
    if (!score || !choice) k3_fatal_oom("router scores", (size_t)n_experts * sizeof(float) * 2);

    /* logits in float32 with no bias, then an independent sigmoid per expert. The
     * reference upcasts both operands explicitly; a double accumulator here matches
     * it and costs nothing at this width. */
    /* PARALLEL over experts, and bit-identical because of it.
     *
     * This is 896 dot products of length 7168 = 6.4M multiply-adds, per token, per MoE
     * layer, so 590M across the 92 of them -- and it ran on ONE core while every other
     * matmul in the engine was already threaded. It is pure arithmetic with no I/O to
     * hide behind, so it sat squarely on the critical path.
     *
     * Each iteration writes only its own score[e] and choice[e], and the ACCUMULATION
     * ORDER INSIDE an expert is untouched: thread t still sums i = 0..hidden-1 in
     * sequence into its own double. Splitting the outer loop therefore cannot change a
     * single bit, which is why this needs no tolerance and no re-gating. */
#ifdef _OPENMP
#   pragma omp parallel for schedule(static)
#endif
    for (int e = 0; e < n_experts; e++) {
        const float *row = W + (size_t)e * hidden;
        double acc = 0.0;
        for (int i = 0; i < hidden; i++) acc += (double)row[i] * (double)x[i];
        score[e]  = 1.0f / (1.0f + expf(-(float)acc));
        choice[e] = score[e] + (bias ? bias[e] : 0.0f);   /* selection score only */
    }

    /* top-k by repeated max. n_experts is 896 and topk is 16, so this is 14k
     * comparisons per token per layer: cheap next to an 18 MB expert read, and it
     * avoids a sort. Marking taken entries with -INFINITY keeps ties deterministic
     * in first-index order, matching a stable selection. */
    for (int j = 0; j < topk; j++) {
        int best = -1; float bv = -INFINITY;
        for (int e = 0; e < n_experts; e++)
            if (choice[e] > bv) { bv = choice[e]; best = e; }
        if (best < 0) { idx[j] = 0; w[j] = 0.0f; continue; }
        idx[j] = best;
        w[j]   = score[best];              /* UNBIASED score, not choice[best] */
        choice[best] = -INFINITY;
    }

    if (renorm && topk > 1) {
        double s = 0.0;
        for (int j = 0; j < topk; j++) s += (double)w[j];
        const float inv = (float)(1.0 / (s + 1e-20));
        for (int j = 0; j < topk; j++) w[j] *= inv;
    }
    for (int j = 0; j < topk; j++) w[j] *= routed_scale;

    free(score); free(choice);
}

/* --------------------------------------------------------------- AttnRes ---- */
void k3_attn_res(float *out, const float *src, const float *fold,
                 int nsrc, int n, float eps)
{
    /* Returning early would leave `out` holding the previous layer's residual, which
     * the caller cannot distinguish from a computed one. */
    float *score = (float *)malloc((size_t)nsrc * sizeof(float));
    if (!score) k3_fatal_oom("AttnRes scores", (size_t)nsrc * sizeof(float));

    for (int s = 0; s < nsrc; s++) {
        const float *v = src + (size_t)s * n;
        double ss = 0.0;
        for (int i = 0; i < n; i++) ss += (double)v[i] * (double)v[i];
        const float inv = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
        /* key is the NORMALISED source; fold already carries norm.weight*proj.weight */
        double acc = 0.0;
        for (int i = 0; i < n; i++) acc += (double)(v[i] * inv) * (double)fold[i];
        score[s] = (float)acc;
    }

    float m = score[0];
    for (int s = 1; s < nsrc; s++) if (score[s] > m) m = score[s];
    double z = 0.0;
    for (int s = 0; s < nsrc; s++) { score[s] = expf(score[s] - m); z += score[s]; }

    for (int i = 0; i < n; i++) out[i] = 0.0f;
    for (int s = 0; s < nsrc; s++) {
        const float p = (float)(score[s] / z);
        const float *v = src + (size_t)s * n;   /* the RAW source, not the key */
        for (int i = 0; i < n; i++) out[i] += p * v[i];
    }
    free(score);
}

/* Exact scratch requirement for k3_mla. Callers should use this rather than
 * duplicating the arithmetic; getting it wrong overruns silently. */
/* Scratch for the self-contained path: keys and values live here, so they scale with T.
 * `cap` is the highest position that will be attended over plus one; without a cache
 * that is just T. */
size_t k3_mla_scratch_cached(const K3Cfg *c, int T, int cap, int cached_mode)
{
    const int H = c->n_heads, qh = c->qk_nope + c->qk_rope, vh = c->v_head;
    const size_t kvd = (size_t)(c->qk_nope + vh);
    size_t n = (size_t)T * H * qh                      /* q            */
             + (size_t)(c->kv_lora + c->qk_rope)       /* ct transient */
             + (size_t)c->q_lora
             + (size_t)2 * H * vh                      /* acc, gbuf    */
             + (size_t)(cap > T ? cap : T);            /* scores       */
    if (!cached_mode) n += (size_t)T * H * kvd + (size_t)T * c->qk_rope;
    return n;
}

size_t k3_mla_scratch(const K3Cfg *c, int T)
{
    return k3_mla_scratch_cached(c, T, T, 0);
}

/* ------------------------------------------------------- Stable LatentMoE ---- */
/* Verified against modeling_kimi_linear.py:815-838. The ORDER is load bearing:
 *   1. route on the FULL hidden width, before any projection      :818
 *   2. down-project to the latent width                            :822
 *   3. run the selected experts IN LATENT SPACE and sum, weighted  :825
 *   4. RMSNorm the AGGREGATE, never per expert                     :831
 *   5. up-project back to hidden                                   :832
 *   6. add the shared expert computed on the ORIGINAL input,
 *      with NO routing weight and NO scaling                       :837
 *
 * Routed experts live at the latent width (latent -> moe_inter -> latent); the
 * shared expert is one wider MLP at full width with intermediate moe_inter*n_shared.
 *
 * Every scratch region below is DISJOINT and must stay so. Reusing the gate/up buffer
 * for the down-projection output is safe only while 2*moe_inter >= latent. That holds
 * for the released config and for the tiny one, but it is an accident of the
 * numbers rather than an invariant, and it is the same class of hazard documented at
 * the scratch layout in k3_mla_cached. Size with k3_moe_scratch().
 */
void k3_moe(float *out, const float *x, const K3MoeW *w, const K3Cfg *c,
            int T, int *idx, float *wt, float *scratch)
{
    const int E = c->hidden, L = c->latent, I = c->moe_inter;
    const int SI = I * c->n_shared;

    float *z    = scratch;              /* [L]    latent input              */
    float *accL = z    + L;             /* [L]    weighted expert aggregate */
    float *gu   = accL + L;             /* [2*I]  gate|up, one expert       */
    float *act  = gu   + 2 * I;         /* [I]    after SiTU                */
    float *edn  = act  + I;             /* [L]    expert down-projection    */
    float *sgu  = edn  + L;             /* [2*SI] shared gate|up            */
    float *sact = sgu  + 2 * SI;        /* [SI]   shared after SiTU         */
    float *sdn  = sact + SI;            /* [E]    shared down-projection    */

    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        float *ot = out + (size_t)t * E;

        /* 1. route on the FULL width, before the down-projection */
        k3_router(idx, wt, xt, w->gate, w->bias, E, c->n_experts, c->topk,
                  c->moe_renorm, c->routed_scale);

        int nk = c->topk;
        /* Draft cache-only routing: keep only the top-k experts already resident, and
         * renormalise their weights so the mixture still sums as intended. This makes a
         * draft token read ZERO new expert bytes. It is an approximation, which is exactly
         * what a draft is; the exact model verifies every proposed token. */
        if (w->cache_only && w->src && w->src->resident) {
            int m = 0; float wsum = 0.0f;
            for (int j = 0; j < c->topk; j++) {
                if (w->src->resident(w->src, w->layer, idx[j], NULL)) {
                    idx[m] = idx[j]; wt[m] = wt[j]; wsum += wt[j]; m++;
                }
            }
            nk = m;
            if (wsum > 0.0f) for (int j = 0; j < nk; j++) wt[j] /= wsum;
        }

        /* 2. down-project into the latent space */
        k3_mmw(z, xt, w->down, w->wdt, E, L);

        /* 3. the selected experts, in latent space, weighted and summed */
        for (int i = 0; i < L; i++) accL[i] = 0.0f;
        /* Hand the WHOLE top-k to the source first, so its reads can overlap. Without
         * this the loop below misses, blocks on a 17.55 MB read, computes, misses
         * again: a queue depth of one against a drive that needs depth to reach its
         * rated bandwidth. getmany is optional and may be NULL, in which case nothing
         * changes and the loop reads them one at a time exactly as before. */
        if (!w->cache_only && w->src && w->src->getmany)
            w->src->getmany(w->src, w->layer, idx, nk);
        for (int j = 0; j < nk; j++) {
            if (w->src) {
                /* Streamed: the expert stays MXFP4 and the matmul reads nibbles. In
                 * cache-only mode every idx[j] is known resident, so resident() serves it
                 * with no disk read; otherwise get() may read it. */
                K3ExpertQ q;
                int miss = w->cache_only
                    ? !w->src->resident(w->src, w->layer, idx[j], &q)
                    : (w->src->get(w->src, w->layer, idx[j], &q) != 0);
                if (miss) {
                    /* A cache-only draft filtered to resident experts already, so a miss
                     * here is a benign race at worst; skip it, since the draft is
                     * approximate by construction and the exact model verifies. On the
                     * exact path a miss is the unacceptable silent-corruption case: count
                     * it in k3_expert_drops so the caller fails the run (see docs/API.md). */
                    if (w->cache_only) continue;
                    k3_expert_drops++;
                    fprintf(stderr, "EXPERT DROP: layer %d expert %d failed to load; "
                                    "this token is CORRUPT\n", w->layer, idx[j]);
                    continue;
                }
                k3_matmul_mxfp4(gu,     z, q.p1, q.s1, L, I, K3_MXFP4_GROUP);
                k3_matmul_mxfp4(gu + I, z, q.p3, q.s3, L, I, K3_MXFP4_GROUP);
                k3_situ_glu(act, gu, I, c->situ_b1, c->situ_b2);
                k3_matmul_mxfp4(edn, act, q.p2, q.s2, I, L, K3_MXFP4_GROUP);
            } else {
                const float *e1 = w->w1 + (size_t)idx[j] * I * L;   /* gate */
                const float *e3 = w->w3 + (size_t)idx[j] * I * L;   /* up   */
                const float *e2 = w->w2 + (size_t)idx[j] * L * I;   /* down */
                k3_matmul(gu,     z, e1, L, I);
                k3_matmul(gu + I, z, e3, L, I);
                k3_situ_glu(act, gu, I, c->situ_b1, c->situ_b2);
                k3_matmul(edn, act, e2, I, L);
            }
            const float wj = wt[j];
            for (int i = 0; i < L; i++) accL[i] += wj * edn[i];
        }

        /* 4. RMSNorm the AGGREGATE (not per expert), then 5. up-project */
        if (c->latent_norm) k3_rmsnorm(accL, accL, w->latent_norm, L, c->rms_eps);
        k3_mmw(ot, accL, w->up, w->wdt, L, E);

        /* 6. shared expert on the ORIGINAL full-width input, added UNWEIGHTED */
        k3_mmw(sgu,      xt, w->sh1, w->wdt, E, SI);
        k3_mmw(sgu + SI, xt, w->sh3, w->wdt, E, SI);
        k3_situ_glu(sact, sgu, SI, c->situ_b1, c->situ_b2);
        k3_mmw(sdn, sact, w->sh2, w->wdt, SI, E);
        for (int i = 0; i < E; i++) ot[i] += sdn[i];
    }
}

size_t k3_moe_scratch(const K3Cfg *c)
{
    const int SI = c->moe_inter * c->n_shared;
    return (size_t)2 * c->latent          /* z, accL            */
         + (size_t)3 * c->moe_inter       /* gu (2*I) + act (I) */
         + (size_t)c->latent              /* edn                */
         + (size_t)3 * SI                 /* sgu (2*SI) + sact  */
         + (size_t)c->hidden;             /* sdn                */
}

/* Batched MoE for PREFILL over a chunk of T tokens, streamed experts only.
 *
 * k3_moe walks the top-k for each token independently, so across a T-token chunk it
 * fetches an expert once per token that routes to it. Under near-uniform routing that is
 * mostly waste: measured on the released trace, a 32-token chunk touches only ~2.7x fewer
 * unique experts than 16*32 draws, so reading each unique expert ONCE and reusing it for
 * every token in the chunk cuts prefill expert bytes ~3-4x. Prefill is where that matters,
 * because decode feeds one token at a time and has nothing to batch.
 *
 * Exactness is preserved to the last bit. Per token the arithmetic is identical to
 * k3_moe: the routed latent contributions are accumulated in the ORIGINAL top-k order
 * (j = 0..k-1) from a per-(token, slot) buffer, then normalised, up-projected and given
 * the shared expert exactly as before. Only the ORDER in which experts are fetched from
 * disk changes, and that touches no floating-point result.
 *
 * out/x are [T][E], idx/wt scratch are topk-wide (reused per token), scratch is one
 * k3_moe_scratch. This path requires w->src (streamed); the resident path stays on
 * k3_moe, which is what the oracle gates exercise. */
static void moe_prefill_chunk(float *out, const float *x, const K3MoeW *w,
                              const K3Cfg *c, int T, float *scratch);

void k3_moe_prefill(float *out, const float *x, const K3MoeW *w, const K3Cfg *c,
                    int T, int *idx, float *wt, float *scratch)
{
    /* K3_NO_BATCH_PREFILL forces the per-token path, so one binary can produce both the
     * batched and the reference token streams for a bit-identity A/B. */
    static int no_batch = -1;
    if (no_batch < 0) no_batch = getenv("K3_NO_BATCH_PREFILL") ? 1 : 0;
    /* cache_only renormalises per token over the resident subset, which the per-token
     * path already does; the draft's prompt prefill is one-time, so defer rather than
     * duplicate the renorm in the batch. */
    if (!w->src || T <= 1 || no_batch || w->cache_only) {
        k3_moe(out, x, w, c, T, idx, wt, scratch);
        return;
    }
    /* Fixed sub-chunks bound the contribution buffer (14.7 MB at 64 tokens) no matter
     * how long the prompt is; a 32k prefill would otherwise want 7.3 GB of it. Most of
     * the dedup is already captured at this width: the unique-expert count grows far
     * slower than the request count under near-uniform routing. */
    const int CHUNK = 64;
    for (int t0 = 0; t0 < T; t0 += CHUNK) {
        const int n = (T - t0) < CHUNK ? (T - t0) : CHUNK;
        if (n == 1) { k3_moe(out + (size_t)t0 * c->hidden, x + (size_t)t0 * c->hidden,
                             w, c, 1, idx, wt, scratch); continue; }
        moe_prefill_chunk(out + (size_t)t0 * c->hidden, x + (size_t)t0 * c->hidden,
                          w, c, n, scratch);
    }
}

static void moe_prefill_chunk(float *out, const float *x, const K3MoeW *w,
                              const K3Cfg *c, int T, float *scratch)
{
    const int E = c->hidden, Ll = c->latent, I = c->moe_inter;
    const int SI = I * c->n_shared, K = c->topk;

    /* Per-token routing decisions and latent inputs, plus a contribution buffer holding
     * every routed expert's latent output for every token: [T][K][Ll]. At T=32, K=16,
     * Ll=3584 that is ~7.3 MB, trivial beside the tens of GB already reserved. */
    int   *ridx = (int *)  malloc((size_t)T * K * sizeof(int));
    float *rwt  = (float *)malloc((size_t)T * K * sizeof(float));
    float *zz   = (float *)malloc((size_t)T * Ll * sizeof(float));
    float *contrib = (float *)malloc((size_t)T * K * Ll * sizeof(float));
    if (!ridx || !rwt || !zz || !contrib)
        k3_fatal_oom("MoE prefill batch", (size_t)T * K * Ll * sizeof(float));

    /* 1. route every token and down-project it, and collect the batch's unique experts. */
    int  *uniq = (int *)malloc((size_t)T * K * sizeof(int));
    char *seen = (char *)calloc((size_t)c->n_experts, 1);
    if (!uniq || !seen) k3_fatal_oom("MoE prefill index", (size_t)c->n_experts);
    int nu = 0;
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        int   *it = ridx + (size_t)t * K;
        float *wtt = rwt + (size_t)t * K;
        k3_router(it, wtt, xt, w->gate, w->bias, E, c->n_experts, K,
                  c->moe_renorm, c->routed_scale);
        k3_mmw(zz + (size_t)t * Ll, xt, w->down, w->wdt, E, Ll);
        for (int j = 0; j < K; j++) {
            const int e = it[j];
            if (e >= 0 && e < c->n_experts && !seen[e]) { seen[e] = 1; uniq[nu++] = e; }
        }
    }

    /* 2. expert-major: fetch each unique expert ONCE, apply it to every (token, slot)
     * that selected it. gu/act/edn are reused per (expert, token). */
    float *gu  = scratch;                 /* [2*I] */
    float *act = gu + 2 * I;              /* [I]   */
    float *edn = act + I;                 /* [Ll]  */
    if (w->src->getmany) w->src->getmany(w->src, w->layer, uniq, nu);
    for (int u = 0; u < nu; u++) {
        const int e = uniq[u];
        K3ExpertQ q;
        if (w->src->get(w->src, w->layer, e, &q) != 0) {
            k3_expert_drops++;
            fprintf(stderr, "EXPERT DROP: layer %d expert %d failed to load; "
                            "this chunk is CORRUPT\n", w->layer, e);
            continue;
        }
        for (int t = 0; t < T; t++) {
            const int   *it = ridx + (size_t)t * K;
            const float *zt = zz  + (size_t)t * Ll;
            for (int j = 0; j < K; j++) {
                if (it[j] != e) continue;
                k3_matmul_mxfp4(gu,     zt, q.p1, q.s1, Ll, I, K3_MXFP4_GROUP);
                k3_matmul_mxfp4(gu + I, zt, q.p3, q.s3, Ll, I, K3_MXFP4_GROUP);
                k3_situ_glu(act, gu, I, c->situ_b1, c->situ_b2);
                k3_matmul_mxfp4(edn, act, q.p2, q.s2, I, Ll, K3_MXFP4_GROUP);
                memcpy(contrib + ((size_t)t * K + j) * Ll, edn, (size_t)Ll * sizeof(float));
            }
        }
    }

    /* 3. per token, sum contributions in the ORIGINAL top-k order, then the tail of the
     * MoE exactly as k3_moe does it, so every float matches the per-token path. */
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        float *ot = out + (size_t)t * E;
        const float *wtt = rwt + (size_t)t * K;
        /* Reuse this token's now-dead down-projection slot as the aggregate. */
        float *acc = zz + (size_t)t * Ll;
        for (int i = 0; i < Ll; i++) acc[i] = 0.0f;
        for (int j = 0; j < K; j++) {
            const float wj = wtt[j];
            const float *cb = contrib + ((size_t)t * K + j) * Ll;
            for (int i = 0; i < Ll; i++) acc[i] += wj * cb[i];
        }
        if (c->latent_norm) k3_rmsnorm(acc, acc, w->latent_norm, Ll, c->rms_eps);
        k3_mmw(ot, acc, w->up, w->wdt, Ll, E);

        float *sgu  = gu;                 /* [2*SI] */
        float *sact = sgu + 2 * SI;       /* [SI]   */
        float *sdn  = sact + SI;          /* [E]    */
        k3_mmw(sgu,      xt, w->sh1, w->wdt, E, SI);
        k3_mmw(sgu + SI, xt, w->sh3, w->wdt, E, SI);
        k3_situ_glu(sact, sgu, SI, c->situ_b1, c->situ_b2);
        k3_mmw(sdn, sact, w->sh2, w->wdt, SI, E);
        for (int i = 0; i < E; i++) ot[i] += sdn[i];
    }

    free(ridx); free(rwt); free(zz); free(contrib); free(uniq); free(seen);
}

/* --------------------------------------------------------- KDA full layer ---- */
/* L2 normalisation over the last dimension. The reference uses the SUM of squares
 * with eps inside the rsqrt, NOT the mean: k3_ref.py l2norm(). Using the mean here
 * scales every q and k by sqrt(d_k) and quietly changes the attention temperature. */
static void l2norm_(float *v, int n, float eps)
{
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)v[i] * (double)v[i];
    const float inv = (float)(1.0 / sqrt(ss + (double)eps));
    for (int i = 0; i < n; i++) v[i] *= inv;
}

size_t k3_kda_scratch(const K3Cfg *c, int T)
{
    const size_t P = (size_t)c->kda_heads * c->kda_head_dim;
    return 3 * (size_t)T * P        /* q, k, v after conv            */
         + 2 * (size_t)T * P        /* z then alpha                  */
         + (size_t)T * c->kda_heads /* beta                          */
         + (size_t)T * P            /* recurrence output             */
         + 2 * P                    /* gate buffer and one work row  */
         + (size_t)c->kda_head_dim; /* f_a output                    */
}

void k3_kda_layer(float *out, const float *x, const K3KdaW *w, const K3Cfg *c,
                  int T, float *state, float *scratch)
{
    const int E = c->hidden, H = c->kda_heads, D = c->kda_head_dim;
    const int P = H * D, K = c->conv_k, hist = K - 1;

    float *q  = scratch;                 float *k  = q + (size_t)T * P;
    float *v  = k + (size_t)T * P;       float *z  = v + (size_t)T * P;
    float *al = z + (size_t)T * P;       float *bt = al + (size_t)T * P;
    float *o  = bt + (size_t)T * H;      float *gb = o + (size_t)T * P;
    float *wr = gb + P;                  float *fa = wr + P;

    /* 1. projections */
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        k3_mmw(q + (size_t)t * P, xt, w->q, w->wdt, E, P);
        k3_mmw(k + (size_t)t * P, xt, w->k, w->wdt, E, P);
        k3_mmw(v + (size_t)t * P, xt, w->v, w->wdt, E, P);
        k3_mmw(bt + (size_t)t * H, xt, w->b, w->wdt, E, H);
        /* ONE shared low-rank pair feeds every head: [E->D] then [D->H*D] */
        k3_mmw(fa, xt, w->f_a, w->wdt, E, D);
        k3_mmw(z + (size_t)t * P, fa, w->f_b, w->wdt, D, P);
    }

    /* 2. ShortConv with fused SiLU, carrying state across calls */
    float *cs = state ? state + (size_t)H * D * D : NULL;
    k3_shortconv(q, q, w->q_conv, cs ? cs : NULL, P, K, T);
    k3_shortconv(k, k, w->k_conv, cs ? cs + (size_t)P * hist : NULL, P, K, T);
    k3_shortconv(v, v, w->v_conv, cs ? cs + (size_t)2 * P * hist : NULL, P, K, T);

    /* 3. L2Norm on q and k ONLY, per head. v is deliberately left alone. */
    for (int t = 0; t < T; t++)
        for (int h = 0; h < H; h++) {
            l2norm_(q + (size_t)t * P + (size_t)h * D, D, 1e-6f);
            l2norm_(k + (size_t)t * P + (size_t)h * D, D, 1e-6f);
        }

    /* 4/5. beta and the decay chain */
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < H; h++) bt[(size_t)t * H + h] = sigmoidf_(bt[(size_t)t * H + h]);
        k3_kda_decay(z + (size_t)t * P, al + (size_t)t * P, z + (size_t)t * P,
                     w->A_log, w->dt_bias, H, D, c->gate_lb);
    }

    /* 6. recurrence, per head, with q pre-scaled by d_k^-0.5 */
    float *S = state;
    float *Sown = NULL;
    if (!S) {
        /* Dereferenced at a computed offset immediately below; an unchecked NULL here
         * is a wild write, not a missing result. */
        Sown = (float *)calloc((size_t)H * D * D, sizeof(float));
        if (!Sown) k3_fatal_oom("KDA recurrent state", (size_t)H * D * D * sizeof(float));
        S = Sown;
    }
    const float qscale = 1.0f / sqrtf((float)D);
    /* Heads are independent: each reads and writes only its own S block, its own D-wide
     * slice of q/k/v/al/o, and its own beta column. The recurrence is sequential in t
     * WITHIN a head, so the loops nest head-outer here and each head walks its own t in
     * order; per-head arithmetic is untouched and the results are bit-identical to the
     * serial form (gated by test_ops' kda fixtures under 1 vs N threads). The recurrence
     * is 0.4% of FLOPs but, serial, it is a majority of non-matmul wall time at high
     * core counts. wr is a full P-wide work row, so wr + h*D gives each head a private
     * slice with no new allocation. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < H; h++) {
        float *wh = wr + (size_t)h * D;
        for (int t = 0; t < T; t++) {
            const size_t off = (size_t)t * P + (size_t)h * D;
            for (int i = 0; i < D; i++) wh[i] = q[off + i] * qscale;
            k3_kda_step(S + (size_t)h * D * D, o + off, wh, k + off, v + off,
                        al + off, bt[(size_t)t * H + h], D, D);
        }
    }

    /* 7/8/9. head-wise RMSNorm, THEN the gate, THEN the output projection */
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * E;
        float *ot = o + (size_t)t * P;
        for (int h = 0; h < H; h++)
            k3_rmsnorm(ot + (size_t)h * D, ot + (size_t)h * D, w->o_norm, D, c->rms_eps);
        k3_mmw(gb, xt, w->g, w->wdt, E, P);
        for (int i = 0; i < P; i++) ot[i] *= sigmoidf_(gb[i]);
        k3_mmw(out + (size_t)t * E, ot, w->o, w->wdt, P, E);
    }
    free(Sown);
}

/* ----------------------------------------------------------- decoder layer ---- */
size_t k3_layer_scratch(const K3Cfg *c, int T)
{
    size_t a = k3_mla_scratch(c, T);
    size_t b = k3_kda_scratch(c, T);
    size_t m = k3_moe_scratch(c);
    size_t sub = a > b ? a : b;
    if (m > sub) sub = m;
    /* prefix_sum, tmp, fold vectors, one attn_res source stack, plus the sub-block */
    return (size_t)3 * T * c->hidden
         + (size_t)2 * c->hidden
         + (size_t)(c->n_layers / c->attn_res_block + 2) * c->hidden
         + (size_t)2 * c->dense_inter
         + sub;
}

/* The incremental form. Everything except MLA already carries its own state:
 *   - k3_kda_layer updates the recurrent matrix and the ShortConv history in place, so
 *     a decode step simply does not clear them.
 *   - The attn-res block stack is built per token from that token's own hidden states,
 *     with no dependency on earlier tokens, so T=1 needs nothing carried.
 * Only softmax attention has to see every earlier position, which is why the KV cache
 * exists and why it is threaded through here rather than hidden inside k3_mla.
 *
 * kvc == NULL gives exactly the behaviour k3_decoder_layer has always had. */
void k3_decoder_layer_inc(float *h, float *block_residual, int *n_blocks,
                          const K3LayerW *w, const K3Cfg *c, int layer_idx,
                          int T, float *state, float *scratch,
                          float *kvc, float *ropec, int cached, int cap)
{
    const int E = c->hidden;
    const int maxb = c->n_layers / c->attn_res_block + 2;

    float *pref   = scratch;                    /* [T][E] the running residual   */
    float *tmp    = pref + (size_t)T * E;       /* [T][E] module output          */
    float *hin    = tmp  + (size_t)T * E;       /* [T][E] normalised layer input */
    float *foldA  = hin  + (size_t)T * E;       /* [E] attention aggregator      */
    float *foldM  = foldA + E;                  /* [E] mlp aggregator            */
    float *src    = foldM + E;                  /* [maxb+1][E] source stack      */
    float *dgu    = src + (size_t)(maxb) * E;   /* [2*dense_inter]               */
    float *sub    = dgu + (size_t)2 * c->dense_inter;

    /* The norm gain and the scoring projection collapse to ONE vector. Folding them
     * here costs 2*hidden multiplies per layer; a real engine folds at load time. */
    for (int i = 0; i < E; i++) {
        foldA[i] = w->attn_res_norm[i] * w->attn_res_proj[i];
        foldM[i] = w->mlp_res_norm[i]  * w->mlp_res_proj[i];
    }

    memcpy(pref, h, (size_t)T * E * sizeof(float));
    int have_prefix = 1;                        /* mirrors "prefix_sum is not None" */

    /* aggregation before attention, only when snapshots already exist */
    if (*n_blocks > 0) {
        for (int t = 0; t < T; t++) {
            for (int b = 0; b < *n_blocks; b++)
                memcpy(src + (size_t)b * E,
                       block_residual + ((size_t)t * maxb + b) * E,
                       (size_t)E * sizeof(float));
            memcpy(src + (size_t)(*n_blocks) * E, pref + (size_t)t * E,
                   (size_t)E * sizeof(float));
            k3_attn_res(h + (size_t)t * E, src, foldA, *n_blocks + 1, E, c->rms_eps);
        }
    }

    /* block boundary: snapshot the running residual, then CLEAR it */
    if (layer_idx % c->attn_res_block == 0) {
        for (int t = 0; t < T; t++)
            memcpy(block_residual + ((size_t)t * maxb + *n_blocks) * E,
                   pref + (size_t)t * E, (size_t)E * sizeof(float));
        (*n_blocks)++;
        have_prefix = 0;
    }

    /* attention */
    for (int t = 0; t < T; t++)
        k3_rmsnorm(hin + (size_t)t * E, h + (size_t)t * E, w->in_norm, E, c->rms_eps);
    if (w->kda) k3_kda_layer(tmp, hin, w->kda, c, T, state, sub);
    else        k3_mla_cached(tmp, hin, w->mla, c, T, sub, kvc, ropec, cached, cap);

    if (have_prefix) for (size_t i = 0; i < (size_t)T * E; i++) pref[i] += tmp[i];
    else             { memcpy(pref, tmp, (size_t)T * E * sizeof(float)); have_prefix = 1; }

    /* aggregation before the MLP. NO emptiness guard in the reference. */
    for (int t = 0; t < T; t++) {
        for (int b = 0; b < *n_blocks; b++)
            memcpy(src + (size_t)b * E,
                   block_residual + ((size_t)t * maxb + b) * E,
                   (size_t)E * sizeof(float));
        memcpy(src + (size_t)(*n_blocks) * E, pref + (size_t)t * E,
               (size_t)E * sizeof(float));
        k3_attn_res(h + (size_t)t * E, src, foldM, *n_blocks + 1, E, c->rms_eps);
    }

    for (int t = 0; t < T; t++)
        k3_rmsnorm(hin + (size_t)t * E, h + (size_t)t * E, w->post_norm, E, c->rms_eps);

    if (w->moe) {
        int   idx[K3_MAX_TOPK]; float wt[K3_MAX_TOPK];
        /* Prefill batches (T > 1, streamed source) fetch each unique expert once for
         * the whole chunk; decode (T == 1) and the resident path fall straight through
         * to k3_moe inside, byte-identical. */
        k3_moe_prefill(tmp, hin, w->moe, c, T, idx, wt, sub);
    } else {
        for (int t = 0; t < T; t++) {
            k3_mmw(dgu, hin + (size_t)t * E, w->dense_gate, w->wdt, E, c->dense_inter);
            k3_mmw(dgu + c->dense_inter, hin + (size_t)t * E, w->dense_up, w->wdt,
                      E, c->dense_inter);
            k3_situ_glu(sub, dgu, c->dense_inter, c->situ_b1, c->situ_b2);
            k3_mmw(tmp + (size_t)t * E, sub, w->dense_down, w->wdt, c->dense_inter, E);
        }
    }

    for (size_t i = 0; i < (size_t)T * E; i++) pref[i] += tmp[i];
    memcpy(h, pref, (size_t)T * E * sizeof(float));
}

void k3_decoder_layer(float *h, float *block_residual, int *n_blocks,
                      const K3LayerW *w, const K3Cfg *c, int layer_idx,
                      int T, float *state, float *scratch)
{
    k3_decoder_layer_inc(h, block_residual, n_blocks, w, c, layer_idx, T, state,
                         scratch, NULL, NULL, 0, 0);
}

/* ---------------------------------------------------------------- MXFP4 ---- */
/* OCP MX E2M1: index by the 4-bit code; bit 3 is the sign. */
static const float K3_E2M1[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* y[out] = W[out][in] . x[in], with W stored as bf16 and widened on read.
 *
 * WHY THIS EXISTS
 *   The checkpoint ships the dense trunk as bf16. Holding it as fp32 would double the
 *   resident set and double the weight traffic per token. These kernels are bandwidth
 *   bound, so halving the bytes read makes them faster, not slower, the same reason
 *   the MXFP4 path beats dequantise-then-multiply.
 *
 * WHY IT LOSES NOTHING
 *   bf16 is what the checkpoint already contains. Widening bf16 to fp32 is a pure left
 *   shift by 16 bits with no rounding, so multiplying from bf16 storage computes with
 *   exactly the values an fp32 copy would have supplied. The only difference from
 *   k3_matmul is WHERE the widening happens, not what is widened.
 *
 * The accumulator layout mirrors k3_matmul deliberately, four partial sums in double,
 * reduced in the same order, so the two kernels agree to the bit on identical input.
 *
 * THE AVX2 PATH IS BIT-IDENTICAL TO THE SCALAR PATH, not merely close. A __m256d holds
 * exactly four doubles, and loading four consecutive elements per iteration places
 * element i in lane i%4: the same partition as the scalar accumulators, with the same
 * sequential order within each lane. Reducing with (a0+a1)+(a2+a3) then reproduces the
 * scalar result exactly. Two details carry that guarantee:
 *
 *   - MUL THEN ADD, never _mm256_fmadd_pd. The build sets -ffp-contract=off, so the
 *     scalar code rounds the product and the sum separately while an FMA rounds once.
 *     Here the product happens to be exact (a bf16 widens exactly, and float x float
 *     needs 48 mantissa bits, which fits double's 53), so the two would agree anyway
 *     but that is a proof about the inputs. Mul-then-add is a proof about the code.
 *   - The scalar tail loop is reused verbatim for the in % 4 remainder.
 */
void k3_matmul_bf16(float *y, const float *x, const uint16_t *W, int in, int out)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) {
        const uint16_t *row = W + (size_t)o * in;
        int i = 0;
        double acc;
#if defined(__AVX2__)
        {
            /* Four vector accumulators, fused. _mm256_fmadd_pd per lane is the same
             * IEEE operation as scalar fma() in double, and the reduction below is
             * lane-for-lane the tree k3_matmul's sixteen scalar accumulators use, so
             * the two kernels remain BITWISE identical (test_ops asserts it). The old
             * one-accumulator mul+add form serialized on add latency at 4 elements
             * per ~4 cycles; this runs the memory-bound side of the roof instead. */
            __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
            __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();
            for (; i + 15 < in; i += 16) {
                /* bf16 -> f32 is a 16-bit left shift, so widen u16 to u32, shift,
                 * and reinterpret. No table, no rounding. */
                const __m128i h0 = _mm_loadl_epi64((const __m128i *)(row + i));
                const __m128i h1 = _mm_loadl_epi64((const __m128i *)(row + i + 4));
                const __m128i h2 = _mm_loadl_epi64((const __m128i *)(row + i + 8));
                const __m128i h3 = _mm_loadl_epi64((const __m128i *)(row + i + 12));
                v0 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h0), 16))),
                    _mm256_cvtps_pd(_mm_loadu_ps(x + i)), v0);
                v1 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h1), 16))),
                    _mm256_cvtps_pd(_mm_loadu_ps(x + i + 4)), v1);
                v2 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h2), 16))),
                    _mm256_cvtps_pd(_mm_loadu_ps(x + i + 8)), v2);
                v3 = _mm256_fmadd_pd(
                    _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h3), 16))),
                    _mm256_cvtps_pd(_mm_loadu_ps(x + i + 12)), v3);
            }
            /* (v0+v1)+(v2+v3) lanewise, then the same cross-lane pairing as scalar */
            const __m256d vt = _mm256_add_pd(_mm256_add_pd(v0, v1),
                                             _mm256_add_pd(v2, v3));
            double a[4];
            _mm256_storeu_pd(a, vt);
            acc = (a[0] + a[1]) + (a[2] + a[3]);
        }
#else
        {
            double a[16] = {0};
            for (; i + 15 < in; i += 16)
                for (int l = 0; l < 16; l++)
                    a[l] = fma((double)k3_bf16f(row[i + l]), (double)x[i + l], a[l]);
            double b0 = (a[0] + a[4]) + (a[8]  + a[12]);
            double b1 = (a[1] + a[5]) + (a[9]  + a[13]);
            double b2 = (a[2] + a[6]) + (a[10] + a[14]);
            double b3 = (a[3] + a[7]) + (a[11] + a[15]);
            acc = (b0 + b1) + (b2 + b3);
        }
#endif
        for (; i < in; i++) acc = fma((double)k3_bf16f(row[i]), (double)x[i], acc);
        y[o] = (float)acc;
    }
}

/* Per-row int8 matmul for the draft model: each row is [f32 scale][int8 * in]. The int8
 * weights are widened to float, dotted with the fp32 activation, and the row's scale is
 * applied once at the end. Unlike the trunk kernels this carries NO cross-path
 * determinism contract (K3_WI8 is draft-only, and the exact model decides every emitted
 * token), so it accumulates in float with fused products and the natural AVX2 reduction,
 * which is what makes it fast. */
void k3_matmul_q8(float *y, const float *x, const void *W, int in, int out)
{
    const unsigned char *base = (const unsigned char *)W;
    const size_t rowb = (size_t)4 + (size_t)in;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) {
        const unsigned char *row = base + (size_t)o * rowb;
        float scale;
        memcpy(&scale, row, 4);
        const int8_t *w = (const int8_t *)(row + 4);
        int i = 0;
        float acc;
#if defined(__AVX2__)
        {
            __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
            for (; i + 15 < in; i += 16) {
                const __m128i b0 = _mm_loadl_epi64((const __m128i *)(w + i));
                const __m128i b1 = _mm_loadl_epi64((const __m128i *)(w + i + 8));
                v0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)),
                                     _mm256_loadu_ps(x + i), v0);
                v1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1)),
                                     _mm256_loadu_ps(x + i + 8), v1);
            }
            __m256 vs = _mm256_add_ps(v0, v1);
            __m128 lo = _mm_add_ps(_mm256_castps256_ps128(vs),
                                   _mm256_extractf128_ps(vs, 1));
            lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
            lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
            acc = _mm_cvtss_f32(lo);
        }
#else
        {
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (; i + 3 < in; i += 4) {
                a0 += (float)w[i]     * x[i];
                a1 += (float)w[i + 1] * x[i + 1];
                a2 += (float)w[i + 2] * x[i + 2];
                a3 += (float)w[i + 3] * x[i + 3];
            }
            acc = (a0 + a1) + (a2 + a3);
        }
#endif
        for (; i < in; i++) acc += (float)w[i] * x[i];
        y[o] = acc * scale;
    }
}

/* A whole BYTE to its two E2M1 values, so the inner loop does one 8-byte load instead
 * of masking, shifting and two separate lookups. 2 KB, built once, shared by all
 * threads after initialisation. */
static float K3_E2M1_PAIR[256][2];
static int   k3_pair_ready = 0;

static void k3_pair_init(void)
{
    for (int b = 0; b < 256; b++) {
        K3_E2M1_PAIR[b][0] = K3_E2M1[b & 0x0F];   /* low nibble  = EVEN element */
        K3_E2M1_PAIR[b][1] = K3_E2M1[b >> 4];     /* high nibble = ODD element  */
    }
    k3_pair_ready = 1;
}

/* E8M0 byte to its power of two. 255 is NaN by spec and maps to zero. Precomputed
 * because ldexpf in the group loop is a function call the compiler will not inline
 * into a vectorised body. */
static float K3_E8M0[256];
static int   k3_e8m0_ready = 0;

static void k3_e8m0_init(void)
{
    for (int b = 0; b < 256; b++) K3_E8M0[b] = (b == 255) ? 0.0f : ldexpf(1.0f, b - 127);
    k3_e8m0_ready = 1;
}

/* y[rows] = W[rows][in] . x[in], with W read straight out of packed MXFP4 and never
 * materialised as floats. This is not an optimisation; it is what makes streaming
 * experts possible at all.
 *
 * One routed expert is 33,030,144 parameters. Dequantised to fp32 that is 132 MB, so
 * the 1,472 experts a single token touches would be 194 GB of materialised weights. As
 * packed nibbles the same expert is 17.55 MB. A matrix-vector product is memory bound,
 * so reading 7.5x fewer bytes makes this kernel FASTER than dequantising first.
 *
 * The loop is structured around the 32-element group because the scale is constant
 * within one: it factors out of the inner sum and is applied once per group instead of
 * once per element. At group 32 each group is exactly 16 packed bytes.
 *
 * PRECONDITIONS, none of these are checked, and violating them corrupts memory:
 *   - group <= 64. The expanded group goes into a fixed wf[64] stack buffer.
 *   - `in` is even. The packed row stride is in/2 bytes, two elements per byte.
 *   - `packed` is rows x (in/2) bytes; `scales` is rows x ceil(in/group) bytes.
 *   - A scale byte of 255 is NaN by the OCP MX spec and zeroes its whole group.
 *
 * ACCURACY CONTRACT. This kernel is deliberately NOT bit-identical to
 * dequantise-then-k3_matmul, and no caller should assume it is. It sums each group of
 * 32 and applies that group's scale before accumulating; dequantise-then-matmul sums
 * every term of the row under one set of accumulators. The orders differ.
 *
 * The difference is bounded and tiny. Every individual product is EXACT in double, an
 * E2M1 value carries 3 mantissa bits and x carries 24, so the product needs 27 of the
 * 53 available, so only the additions round, and reassociating exact terms moves the
 * result by roughly 1 ULP of double, order 1e-16 relative. The required agreement is
 * 1e-6 against dequantise-then-matmul on real checkpoint weights, gated by
 * tests/unit/test_expert.c. The margin is nine orders of magnitude.
 */
void k3_matmul_mxfp4(float *y, const float *x, const unsigned char *packed,
                     const unsigned char *scales, int in, int rows, int group)
{
    const int pcols = in / 2;                     /* two elements per byte */
    const int ngrp  = (in + group - 1) / group;
    const int gbyte = group / 2;

    if (!k3_pair_ready)  k3_pair_init();
    if (!k3_e8m0_ready)  k3_e8m0_init();

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (rows > 64)
#endif
    for (int r = 0; r < rows; r++) {
        const unsigned char *pr = packed + (size_t)r * pcols;
        const unsigned char *sr = scales + (size_t)r * ngrp;
        double acc = 0.0;

        for (int g = 0; g < ngrp; g++) {
            const unsigned char sb = sr[g];
            if (sb == 255) continue;              /* NaN scale: contribute nothing */
            const unsigned char *pb = pr + (size_t)g * gbyte;
            const float *xg = x + (size_t)g * group;

            int n = in - g * group;
            if (n > group) n = group;

            /* Expand the group to floats first, then take a plain dot product. The
             * split exists so the second loop can vectorise, which it cannot do while
             * a table lookup sits in the middle of the accumulation. */
            float wf[64];                         /* group is 32 for K3; 64 is headroom */
            const int half = n >> 1;
            for (int j = 0; j < half; j++) {
                const float *pv = K3_E2M1_PAIR[pb[j]];
                wf[2 * j]     = pv[0];
                wf[2 * j + 1] = pv[1];
            }
            if (n & 1) wf[n - 1] = K3_E2M1_PAIR[pb[half]][0];

            /* Four double lanes partitioned by i%4, reduced as (s0+s1)+(s2+s3), in BOTH
             * the scalar and the AVX2 path so the two agree bit for bit on every
             * machine. The split is written out rather than left to the compiler
             * because a sequential floating-point reduction may not be reassociated
             * without -ffast-math, which this build does not set: expressed as one
             * serial accumulator, the hottest loop in the engine compiles to scalar
             * adds no matter what the surrounding code looks like.
             *
             * The lane split changes the summation order. See the accuracy contract on
             * the function above for why that is bounded at ~1e-16 relative. */
            /* Two fused vector accumulators over the 32-element group, element i to
             * accumulator (i/4)%2, lanewise-summed then cross-lane paired; the scalar
             * path is the identical partition with fma(), which is the same IEEE
             * operation, so both paths stay bit-identical. Same reasoning as the
             * fp32/bf16 kernels above; the group is short, so two accumulators
             * suffice to break the add-latency chain. */
            double sub;
            int i = 0;
#if defined(__AVX2__)
            {
                __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
                for (; i + 7 < n; i += 8) {
                    v0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(wf + i)),
                                         _mm256_cvtps_pd(_mm_loadu_ps(xg + i)), v0);
                    v1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(wf + i + 4)),
                                         _mm256_cvtps_pd(_mm_loadu_ps(xg + i + 4)), v1);
                }
                double a[4];
                _mm256_storeu_pd(a, _mm256_add_pd(v0, v1));
                sub = (a[0] + a[1]) + (a[2] + a[3]);
            }
#else
            {
                double s[8] = {0};
                for (; i + 7 < n; i += 8)
                    for (int l = 0; l < 8; l++)
                        s[l] = fma((double)wf[i + l], (double)xg[i + l], s[l]);
                double b0 = s[0] + s[4], b1 = s[1] + s[5];
                double b2 = s[2] + s[6], b3 = s[3] + s[7];
                sub = (b0 + b1) + (b2 + b3);
            }
#endif
            for (; i < n; i++) sub = fma((double)wf[i], (double)xg[i], sub);
            acc += sub * (double)K3_E8M0[sb];
        }
        y[r] = (float)acc;
    }
}

void k3_mxfp4_dequant(float *out, const unsigned char *packed,
                      const unsigned char *scales, int rows, int pcols, int group)
{
    const int width = pcols * 2;                  /* logical elements per row */
    const int ngrp  = (width + group - 1) / group;

    for (int r = 0; r < rows; r++) {
        const unsigned char *pr = packed + (size_t)r * pcols;
        const unsigned char *sr = scales + (size_t)r * ngrp;
        float *orow = out + (size_t)r * width;

        for (int g = 0; g < ngrp; g++) {
            /* E8M0: a bare biased exponent. 255 is NaN by spec; map it to zero so one
             * bad byte cannot poison the row. ldexpf is exact for powers of two. */
            const unsigned char sb = sr[g];
            const float mult = (sb == 255) ? 0.0f : ldexpf(1.0f, (int)sb - 127);

            const int lo = g * group;
            int hi = lo + group;
            if (hi > width) hi = width;

            for (int i = lo; i < hi; i++) {
                const unsigned char byte = pr[i >> 1];
                /* low nibble = EVEN element. Reversing this gives right values in
                 * wrong places, which every statistical check would pass. */
                const unsigned char nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
                orow[i] = K3_E2M1[nib] * mult;
            }
        }
    }
}
