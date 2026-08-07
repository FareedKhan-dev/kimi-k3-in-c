/* SPDX-License-Identifier: Apache-2.0 */
/*
 * k3.h, Kimi K3 inference engine: public configuration and core types.
 *
 * OVERVIEW
 *   Kimi K3 is a 2.78-trillion-parameter mixture-of-experts model. This engine runs it
 *   on a single CPU by treating memory as a dial rather than a floor:
 *
 *     - the dense trunk (108.81 GB) is either held resident or streamed from disk in a
 *       fixed layer order, so the next read is always known in advance;
 *     - the 1.45 TB of routed experts are never resident. They stream on demand and are
 *       multiplied straight out of their packed MXFP4 form, never widened to fp32.
 *
 * THE THREE WEIGHT FIGURES, since they are easy to confuse
 *   108.81 GB   the 93 per-layer trunk runs, at bf16. Streamable, and re-read IN FULL
 *               on every token, which is why docs/TUNING.md says to feed the trunk
 *               before the expert cache.
 *     4.70 GB   embed and lm_head. Always resident; not part of the streamed trunk.
 *   113.49 GB   the two together: 56,743,648,000 always-active parameters at bf16, the
 *               figure tools/budget.py reports from the shard headers. Doubles to
 *               ~227 GB if anything widens this to fp32, which is why nothing does.
 *   1.45 TB     the routed experts, at MXFP4. Never resident at any budget.
 *
 *   The practical consequence is that the model runs in 8 GB of RAM and in 224 GB, and
 *   produces byte-identical output at both. See docs/PERFORMANCE.md.
 *
 * ARCHITECTURE (all values verified against the released config.json)
 *   93 layers: 69 Kimi Delta Attention (KDA) + 24 Gated MLA, plus one dense layer.
 *   Hidden 7168, 96 heads, 896 routed experts with top-16 selection and 2 shared,
 *   latent width 3584, SiTU-GLU activation, MXFP4 expert weights.
 *   docs/ARCHITECTURE.md maps each of these onto the technical report.
 *
 * THREE INVARIANTS THAT MUST HOLD
 *   Each of these is a place where a plausible-looking implementation produces a model
 *   that runs, emits fluent text, and is wrong. Each is gated by a fixture in
 *   tests/fixtures/ops chosen so that getting it wrong changes the output, and each is
 *   restated at its point of use.
 *
 *   1. A_log is indexed PER HEAD, not per channel. The checkpoint ships head_dim floats
 *      but only the first num_heads are meaningful; the remainder are padding.
 *      Gated by the kda_decay fixture, whose A_log is a linspace, so a per-channel
 *      misindex moves every element.
 *   2. MLA uses NoPE, yet the 64 rope dimensions still exist and are still cached.
 *      Only the rotation is absent. Dropping the slots changes the head width.
 *      Gated by the mla fixture, which asserts the softmax scale is over the FULL head
 *      width qk_nope + qk_rope, not over qk_nope alone.
 *   3. The MoE routing bias steers SELECTION only. Combining weights come from the
 *      UNBIASED sigmoid scores.
 *      Gated by the router fixture, whose bias reorders the top-k on 5 of its 6 rows.
 *
 * NOT INVARIANTS OF THIS IMPLEMENTATION
 *   Earlier revisions of this header also listed the UT-transform inverse (I + Akk)^-1
 *   and the retention of Aqk's diagonal but not Akk's. Those describe the chunked
 *   parallel form of the delta rule. This engine does not use it: k3_kda_step runs the
 *   naive sequential O(T) recurrence, one position at a time, and so does the reference
 *   in tools/k3_ref.py. Neither matrix is ever formed, so there was nothing to get right
 *   and no test could have caught getting it wrong. They are recorded in
 *   docs/ARCHITECTURE.md as properties of the algorithm, which is where a claim the code
 *   does not make belongs. Anyone adding a chunked KDA path must reinstate them here
 *   together with the fixtures that gate them.
 */
#ifndef K3_H
#define K3_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------- config ---- */
typedef struct {
    int hidden;            /* 7168  */
    int n_layers;          /* 93    */
    int vocab;             /* 163840 */
    float rms_eps;         /* 1e-5  */

    /* Kimi Delta Attention. 69 of the 93 layers. */
    int kda_heads;         /* 96    */
    int kda_head_dim;      /* 128, and d_k == d_v */
    int conv_k;            /* 4, depthwise, causal, SiLU fused */
    float gate_lb;         /* -5.0, the decay lower bound */

    /* Gated MLA. 24 of the 93 layers. */
    int n_heads;           /* 96    */
    int q_lora;            /* 1536  */
    int kv_lora;           /* 512   */
    int qk_nope;           /* 128   */
    int qk_rope;           /* 64, PRESENT BUT NEVER ROTATED */
    int v_head;            /* 128   */
    int mla_out_gate;      /* 1     */

    /* Stable LatentMoE. 92 of the 93 layers. */
    int n_experts;         /* 896   */
    int topk;              /* 16    */
    int n_shared;          /* 2, full width, added UNWEIGHTED */
    int latent;            /* 3584, the routed-expert width */
    int moe_inter;         /* 3072  */
    float routed_scale;    /* 1.0   */
    int moe_renorm;        /* 1     */
    int latent_norm;       /* 1, RMSNorm on the AGGREGATE, not per expert */

    /* the single dense layer, layer 0 */
    int first_dense;       /* 1     */
    int dense_inter;       /* 33792 */

    /* Block Attention Residuals */
    int attn_res_block;    /* 12. Boundaries fire when layer_idx % this == 0. */

    /* SiTU-GLU. The sigmoid takes the UNCAPPED gate. Bound is b1*b2 = 100. */
    float situ_b1;         /* 4.0   */
    float situ_b2;         /* 25.0  */

    /* Layer map. The released config lists these ONE-BASED and
     * configuration_kimi_k3.py:152-156 tests (layer_idx + 1), so zero-based MLA
     * layers are 3,7,11,...,87,91,92 and layers 91 and 92 are BOTH MLA. */
    int  n_full_attn;
    int *full_attn;        /* one-based layer indices */
} K3Cfg;

int  k3_is_mla(const K3Cfg *c, int layer);   /* layer is ZERO-based */
int  k3_is_kda(const K3Cfg *c, int layer);
int  k3_is_dense(const K3Cfg *c, int layer);

/* ------------------------------------------------------------------ ops ---- */

/* y = w * x / sqrt(mean(x^2) + eps). Accumulate in double: the reference upcasts
 * to float32 from bf16 and sums 7168 terms. eps is INSIDE the rsqrt. */
void k3_rmsnorm(float *y, const float *x, const float *w, int n, float eps);

/* SiTU-GLU over a 2*n input laid out as [gate | up].
 *   a  = b1 * tanh(gate / b1) * sigmoid(gate)     sigmoid sees the UNCAPPED gate
 *   u  = b2 * tanh(up / b2)
 *   y  = a * u                                    |y| <= b1*b2
 * modeling_kimi_linear.py:75-82 */
void k3_situ_glu(float *y, const float *x, int n, float b1, float b2);

/* Causal depthwise convolution with a fused SiLU.
 * w is [channels][k]; state holds the (k-1) previous inputs per channel and is
 * UPDATED IN PLACE. Passing state = NULL treats history as zero (start of sequence).
 * This state is a second piece of per-sequence memory beyond the recurrent matrix
 * and is the piece a decode loop forgets. modeling_kimi_linear.py:504-518 */
void k3_shortconv(float *y, const float *x, const float *w, float *state,
                  int channels, int k, int T);

/* Decay chain, per head h and channel d:
 *   z     = f_b(f_a(x))[h][d] + dt_bias[h*D + d]
 *   g     = lb * sigmoid(exp(A_log[h]) * z)      in (lb, 0]
 *   alpha = exp(g)                                in (e^lb, 1]
 * A_log is indexed BY HEAD. In float32 sigmoid underflows to 0 once its argument
 * falls below about -87.3, so alpha == 1.0 exactly is legitimate saturation
 * meaning perfect retention, not an error. fla/ops/kda/gate.py:60-69 */
void k3_kda_decay(float *g, float *alpha, const float *z, const float *A_log,
                  const float *dt_bias, int H, int D, float lb);

/* One KDA recurrence step for one head. S is [d_k][d_v], row-major.
 * ORDER IS LOAD BEARING (fla/ops/kda/naive.py:59-63):
 *    1. decay   S[i][:] *= alpha[i]
 *    2. read    u = S^T k
 *    3. write   S += k (beta*(v-u))^T
 *    4. output  o = S^T q          from the ALREADY UPDATED state
 * q must arrive pre-scaled by d_k^-0.5. */
void k3_kda_step(float *S, float *o, const float *q, const float *k,
                 const float *v, const float *alpha, float beta, int dk, int dv);

/* y[out] = W[out][in] . x[in].  W is row-major, no bias anywhere in this model. */
void k3_matmul(float *y, const float *x, const float *W, int in, int out);

/* Gated MLA, NoPE. One full sequence, no cache. modeling_kimi_linear.py:405-474.
 *
 * Shapes, with H heads:
 *   q  = q_b(q_a_norm(q_a(x)))            [T, H, qk_nope + qk_rope]
 *   c  = kv_a(x)                          [T, kv_lora + qk_rope]  ONE projection
 *   kv = kv_b(kv_a_norm(c[:kv_lora]))     [T, H, qk_nope + v_head]
 *   k_rope = c[kv_lora:]                  [T, qk_rope]  ONE head, broadcast to all
 *
 * NoPE: no rotation is ever applied, yet the qk_rope slots STILL EXIST and are
 * still concatenated onto both query and key. Dropping them changes the head width
 * from 192 to 128 and silently produces a different model.
 * modeling_kimi_linear.py:396 asserts use_nope; :403 sets rotary_emb = None.
 *
 * The softmax scale is (qk_nope + qk_rope)^-0.5, i.e. 192^-0.5, NOT qk_nope^-0.5.
 * modeling_kimi_linear.py:359.
 *
 * Causality is the caller's responsibility in the reference implementation, which
 * applies a mask only when one is supplied. It is applied unconditionally here: this is
 * a decoder-only engine and an unmasked position would leak future context.
 *
 * The output gate multiplies the attention output BEFORE o_proj, with no norm,
 * unlike KDA which norms first then gates. :470-473.
 *
 * scratch must hold at least
 *     T*H*(qk_nope+qk_rope)      q
 *   + T*H*(qk_nope+v_head)       kv
 *   + T*(kv_lora+qk_rope)        compressed latent plus the shared rope slot
 *   + q_lora                     transient
 *   + 2*H*v_head                 attention accumulator and gate buffer
 *   + T                          scores
 * floats. Use k3_mla_scratch() rather than recomputing this.
 */
typedef struct {
    /* Tagged by wdt: fp32 when zero (every fixture), bf16 for the real checkpoint. */
    const void  *q_a, *q_b;
    const void  *kv_a, *kv_b;
    const void  *o, *g;                 /* g may be NULL when the gate is disabled */
    const float *q_a_norm, *kv_a_norm;  /* elementwise in k3_rmsnorm: stays fp32   */
    int          wdt;
} K3MlaW;

size_t k3_mla_scratch(const K3Cfg *c, int T);
/* Scratch when a KV cache supplies the keys and values. cap is the cache capacity. */
size_t k3_mla_scratch_cached(const K3Cfg *c, int T, int cap, int cached_mode);

/* MLA with an optional KV cache. kvc is [cap][n_heads*(qk_nope+v_head)] and ropec is
 * [cap][qk_rope]; pass NULL for both to get the self-contained behaviour k3_mla has.
 * See the definition for why the EXPANDED keys are cached rather than the latent. */
void   k3_mla_cached(float *out, const float *x, const K3MlaW *w, const K3Cfg *c,
                     int T, float *scratch,
                     float *kvc, float *ropec, int cached, int cap);
void   k3_mla(float *out, const float *x, const K3MlaW *w, const K3Cfg *c,
              int T, float *scratch);

/* MoE routing, one token. modeling_kimi_linear.py:703-759.
 *
 *   logits = W x                      float32, no bias, W is [n_experts, hidden]
 *   scores = sigmoid(logits)          independent; they do NOT sum to 1
 *   sel    = topk(scores + bias)      the frozen bias steers SELECTION ONLY
 *   w      = scores[sel]              gathered from the UNBIASED scores
 *   w     /= sum(w) + 1e-20           when renorm
 *   w     *= routed_scale
 *
 * Reading the weights from the biased scores instead is the classic silent error:
 * it still routes to the same experts and only perturbs the mixture.
 *
 * The router reads the FULL hidden width, before the latent down-projection.
 * Grouped routing is dead code for K3 (num_expert_group == 1) and is not implemented.
 *
 * idx and w are written in DESCENDING score order. Fills at most topk entries.
 */
void k3_router(int *idx, float *w, const float *x, const float *W,
               const float *bias, int hidden, int n_experts, int topk,
               int renorm, float routed_scale);

/* AttnRes aggregation over nsrc sources of width n.
 *   keys   = RMSNorm(sources)            normalised
 *   score  = dot(key, fold)              fold = norm.weight * proj.weight, ONE vector
 *   out    = softmax(score) @ sources    the RAW, UNNORMALISED sources
 * Fold the two weight vectors at load time. modeling_kimi_linear.py:1075-1088 */
void k3_attn_res(float *out, const float *src, const float *fold,
                 int nsrc, int n, float eps);

/* ---- weight storage format -------------------------------------------------------
 * The always-active weights ship as bf16 and total 113.49 GB; held as fp32 they are
 * ~227 GB (see the three figures at the top of this file). Since these
 * kernels are bandwidth bound, storing bf16 and widening inside the matmul both halves
 * the memory and should run faster. Small vectors (norms, biases, A_log, dt_bias, the
 * conv kernels) stay fp32: together they are well under 0.1% of the bytes, and several
 * are read ELEMENTWISE rather than through a matmul, where a silent type change would
 * be read as garbage.
 *
 * A weight matrix is therefore a tagged pointer. K3_WF32 is zero, so a struct that has
 * been memset keeps the fp32 behaviour every existing fixture and the oracle rely on.
 */
/* K3_WI8 is a per-row int8 weight matrix used ONLY by the hybrid draft model, whose job
 * is to propose tokens that the exact bf16 model then verifies. Its output is therefore
 * never emitted directly and it carries no exactness contract, which is what lets it use
 * a fast, non-deterministic kernel. Each row is stored inline as [f32 scale][int8 * in],
 * so a matrix stays a single tagged pointer. Never tagged on the exact model. */
enum { K3_WF32 = 0, K3_WBF16 = 1, K3_WI8 = 2 };

/* bf16 -> f32 is a pure left shift: bf16 IS the top 16 bits of an f32. No rounding,
 * no table, no exponent rebias. */
static inline float k3_bf16f(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

void k3_matmul_bf16(float *y, const float *x, const uint16_t *W, int in, int out);

/* Per-row int8 matmul for the draft model. W is `out` rows of [f32 scale][int8 * in].
 * No determinism contract (see K3_WI8): uses the fastest AVX2 form available. */
void k3_matmul_q8(float *y, const float *x, const void *W, int in, int out);

/* The one call every trunk matmul goes through. Dispatch is a predictable branch on a
 * per-layer flag, outside the inner loops, so it costs nothing measurable. */
static inline void k3_mmw(float *y, const float *x, const void *W, int wdt,
                          int in, int out)
{
    if (wdt == K3_WBF16)     k3_matmul_bf16(y, x, (const uint16_t *)W, in, out);
    else if (wdt == K3_WI8)  k3_matmul_q8(y, x, W, in, out);
    else                     k3_matmul(y, x, (const float *)W, in, out);
}

/* Byte stride of one row for a per-row int8 matrix: the f32 scale plus `in` int8 weights.
 * bf16 and fp32 are per-element and take the element form. */
static inline size_t k3_wsz(int wdt) { return wdt == K3_WBF16 ? 2u : 4u; }
static inline size_t k3_row_bytes(int wdt, int in)
{
    return wdt == K3_WI8 ? (size_t)4 + (size_t)in : (size_t)in * k3_wsz(wdt);
}

/* ZERO-INITIALISE EVERY WEIGHT STRUCT BEFORE FILLING IT. K3MoeW holds an optional
 * expert source, which is a function pointer; K3LayerW and K3MlaW hold pointers whose
 * NULL-ness selects a code path (dense versus MoE, gated versus ungated MLA). An
 * uninitialised stack struct therefore does not merely read wrong weights, it jumps to
 * a garbage address. memset first, then assign.
 *
 * Stable LatentMoE weights. Routed experts are stored contiguously:
 *   w1[expert][moe_inter][latent]   gate
 *   w3[expert][moe_inter][latent]   up
 *   w2[expert][latent][moe_inter]   down
 * The shared expert is ONE wider MLP at FULL width, intermediate moe_inter*n_shared.
 */

/* ---- streamed experts -------------------------------------------------------------
 * One routed expert as it sits in the cache: still MXFP4, never widened. A dequantised
 * expert is 132 MB against 17.55 MB packed, and a token touches 1,472 of them, so
 * widening on load would need 194 GB per token. k3_matmul_mxfp4 consumes these directly.
 */
typedef struct {
    const unsigned char *p1, *s1;        /* w1 gate, packed and E8M0 scales           */
    const unsigned char *p3, *s3;        /* w3 up                                     */
    const unsigned char *p2, *s2;        /* w2 down                                   */
} K3ExpertQ;

/* A source of experts. get() must leave the returned pointers valid until the caller
 * finishes the token; a cache satisfies that by pinning what the current token needs
 * before evicting anything. Returns 0 on success. */
typedef struct K3ExpertSrc {
    int (*get)(struct K3ExpertSrc *self, int layer, int expert, K3ExpertQ *out);
    /* OPTIONAL batch prefetch: bring all n experts resident, issuing their reads
     * concurrently. May be NULL, and callers MUST cope with that by falling back to
     * get() alone -- which is always correct, just slower.
     *
     * WHY IT EXISTS. k3_moe walks the top-16 calling get() one at a time, so each miss
     * is a blocking 17.55 MB pread and the drive sees one request in flight. On a
     * 92-layer decode that is 1,472 serial round trips. Handing the whole top-k over at
     * once lets the reads overlap, which is the difference between a queue depth of 1
     * and one of 16 on hardware that needs depth to reach its rated bandwidth.
     *
     * Returns the number brought resident, or -1. A short return is NOT fatal: get()
     * will simply miss on the remainder and read it the slow way.
     *
     * ZERO THIS FIELD. It is a function pointer in a struct that callers build on the
     * stack; an uninitialised one is a jump to garbage. See the warning below. */
    int (*getmany)(struct K3ExpertSrc *self, int layer, const int *experts, int n);
    /* OPTIONAL: 1 if the expert is already resident (get() would read no disk), filling
     * out when non-NULL. The draft model's cache-only routing uses this to propose tokens
     * with zero expert I/O. May be NULL; callers must cope. */
    int (*resident)(struct K3ExpertSrc *self, int layer, int expert, K3ExpertQ *out);
    void *ctx;
} K3ExpertSrc;

typedef struct {
    /* gate stays fp32 on purpose. k3_router carries its own inline matmul rather than
     * calling k3_matmul, so tagging it would change that function's signature and its
     * fixture, to save 1.18 GB out of about 114 GB. Not worth the churn.
     * w1/w3/w2 also stay fp32: that resident bank is indexed by pointer arithmetic and
     * is only ever used by the fixtures, because the real model streams experts
     * through src and multiplies them straight out of MXFP4. */
    const float *gate, *bias;            /* router: [n_experts][hidden], [n_experts] */
    const float *w1, *w3, *w2;           /* resident expert bank, fixtures only      */
    const float *latent_norm;            /* [latent], elementwise: stays fp32        */
    const void  *down, *up;              /* [latent][hidden], [hidden][latent]       */
    const void  *sh1, *sh3, *sh2;        /* shared expert, full width                */
    int          wdt;                    /* applies to down/up/sh1/sh3/sh2           */
    /* When src is non-NULL the resident w1/w3/w2 bank is ignored and experts are
     * fetched per token. layer identifies this layer to the source. Both default to
     * zero, so every existing caller keeps the resident path unchanged. */
    struct K3ExpertSrc *src;
    int          layer;
    /* Draft-only: when 1, route among ONLY the experts already resident in the cache and
     * renormalise the combining weights over them, so a draft token reads zero new expert
     * bytes. Never set on the exact model, whose output is authoritative; the draft merely
     * proposes and the exact model verifies. */
    int          cache_only;
} K3MoeW;

size_t k3_moe_scratch(const K3Cfg *c);

/* Number of routed experts that failed to load and were dropped from a MoE sum.
 * Non-zero means some token was computed with part of its routed contribution missing,
 * which is silent numerical corruption: the run still finishes and still prints a
 * plausible token. Callers MUST check this and fail. Defined in k3_ops.c. */
extern long k3_expert_drops;

/* Context ceilings, a sanity bound, not the binding constraint.
 *
 * Prompt, sequence and output buffers are heap-allocated and sized from the actual
 * request, so these limits exist only to reject implausible input early.
 *
 * The binding constraint is the MLA KV cache, which costs ~2.37 MB per position
 * (24 MLA layers x expanded k and v, fp32):
 *
 *       4,096 positions  ->     9.7 GB
 *      16,384 positions  ->    38.8 GB
 *      32,768 positions  ->    77.7 GB
 *     131,072 positions  ->   310.6 GB
 *   1,048,576 positions  ->  2485.0 GB     the model's advertised 1M context
 *
 * The CLI computes the requirement for the request it was given and refuses, with both
 * figures side by side, when it will not fit in available memory. Reaching the model's
 * full 1M context is a hardware question, not an engine limit.
 *
 * Note that long prompts are also bounded in practice by prefill cost: attention in the
 * MLA layers is quadratic in sequence length, and prefill is not yet chunked. See
 * docs/ROADMAP.md. */
#define K3_MAX_PROMPT 32768
#define K3_MAX_GEN     4096

/* Bytes of MLA KV cache per position, measured on the released checkpoint. */
#define K3_KV_BYTES_PER_POS 2370000.0

/* idx and wt must each hold topk entries. */
void   k3_moe(float *out, const float *x, const K3MoeW *w, const K3Cfg *c,
              int T, int *idx, float *wt, float *scratch);

/* Batched MoE for prefill over a chunk of T tokens: fetches each unique routed expert
 * from disk ONCE and reuses it across the chunk, cutting prefill expert I/O ~3-4x, with
 * per-token output bit-identical to k3_moe. Streamed source only (w->src != NULL); falls
 * back to k3_moe for the resident path or T <= 1. */
void   k3_moe_prefill(float *out, const float *x, const K3MoeW *w, const K3Cfg *c,
                      int T, int *idx, float *wt, float *scratch);

/* Kimi Delta Attention, one full layer, one sequence, no cache.
 * Verified against modeling_kimi_linear.py:543-663 and fla/ops/kda/naive.py.
 *
 * Order, and every step of it is load bearing:
 *   1. q,k,v = Linear(x)                          separate projections
 *   2. ShortConv on each, SiLU FUSED inside       :504-518
 *   3. L2Norm on q and k ONLY, never on v         Eq 2. sum of squares, not mean.
 *   4. beta = sigmoid(b_proj(x))                  PER-HEAD SCALAR, not per channel
 *   5. z = f_b(f_a(x)) + dt_bias                  ONE shared low-rank pair, all heads
 *      g = lb*sigmoid(exp(A_log[h]) * z)          A_log indexed BY HEAD
 *      alpha = exp(g)
 *   6. recurrence, q pre-scaled by d_k^-0.5       decay, delta write, read UPDATED
 *   7. head-wise RMSNorm on the output            over d_v, per head
 *   8. multiply by sigmoid(g_proj(x))             norm FIRST, then gate  :651-656
 *   9. o_proj
 *
 * Step 8 is the opposite order to MLA, which gates before its projection with no
 * norm at all. Sharing one code path between them is wrong.
 */
typedef struct {
    const void  *q, *k, *v;              /* [H*D][hidden] each              */
    /* The conv kernels, A_log and dt_bias are read ELEMENTWISE by k3_shortconv and
     * k3_kda_decay, not through a matmul, so they stay fp32. They are also tiny: the
     * three conv kernels are 49k values each and A_log is 96. */
    const float *q_conv, *k_conv, *v_conv; /* [H*D][conv_k] depthwise       */
    const void  *f_a, *f_b;              /* [D][hidden], [H*D][D]           */
    const float *A_log;                  /* [H]  PER HEAD                   */
    const float *dt_bias;                /* [H*D] per (head, channel)       */
    const void  *b;                      /* [H][hidden] -> per-head scalar  */
    const void  *g;                      /* [H*D][hidden] full-rank gate    */
    const float *o_norm;                 /* [D] head-wise norm gain         */
    const void  *o;                      /* [hidden][H*D]                   */
    int          wdt;                    /* q,k,v,g,o,f_a,f_b,b             */
} K3KdaW;

size_t k3_kda_scratch(const K3Cfg *c, int T);
/* state may be NULL (fresh sequence). If non-NULL it must hold
 * H*D*D recurrent floats followed by 3*H*D*(conv_k-1) convolution floats, and it is
 * UPDATED IN PLACE so a decode loop can carry it. */
void   k3_kda_layer(float *out, const float *x, const K3KdaW *w, const K3Cfg *c,
                    int T, float *state, float *scratch);

/* One decoder layer, reproducing _forward_attn_residual (modeling_kimi_linear.py
 * :984-1046) statement for statement.
 *
 *   prefix_sum = h
 *   if block_residual is NON-EMPTY:
 *       h = attn_res([blocks..., prefix_sum], self_attention_res)   <- REPLACES h
 *   if layer_idx % attn_res_block == 0:
 *       push prefix_sum onto block_residual
 *       prefix_sum = NONE                                           <- the reset
 *   h = input_layernorm(h)
 *   h = attention(h)
 *   prefix_sum = (prefix_sum == NONE) ? h : prefix_sum + h
 *   h = attn_res([blocks..., prefix_sum], mlp_res)                  <- UNCONDITIONAL
 *   h = post_attention_layernorm(h)
 *   h = moe(h) or dense_mlp(h)
 *   prefix_sum = (prefix_sum == NONE) ? h : prefix_sum + h
 *   return prefix_sum
 *
 * THE SUBTLETY: on a boundary layer the running residual is pushed into the snapshot
 * stack and then CLEARED, so it does NOT also survive as a separate softmax source
 * there. On every other layer it does. Getting this wrong is silent.
 *
 * The second aggregation has NO emptiness guard, unlike the first. That is safe only
 * because layer 0 is itself a boundary and has already pushed one snapshot.
 *
 * block_residual is [T][max_blocks][hidden]; *n_blocks is read and updated.
 */
typedef struct {
    const float *in_norm, *post_norm;      /* [hidden] */
    const float *attn_res_norm, *attn_res_proj;  /* [hidden] each, folded at load */
    const float *mlp_res_norm,  *mlp_res_proj;
    const K3KdaW *kda;                     /* exactly one of kda/mla is non-NULL */
    const K3MlaW *mla;
    const K3MoeW *moe;                     /* NULL on the dense layer */
    const void  *dense_gate, *dense_up, *dense_down;  /* used when moe is NULL */
    int          wdt;                    /* applies to the three dense_* only */
} K3LayerW;

size_t k3_layer_scratch(const K3Cfg *c, int T);
void   k3_decoder_layer(float *h, float *block_residual, int *n_blocks,
                        const K3LayerW *w, const K3Cfg *c, int layer_idx,
                        int T, float *state, float *scratch);

/* Incremental form: identical except that MLA attends over a KV cache of `cached`
 * earlier positions and appends its own. KDA and the attn-res stack need nothing
 * carried, because KDA updates its state in place and the block stack is per token.
 * Pass kvc = NULL for the full-recompute behaviour. */
void   k3_decoder_layer_inc(float *h, float *block_residual, int *n_blocks,
                            const K3LayerW *w, const K3Cfg *c, int layer_idx,
                            int T, float *state, float *scratch,
                            float *kvc, float *ropec, int cached, int cap);

/* ---------------------------------------------------------------- MXFP4 ---- */
/* Dequantise OCP MX FP4, the format Kimi K3 ships its routed experts in.
 *
 *   packed  [rows][pcols]      uint8, TWO 4-bit elements per byte
 *   scales  [rows][pcols*2/32] uint8, one E8M0 exponent per 32 elements
 *   out     [rows][pcols*2]    float32
 *
 *   value = E2M1[nibble] * 2^(scale - 127)
 *
 * E2M1 is 1 sign, 2 exponent, 1 mantissa, so exactly sixteen values:
 *   0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives.
 * E8M0 is a bare biased exponent; 255 is NaN by spec and is mapped to zero here so a
 * single bad byte cannot poison a whole row.
 *
 * NIBBLE ORDER IS A CONVENTION, NOT A RULE. The low nibble of each byte is the EVEN
 * element. Reversing it yields a matrix with exactly the right values in the wrong
 * places: every statistic looks perfect and the model is wrong. Verified against real
 * checkpoint bytes in fixtures/mxfp4.json, which also records the swapped result.
 *
 * Bytes on disk per element are 0.5 for the nibble plus 1/32 for the shared scale,
 * i.e. 0.53125, which is why one 33,030,144-parameter expert occupies 17,547,264
 * bytes exactly.
 */
/* config.json: quantization_config.group_size = 32. Named rather than spelled 32 at
 * each call site so a checkpoint that changed it fails at one place, and so k3_load.c
 * can check the shipped scale count against it per tensor. */
#define K3_MXFP4_GROUP 32

/* Upper bound on num_experts_per_token, and therefore on the fixed-size top-k arrays in
 * k3_decoder_layer and the cache's batch-prefetch work list. K3 selects 16.
 *
 * This is a HARD limit, not a hint: k3_router fills the caller's idx[] with cfg->topk
 * entries and those arrays are stack allocated, so a config with a larger top-k would
 * overflow them. k3_cfg.h validates topk against this so the failure is a clear message
 * at load time rather than a corrupted stack mid-decode. */
#define K3_MAX_TOPK 64

void k3_mxfp4_dequant(float *out, const unsigned char *packed,
                      const unsigned char *scales, int rows, int pcols, int group);

/* y[rows] = W[rows][in] . x[in], with W read directly as MXFP4. Never materialises the
 * fp32 matrix, which is what allows a streamed expert to stay 17.55 MB instead of
 * becoming 132 MB. See the comment on the definition. */
void k3_matmul_mxfp4(float *y, const float *x, const unsigned char *packed,
                     const unsigned char *scales, int in, int rows, int group);

#ifdef __cplusplus
}
#endif

#endif /* K3_H */
