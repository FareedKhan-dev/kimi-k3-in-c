/* k3_model.c - the full Kimi K3 forward pass, and the oracle gate.
 *
 * This is the point at which "every component passes" becomes "the engine is
 * correct". It loads the tiny checkpoint exported by tools/export_weights.py, runs
 * the complete 13-layer forward, and compares against fixtures/ref_k3.json, which
 * was produced by the pure-torch reference.
 *
 * Two things must match, and they test different code paths:
 *   tf_pred    argmax at every position of ONE teacher-forced forward. Exercises
 *              prefill: the chunked-equivalent path, AttnRes across the whole stack,
 *              and the KDA recurrence run over the full sequence in one call.
 *   full_ids   greedy decode. Exercises the same machinery re-entered per token.
 *
 * The bar is exact: 32/32 teacher-forced positions and 20/20 greedily decoded tokens
 * must match ref_k3.json. Not "close", the same integers. A single mismatch fails
 * the gate, because on a discrete argmax there is no such thing as a small error:
 * either the token is right or the sentence has diverged.
 *
 * Positions inside the PROMPT are expected to disagree: the model is randomly
 * initialised and the prompt is random, so it has no reason to predict its own
 * input. Only the GENERATED span is a real test, and it must be exact.
 */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "k3.h"
#include "k3_cfg.h"   /* one config reader for both shapes; never defaults a field */

/* ------------------------------------------------------------ weight store ---- */
typedef struct {
    float *blob;
    jval  *man;          /* the "tensors" object: name -> {offset, shape} */
    char  *arena;
    char  *raw;
} Store;

static char *slurp_(const char *p, long *len)
{
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); return NULL; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    b[n] = 0; fclose(f); if (len) *len = n;
    return b;
}

static int store_open(Store *s, const char *dir)
{
    char p[512];
    snprintf(p, sizeof p, "%s/tiny_k3.json", dir);
    s->raw = slurp_(p, NULL);
    if (!s->raw) return 0;
    jval *root = json_parse(s->raw, &s->arena);
    s->man = json_get(root, "tensors");

    snprintf(p, sizeof p, "%s/tiny_k3.bin", dir);
    long n = 0;
    char *bytes = slurp_(p, &n);
    if (!bytes) return 0;
    s->blob = (float *)bytes;
    return s->man != NULL;
}

/* Look a tensor up by name. Returns NULL when absent, which the caller must treat
 * as fatal: a silently-missing weight would be read as zeros and the model would
 * still run, producing plausible garbage. */
static const float *W(Store *s, const char *fmt, ...)
{
    char key[160];
    va_list ap; va_start(ap, fmt); vsnprintf(key, sizeof key, fmt, ap); va_end(ap);
    jval *e = json_get(s->man, key);
    if (!e) { fprintf(stderr, "MISSING WEIGHT: %s\n", key); return NULL; }
    jval *o = json_get(e, "offset");
    return s->blob + (size_t)o->num;
}


/* ---------------------------------------------------------------- forward ---- */
typedef struct {
    K3KdaW kda[64]; K3MlaW mla[64]; K3MoeW moe[64]; K3LayerW lay[64];
    float *w1[64], *w3[64], *w2[64];
    const float *embed, *lm_head, *final_norm;
    const float *out_res_norm, *out_res_proj;
} Model;

/* Pack per-expert tensors contiguously per layer so the MoE can index expert e at a
 * fixed stride, which is also how a streaming engine will lay them out on disk. */
static float *pack(Store *s, int layer, const char *which, int n, size_t per)
{
    float *B = (float *)malloc(per * (size_t)n * sizeof(float));
    if (!B) return NULL;
    for (int e = 0; e < n; e++) {
        const float *a = W(s, "layers_%d_mlp_experts_%d_%s_weight", layer, e, which);
        if (!a) { free(B); return NULL; }
        memcpy(B + (size_t)e * per, a, per * sizeof(float));
    }
    return B;
}

static int model_bind(Model *m, Store *s, const K3Cfg *c)
{
    m->embed        = W(s, "embed_tokens_weight");
    m->lm_head      = W(s, "lm_head_weight");
    m->final_norm   = W(s, "norm_weight");
    m->out_res_norm = W(s, "output_attn_res_norm_weight");
    m->out_res_proj = W(s, "output_attn_res_proj_weight");
    if (!m->embed || !m->lm_head || !m->final_norm) return 0;

    const size_t p13 = (size_t)c->moe_inter * (size_t)c->latent;
    const size_t p2  = (size_t)c->latent * (size_t)c->moe_inter;

    for (int L = 0; L < c->n_layers; L++) {
        K3LayerW *lw = &m->lay[L]; memset(lw, 0, sizeof *lw);
        lw->in_norm       = W(s, "layers_%d_input_layernorm_weight", L);
        lw->post_norm     = W(s, "layers_%d_post_attention_layernorm_weight", L);
        lw->attn_res_norm = W(s, "layers_%d_self_attention_res_norm_weight", L);
        lw->attn_res_proj = W(s, "layers_%d_self_attention_res_proj_weight", L);
        lw->mlp_res_norm  = W(s, "layers_%d_mlp_res_norm_weight", L);
        lw->mlp_res_proj  = W(s, "layers_%d_mlp_res_proj_weight", L);
        if (!lw->in_norm || !lw->mlp_res_proj) return 0;

        if (k3_is_mla(c, L)) {
            K3MlaW *w = &m->mla[L];
            w->q_a       = W(s, "layers_%d_self_attn_q_a_proj_weight", L);
            w->q_a_norm  = W(s, "layers_%d_self_attn_q_a_layernorm_weight", L);
            w->q_b       = W(s, "layers_%d_self_attn_q_b_proj_weight", L);
            w->kv_a      = W(s, "layers_%d_self_attn_kv_a_proj_with_mqa_weight", L);
            w->kv_a_norm = W(s, "layers_%d_self_attn_kv_a_layernorm_weight", L);
            w->kv_b      = W(s, "layers_%d_self_attn_kv_b_proj_weight", L);
            w->o         = W(s, "layers_%d_self_attn_o_proj_weight", L);
            w->g         = W(s, "layers_%d_self_attn_g_proj_weight", L);
            if (!w->q_a || !w->kv_b || !w->o) return 0;
            lw->mla = w;
        } else {
            K3KdaW *w = &m->kda[L];
            w->q       = W(s, "layers_%d_self_attn_q_proj_weight", L);
            w->k       = W(s, "layers_%d_self_attn_k_proj_weight", L);
            w->v       = W(s, "layers_%d_self_attn_v_proj_weight", L);
            w->q_conv  = W(s, "layers_%d_self_attn_q_conv1d_weight", L);
            w->k_conv  = W(s, "layers_%d_self_attn_k_conv1d_weight", L);
            w->v_conv  = W(s, "layers_%d_self_attn_v_conv1d_weight", L);
            w->f_a     = W(s, "layers_%d_self_attn_f_a_proj_weight", L);
            w->f_b     = W(s, "layers_%d_self_attn_f_b_proj_weight", L);
            w->A_log   = W(s, "layers_%d_self_attn_A_log", L);
            w->dt_bias = W(s, "layers_%d_self_attn_dt_bias", L);
            w->b       = W(s, "layers_%d_self_attn_b_proj_weight", L);
            w->g       = W(s, "layers_%d_self_attn_g_proj_weight", L);
            w->o_norm  = W(s, "layers_%d_self_attn_o_norm_weight", L);
            w->o       = W(s, "layers_%d_self_attn_o_proj_weight", L);
            if (!w->q || !w->f_b || !w->A_log || !w->o) return 0;
            lw->kda = w;
        }

        if (k3_is_dense(c, L)) {
            lw->dense_gate = W(s, "layers_%d_mlp_gate_proj_weight", L);
            lw->dense_up   = W(s, "layers_%d_mlp_up_proj_weight", L);
            lw->dense_down = W(s, "layers_%d_mlp_down_proj_weight", L);
            if (!lw->dense_gate) return 0;
        } else {
            K3MoeW *w = &m->moe[L];
            w->gate        = W(s, "layers_%d_mlp_gate_weight", L);
            w->bias        = W(s, "layers_%d_mlp_e_score_correction_bias", L);
            w->down        = W(s, "layers_%d_mlp_down_weight", L);
            w->up          = W(s, "layers_%d_mlp_up_weight", L);
            w->latent_norm = W(s, "layers_%d_mlp_norm_weight", L);
            w->sh1         = W(s, "layers_%d_mlp_shared_w1_weight", L);
            w->sh3         = W(s, "layers_%d_mlp_shared_w3_weight", L);
            w->sh2         = W(s, "layers_%d_mlp_shared_w2_weight", L);
            m->w1[L] = pack(s, L, "w1", c->n_experts, p13);
            m->w3[L] = pack(s, L, "w3", c->n_experts, p13);
            m->w2[L] = pack(s, L, "w2", c->n_experts, p2);
            w->w1 = m->w1[L]; w->w3 = m->w3[L]; w->w2 = m->w2[L];
            if (!w->gate || !w->w1 || !w->w2) return 0;
            lw->moe = w;
        }
    }
    return 1;
}

static void forward(Model *m, const K3Cfg *c, const int *ids, int T, float *logits,
                    float *scratch, float *h, float *br, float *kstate, int reuse_state)
{
    const int E = c->hidden;
    const int maxb = c->n_layers / c->attn_res_block + 2;
    const int P = c->kda_heads * c->kda_head_dim;
    const size_t kper = (size_t)P * (size_t)c->kda_head_dim
                      + (size_t)3 * (size_t)P * (size_t)(c->conv_k - 1);

    for (int t = 0; t < T; t++)
        memcpy(h + (size_t)t * E, m->embed + (size_t)ids[t] * E,
               (size_t)E * sizeof(float));

    memset(br, 0, (size_t)T * (size_t)maxb * (size_t)E * sizeof(float));
    if (!reuse_state) memset(kstate, 0, kper * (size_t)c->n_layers * sizeof(float));
    int nb = 0;
    for (int L = 0; L < c->n_layers; L++) {
        float *state = kstate + (reuse_state ? 0 : kper * (size_t)L);
        if (reuse_state) memset(state, 0, kper * sizeof(float));
        k3_decoder_layer(h, br, &nb, &m->lay[L], c, L, T,
                         state, scratch);
    }

    /* The MODEL-LEVEL aggregator. Beyond the 2 per layer there is one more:
     * output_attn_res_{norm,proj}. The tensor census counts exactly 1 of these.
     * Skipping it is silent. */
    if (m->out_res_norm && m->out_res_proj) {
        float *fold = scratch;
        float *src  = fold + E;
        for (int i = 0; i < E; i++) fold[i] = m->out_res_norm[i] * m->out_res_proj[i];
        for (int t = 0; t < T; t++) {
            for (int b = 0; b < nb; b++)
                memcpy(src + (size_t)b * E, br + ((size_t)t * maxb + b) * E,
                       (size_t)E * sizeof(float));
            memcpy(src + (size_t)nb * E, h + (size_t)t * E, (size_t)E * sizeof(float));
            k3_attn_res(h + (size_t)t * E, src, fold, nb + 1, E, c->rms_eps);
        }
    }

    float *nrm = scratch;
    for (int t = 0; t < T; t++) {
        k3_rmsnorm(nrm, h + (size_t)t * E, m->final_norm, E, c->rms_eps);
        k3_matmul(logits + (size_t)t * (size_t)c->vocab, nrm, m->lm_head, E, c->vocab);
    }
}

static int argmax_(const float *v, int n)
{ int b = 0; for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i; return b; }

/* Config is read through k3_cfg.h, which never substitutes a default for a missing
 * field: it collects every absent key and refuses the load. Do not reintroduce a
 * defaulting reader here. A default turns "this program cannot understand this config"
 * into "this program runs a different model and says nothing", which no test in this
 * file can detect, every fixture would still load, and every comparison would be
 * against the wrong architecture. */

int main(int argc, char **argv)
{
    const char *dir = (argc > 1) ? argv[1] : "../fixtures";

    /* Open BEFORE printing the banner. Printing it first made a failed run look like a
     * started one: the header and fixture path scrolled past, the error went to stderr,
     * and a captured transcript read as though the gates had been attempted. The first
     * argument is a DIRECTORY holding tiny_k3.json, tiny_k3.bin and ref_k3.json; passing
     * the .bin itself is the easy mistake, so say so. */
    Store st; memset(&st, 0, sizeof st);
    if (!store_open(&st, dir)) {
        fprintf(stderr, "GATE ABORTED: cannot load checkpoint from '%s'\n"
                        "  expected a DIRECTORY containing tiny_k3.json + tiny_k3.bin\n"
                        "  usage: k3_model <fixture_dir>\n", dir);
        return 2;
    }
    char p[512]; snprintf(p, sizeof p, "%s/ref_k3.json", dir);
    char *rtxt = slurp_(p, NULL);
    if (!rtxt) { fprintf(stderr, "GATE ABORTED: cannot read %s\n", p); return 2; }

    printf("Kimi K3 pure-C engine, full-model oracle gate\n");
    printf("fixtures: %s\n\n", dir);
    char *rar = NULL; jval *ref = json_parse(rtxt, &rar);

    jval *jc = json_get(ref, "config");
    jval *jp = json_get(ref, "prompt_ids");
    jval *jf = json_get(ref, "full_ids");
    jval *jt = json_get(ref, "tf_pred");
    if (!jc || !jp || !jf || !jt) { fprintf(stderr, "ref_k3.json incomplete\n"); return 2; }


    /* One reader for both config shapes, and it refuses to default a field it cannot
     * find. The previous inline block read FLAT names only, so pointing this gate at
     * the released nested config.json would have silently produced an all-KDA model
     * with correct-looking SiTU betas. See k3_cfg.h. */
    K3Cfg c; static int fabuf[128];
    if (!k3_cfg_load(&c, fabuf, 128, jc, p)) {
        fprintf(stderr, "GATE ABORTED: config in %s could not be read with confidence\n", p);
        return 2;
    }

    printf("layer map (0-based): ");
    for (int i = 0; i < c.n_layers; i++) printf("%s", k3_is_mla(&c, i) ? "M" : "K");
    printf("   (M=MLA, K=KDA; dense layer = 0)\n");
    printf("attn_res boundaries at: ");
    for (int i = 0; i < c.n_layers; i++) if (i % c.attn_res_block == 0) printf("%d ", i);
    printf("\n\n");

    printf("checkpoint: %d tensors loaded\n", st.man->len);
    printf("prompt_ids %d, full_ids %d, tf_pred %d\n\n", jp->len, jf->len, jt->len);

    Model *m = (Model *)calloc(1, sizeof(Model));
    if (!model_bind(m, &st, &c)) { fprintf(stderr, "weight binding FAILED\n"); return 3; }
    printf("all layer weights bound\n\n");

    const int T = jf->len, np = jp->len;
    int *full = (int *)malloc((size_t)T * sizeof(int));
    for (int i = 0; i < T; i++) full[i] = (int)jf->kids[i]->num;

    const int maxb = c.n_layers / c.attn_res_block + 2;
    const int P = c.kda_heads * c.kda_head_dim;
    const size_t kper = (size_t)P * (size_t)c.kda_head_dim
                      + (size_t)3 * (size_t)P * (size_t)(c.conv_k - 1);

    size_t need = k3_layer_scratch(&c, T);
    size_t alt  = (size_t)(maxb + 2) * (size_t)c.hidden + (size_t)c.vocab;
    if (alt > need) need = alt;

    float *scratch = (float *)malloc(need * sizeof(float));
    float *h    = (float *)malloc((size_t)T * (size_t)c.hidden * sizeof(float));
    float *br   = (float *)malloc((size_t)T * (size_t)maxb * (size_t)c.hidden * sizeof(float));
    float *ks   = (float *)malloc(kper * (size_t)c.n_layers * sizeof(float));
    float *lg   = (float *)malloc((size_t)T * (size_t)c.vocab * sizeof(float));

    forward(m, &c, full, T, lg, scratch, h, br, ks, 0);
    int tf_all = 0, tf_gen = 0, tf_gen_ok = 0;
    for (int i = 0; i < T; i++) {
        const int got = argmax_(lg + (size_t)i * (size_t)c.vocab, c.vocab);
        const int want = (int)jt->kids[i]->num;
        if (got == want) tf_all++;
        if (i >= np - 1 && i < T - 1) { tf_gen++; if (got == want) tf_gen_ok++; }
    }
    printf("GATE 1  teacher forcing : %d/%d positions match tf_pred\n", tf_all, T);
    printf("        generated span  : %d/%d  <- must be exact\n", tf_gen_ok, tf_gen);

    /* Full recompute never returns to a layer after its whole sequence has finished, so
     * its KDA matrix and ShortConv history have no lifetime beyond that layer. The CLI's
     * ultra path therefore reuses one slot instead of retaining n_layers copies. Demand
     * bit-identical logits against the old layout; token-only equality would hide a
     * near-tie or a stale-state error. */
    float *ks_reuse = (float *)malloc(kper * sizeof(float));
    float *lg_reuse = (float *)malloc((size_t)T * (size_t)c.vocab * sizeof(float));
    int reuse_ok = 0;
    if (ks_reuse && lg_reuse) {
        forward(m, &c, full, T, lg_reuse, scratch, h, br, ks_reuse, 1);
        reuse_ok = memcmp(lg, lg_reuse,
                          (size_t)T * (size_t)c.vocab * sizeof(float)) == 0;
    }
    printf("GATE 1b state reuse      : %s  <- all logits bit-identical\n",
           reuse_ok ? "PASS" : "FAIL");

    int *gen = (int *)malloc((size_t)T * sizeof(int));
    memcpy(gen, full, (size_t)np * sizeof(int));
    int cur = np, gok = 0;
    while (cur < T) {
        forward(m, &c, gen, cur, lg, scratch, h, br, ks, 0);
        gen[cur] = argmax_(lg + (size_t)(cur - 1) * (size_t)c.vocab, c.vocab);
        if (gen[cur] == full[cur]) gok++;
        cur++;
    }
    printf("GATE 2  greedy decode   : %d/%d generated tokens match full_ids\n",
           gok, T - np);

    /* ---- GATE 3: INCREMENTAL decode ----------------------------------------------
     * Gate 2 re-runs the whole prefix for every token, which is O(T^2) and, on the real
     * model, means the expert traffic grows with context. The incremental path instead
     * prefills once and then feeds ONE token at a time, carrying:
     *   - the KDA recurrent matrix and ShortConv history, which k3_kda_layer already
     *     updates in place, so the only change is to stop clearing them;
     *   - an MLA KV cache, which is genuinely new state.
     * The attn-res block stack needs nothing carried: it is rebuilt per token from that
     * token's own hidden states.
     *
     * This must produce the SAME tokens. It is a pure restructuring of when work
     * happens, not what is computed, so anything other than an exact match is a bug in
     * the state carrying, and that is exactly the failure this gate exists to catch. */
    {
        const int H = c.n_heads, kvd = c.qk_nope + c.v_head;
        const size_t kvper  = (size_t)T * H * kvd;      /* per layer */
        const size_t rpper  = (size_t)T * c.qk_rope;
        float *kvc = (float *)calloc(kvper * (size_t)c.n_layers, sizeof(float));
        float *rpc = (float *)calloc(rpper * (size_t)c.n_layers, sizeof(float));
        size_t need_i = k3_mla_scratch_cached(&c, T, T, 1);
        size_t li = k3_layer_scratch(&c, T);
        if (li > need_i) need_i = li;
        if (alt > need_i) need_i = alt;
        float *sc_i = (float *)malloc(need_i * sizeof(float));
        float *h_i  = (float *)malloc((size_t)T * (size_t)c.hidden * sizeof(float));
        float *br_i = (float *)malloc((size_t)T * (size_t)maxb * (size_t)c.hidden * sizeof(float));
        float *ks_i = (float *)malloc(kper * (size_t)c.n_layers * sizeof(float));
        float *lg_i = (float *)malloc((size_t)c.vocab * sizeof(float));
        int *gi = (int *)malloc((size_t)T * sizeof(int));
        int iok = 0;

        if (kvc && rpc && sc_i && h_i && br_i && ks_i && lg_i && gi) {
            memcpy(gi, full, (size_t)np * sizeof(int));
            memset(ks_i, 0, kper * (size_t)c.n_layers * sizeof(float));
            int cached = 0;
            for (int step = 0; cached < T - 1 || step == 0; step++) {
                /* first call feeds the whole prompt, later calls feed one token */
                const int base = cached;
                const int nT   = (step == 0) ? np : 1;
                for (int t = 0; t < nT; t++)
                    memcpy(h_i + (size_t)t * c.hidden,
                           m->embed + (size_t)gi[base + t] * c.hidden,
                           (size_t)c.hidden * sizeof(float));
                memset(br_i, 0, (size_t)nT * (size_t)maxb * (size_t)c.hidden * sizeof(float));
                int nb_i = 0;
                for (int L = 0; L < c.n_layers; L++)
                    k3_decoder_layer_inc(h_i, br_i, &nb_i, &m->lay[L], &c, L, nT,
                                         ks_i + kper * (size_t)L, sc_i,
                                         kvc + kvper * (size_t)L,
                                         rpc + rpper * (size_t)L, base, T);
                /* model-level aggregator and head, on the LAST new position */
                float *fold = sc_i, *src = fold + c.hidden;
                const int lastt = nT - 1;
                if (m->out_res_norm && m->out_res_proj) {
                    for (int i = 0; i < c.hidden; i++)
                        fold[i] = m->out_res_norm[i] * m->out_res_proj[i];
                    for (int b = 0; b < nb_i; b++)
                        memcpy(src + (size_t)b * c.hidden,
                               br_i + ((size_t)lastt * maxb + b) * c.hidden,
                               (size_t)c.hidden * sizeof(float));
                    memcpy(src + (size_t)nb_i * c.hidden,
                           h_i + (size_t)lastt * c.hidden, (size_t)c.hidden * sizeof(float));
                    k3_attn_res(h_i + (size_t)lastt * c.hidden, src, fold,
                                nb_i + 1, c.hidden, c.rms_eps);
                }
                float *nrm = sc_i;
                k3_rmsnorm(nrm, h_i + (size_t)lastt * c.hidden, m->final_norm,
                           c.hidden, c.rms_eps);
                k3_matmul(lg_i, nrm, m->lm_head, c.hidden, c.vocab);

                cached = base + nT;
                if (cached >= T) break;
                gi[cached] = argmax_(lg_i, c.vocab);
            }
            for (int i = np; i < T; i++) if (gi[i] == full[i]) iok++;
        }
        printf("GATE 3  incremental    : %d/%d generated tokens match full_ids"
               "  <- KV cache + carried KDA state\n", iok, T - np);
        gok = (iok == T - np) ? gok : -1;   /* fail the verdict if incremental diverged */
        free(kvc); free(rpc); free(sc_i); free(h_i); free(br_i); free(ks_i); free(lg_i); free(gi);
    }

    const int pass = (tf_gen_ok == tf_gen) && reuse_ok && (gok == T - np);
    printf("\nVERDICT: %s\n", pass ? "ENGINE MATCHES THE REFERENCE EXACTLY"
                                   : "MISMATCH, see the counts above");
    if (!pass) {
        printf("\n  ref  ");
        for (int i = np; i < T && i < np + 14; i++) printf("%4d", full[i]);
        printf("\n  got  ");
        for (int i = np; i < T && i < np + 14; i++) printf("%4d", gen[i]);
        printf("\n");
    }

    free(full); free(gen); free(scratch); free(h); free(br); free(ks); free(lg);
    free(ks_reuse); free(lg_reuse);
    for (int L = 0; L < c.n_layers; L++) { free(m->w1[L]); free(m->w3[L]); free(m->w2[L]); }
    free(m);

    free(rtxt); free(rar); free(st.raw); free(st.arena); free(st.blob);
    /* Return the VERDICT, not 0.
     * Propagate the gate result as the exit status so the top-level
     * correctness gate could never go red: `pass` was computed, printed, and thrown
     * away. `make oracle` reported success whether the banner said ENGINE MATCHES THE
     * REFERENCE EXACTLY or MISMATCH, so the exit=0 recorded next to that banner in
     * fixtures/gates/gates.txt carried no information at all. The careful line above,
     * `gok = (iok == T - np) ? gok : -1;`, existed only to poison a value that was then
     * discarded. Both sibling harnesses already propagated (test_ops.c and
     * scale_test.c); this one was left out. */
    return pass ? 0 : 1;
}
