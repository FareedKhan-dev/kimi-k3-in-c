/* test_real_layer.c - run a REAL Kimi K3 layer, from the REAL checkpoint.
 *
 * WHAT IS ACTUALLY NEW HERE
 *   The arithmetic is already validated: 20 op fixtures, the full-model oracle matching
 *   token for token: the MLA implementation against the released KimiMLAAttention, KDA against the
 *   released KimiDeltaAttention on GPU. All of that ran on weights this project
 *   generated.
 *
 *   What has never been tested is the BINDING: whether k3_bind attaches the right
 *   checkpoint tensor to the right kernel argument. That failure mode is invisible to
 *   every test above, because a transposed or swapped weight still has the right shape
 *   and produces finite, plausible numbers. It can only be caught by running real
 *   weights and comparing against an independent implementation reading the same file.
 *
 *   So this dumps its input and its intermediate outputs in stages, and
 *   tools/verify_real_layer.py reproduces each stage in torch from the same shards:
 *     stage A  KDA attention output    -> the 15 attention tensors are wired correctly
 *     stage B  router indices/weights  -> gate and e_score_correction_bias are correct
 *     stage C  MoE output              -> latent down/up, shared experts, and the
 *                                          streamed MXFP4 experts are all correct
 *   Staging matters: a single end-to-end number that disagrees tells you nothing about
 *   which of thirty tensors is wrong.
 *
 * usage: test_real_layer <shard_dir> [layer] [T] [cache_gb] [dump_path]
 */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "k3.h"
#include "k3_bind.h"
#include "k3_cache.h"

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static unsigned rs = 20260728u;
static float rnd_f(void)
{
    rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5;
    return ((float)(rs >> 8) / 8388608.0f - 1.0f);
}

static void dump_f32(FILE *f, const char *name, const float *v, int64_t n, int comma)
{
    fprintf(f, "%s\"%s\":[", comma ? "," : "", name);
    for (int64_t i = 0; i < n; i++) {
        union { float f; uint32_t u; } b; b.f = v[i];
        fprintf(f, "%s%u", i ? "," : "", b.u);   /* bit patterns: exact, and NaN-safe */
    }
    fprintf(f, "]");
}

/* The real Kimi K3 configuration, from config.json and confirmed against the shipped
 * tensor shapes in every case. */
static void real_cfg(K3Cfg *c, int *fa, int *nfa)
{
    memset(c, 0, sizeof *c);
    c->hidden = 7168;  c->n_layers = 93;   c->vocab = 163840; c->rms_eps = 1e-5f;
    c->kda_heads = 96; c->kda_head_dim = 128; c->conv_k = 4;  c->gate_lb = -5.0f;
    c->n_heads = 96;   c->q_lora = 1536;   c->kv_lora = 512;
    c->qk_nope = 128;  c->qk_rope = 64;    c->v_head = 128;   c->mla_out_gate = 1;
    c->n_experts = 896; c->topk = 16;      c->n_shared = 2;
    c->latent = 3584;  c->moe_inter = 3072; c->routed_scale = 1.0f;
    c->moe_renorm = 1; c->latent_norm = 1;
    c->first_dense = 1; c->dense_inter = 33792;
    c->attn_res_block = 12;
    c->situ_b1 = 4.0f; c->situ_b2 = 25.0f;
    int n = 0;
    for (int i = 4; i <= 93; i += 4) fa[n++] = i;    /* one-based in config */
    fa[n++] = 93;
    c->n_full_attn = n; c->full_attn = fa;
    *nfa = n;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: test_real_layer <shard_dir> [layer] [T] [cache_gb] [dump_path]\n"); return 2; }
    const char *dir = argv[1];
    const int L = argc > 2 ? atoi(argv[2]) : 1;
    const int T = argc > 3 ? atoi(argv[3]) : 4;
    const double cache_gb = argc > 4 ? atof(argv[4]) : 2.0;
    const char *dump_path = argc > 5 ? argv[5] : "real_layer.json";

    K3Cfg c; static int fa[32]; int nfa;
    real_cfg(&c, fa, &nfa);

    printf("Kimi K3, released weights, layer %d, T=%d\n", L, T);
    printf("  layer %d is %s and %s\n", L,
           k3_is_mla(&c, L) ? "MLA" : "KDA", k3_is_dense(&c, L) ? "dense" : "MoE");

    K3St st;
    double t0 = now_s();
    if (k3_st_open(&st, dir) != 0) return 1;
    printf("  indexed %d tensors from %d shard(s) in %.2f s\n\n", st.nt, st.nshard, now_s() - t0);

    /* ---- bind ---- */
    const int64_t need = k3_bind_layer_bytes(&st, &c, L);
    if (need < 0) { fprintf(stderr, "binding plan FAILED, see errors above\n"); return 1; }
    printf("layer %d resident weights: %.2f GB in RAM, held in the checkpoint's own bf16\n",
           L, (double)need / 1e9);

    K3LayerBind b;
    t0 = now_s();
    if (k3_bind_layer(&st, &c, L, &b) != 0) { fprintf(stderr, "BIND FAILED\n"); return 1; }
    printf("  bound in %.2f s (%.0f MB/s)\n\n",
           now_s() - t0, (double)need / 1e6 / (now_s() - t0));

    /* ---- the expert cache ---- */
    K3Cache cache;
    if (k3_cache_init(&cache, &st, &c, (int64_t)(cache_gb * 1e9)) != 0) return 1;
    printf("expert cache: %d slots x %.2f MB = %.2f GB\n\n",
           cache.nslot, (double)cache.slot_bytes / 1e6,
           (double)cache.nslot * cache.slot_bytes / 1e9);
    b.moe.src = &cache.src;
    b.moe.layer = L;

    /* ---- input ---- */
    float *x = (float *)malloc((size_t)T * c.hidden * sizeof(float));
    for (int64_t i = 0; i < (int64_t)T * c.hidden; i++) x[i] = rnd_f() * 0.5f;

    FILE *f = fopen(dump_path, "w");
    if (!f) { fprintf(stderr, "cannot open dump %s\n", dump_path); return 1; }
    fprintf(f, "{\"layer\":%d,\"T\":%d,\"hidden\":%d,\"topk\":%d,\"is_mla\":%d",
            L, T, c.hidden, c.topk, k3_is_mla(&c, L));
    dump_f32(f, "x", x, (int64_t)T * c.hidden, 1);

    /* ---- stage A: attention ---- */
    float *att = (float *)malloc((size_t)T * c.hidden * sizeof(float));
    float *xn  = (float *)malloc((size_t)T * c.hidden * sizeof(float));
    for (int t = 0; t < T; t++)
        k3_rmsnorm(xn + (size_t)t * c.hidden, x + (size_t)t * c.hidden,
                   b.lay.in_norm, c.hidden, c.rms_eps);
    dump_f32(f, "x_norm", xn, (int64_t)T * c.hidden, 1);

    const size_t P = (size_t)c.kda_heads * c.kda_head_dim;
    float *state = (float *)calloc(P * c.kda_head_dim + 3 * P * (c.conv_k - 1), sizeof(float));
    float *scr = (float *)malloc(k3_layer_scratch(&c, T) * sizeof(float));
    if (!state || !scr) { fprintf(stderr, "scratch allocation failed\n"); return 1; }

    /* KDA and MLA are different code paths with entirely different weight sets, and
     * only one of them has ever been checked against real weights. Run whichever this
     * layer actually is. */
    const int is_mla = k3_is_mla(&c, L);
    t0 = now_s();
    if (is_mla) k3_mla(att, xn, &b.mla, &c, T, scr);
    else        k3_kda_layer(att, xn, &b.kda, &c, T, state, scr);
    const double t_attn = now_s() - t0;
    dump_f32(f, "attn_out", att, (int64_t)T * c.hidden, 1);
    printf("stage A  %s attention : %.2f s for %d tokens (%.0f ms/token)\n",
           is_mla ? "MLA" : "KDA", t_attn, T, t_attn * 1000 / T);

    /* ---- stage B: routing, on the post-attention normalised hidden ---- */
    float *hn = (float *)malloc((size_t)T * c.hidden * sizeof(float));
    for (int t = 0; t < T; t++)
        k3_rmsnorm(hn + (size_t)t * c.hidden, att + (size_t)t * c.hidden,
                   b.lay.post_norm, c.hidden, c.rms_eps);
    dump_f32(f, "moe_in", hn, (int64_t)T * c.hidden, 1);

    /* LAYER 0 IS DENSE and has no router at all: k3_bind sets lay.moe to NULL for it
     * (k3_bind.c:295) because the checkpoint carries mlp.gate_proj / up_proj / down_proj
     * instead of block_sparse_moe.*. Falling into the routing code below dereferenced a
     * null gate and SEGFAULTED, which is why layer 0 -- the ONLY layer with this code
     * path, a 33792-wide dense MLP -- had never once been conformance-checked against
     * torch. Run the path the engine actually runs for it. */
    if (k3_is_dense(&c, L)) {
        const int DI = c.dense_inter;
        float *dgu = (float *)malloc((size_t)2 * DI * sizeof(float));
        float *dsub = (float *)malloc((size_t)DI * sizeof(float));
        float *dout = (float *)malloc((size_t)T * c.hidden * sizeof(float));
        if (!dgu || !dsub || !dout) { fprintf(stderr, "dense scratch failed\n"); return 1; }
        t0 = now_s();
        for (int t = 0; t < T; t++) {
            k3_mmw(dgu, hn + (size_t)t * c.hidden, b.lay.dense_gate, b.lay.wdt,
                   c.hidden, DI);
            k3_mmw(dgu + DI, hn + (size_t)t * c.hidden, b.lay.dense_up, b.lay.wdt,
                   c.hidden, DI);
            k3_situ_glu(dsub, dgu, DI, c.situ_b1, c.situ_b2);
            k3_mmw(dout + (size_t)t * c.hidden, dsub, b.lay.dense_down, b.lay.wdt,
                   DI, c.hidden);
        }
        const double t_dense = now_s() - t0;
        dump_f32(f, "dense_out", dout, (int64_t)T * c.hidden, 1);
        fprintf(f, "}\n");
        int write_failed = ferror(f);
        if (fclose(f) != 0) write_failed = 1;
        printf("stage B  (skipped: dense layer, no router)\n");
        printf("stage C  dense MLP, inter %d: %.2f s for %d tokens (%.0f ms/token)\n\n",
               DI, t_dense, T, t_dense * 1000 / T);
        int dfin = 1; double dmx = 0.0;
        for (int64_t i = 0; i < (int64_t)T * c.hidden; i++) {
            if (!isfinite(dout[i])) { dfin = 0; break; }
            if (fabs(dout[i]) > dmx) dmx = fabs(dout[i]);
        }
        printf("output finite: %s, max |y| = %.6f\n", dfin ? "YES" : "NO", dmx);
        if (write_failed) fprintf(stderr, "cannot finish writing dump %s\n", dump_path);
        else printf("wrote %s for tools/verify_real_layer.py\n", dump_path);
        free(dgu); free(dsub); free(dout); free(hn); free(att); free(x); free(scr);
        k3_cache_free(&cache); k3_bind_free(&b); k3_st_close(&st);
        return dfin && !write_failed ? 0 : 1;
    }

    int *idx = (int *)malloc((size_t)c.topk * sizeof(int));
    float *wt = (float *)malloc((size_t)c.topk * sizeof(float));
    fprintf(f, ",\"route_idx\":[");
    float *allw = (float *)malloc((size_t)T * c.topk * sizeof(float));
    for (int t = 0; t < T; t++) {
        k3_router(idx, wt, hn + (size_t)t * c.hidden, b.moe.gate, b.moe.bias,
                  c.hidden, c.n_experts, c.topk, c.moe_renorm, c.routed_scale);
        for (int j = 0; j < c.topk; j++) {
            fprintf(f, "%s%d", (t || j) ? "," : "", idx[j]);
            allw[t * c.topk + j] = wt[j];
        }
        if (t == 0) {
            printf("stage B  router, token 0: experts");
            for (int j = 0; j < 8; j++) printf(" %d(%.4f)", idx[j], wt[j]);
            printf(" ...\n");
        }
    }
    fprintf(f, "]");
    dump_f32(f, "route_wt", allw, (int64_t)T * c.topk, 1);

    /* ---- stage C: MoE with STREAMED experts ---- */
    float *moe_out = (float *)malloc((size_t)T * c.hidden * sizeof(float));
    float *mscr = (float *)malloc(k3_moe_scratch(&c) * sizeof(float));
    if (!moe_out || !mscr) { fprintf(stderr, "moe scratch allocation failed\n"); return 1; }

    k3_cache_reset_stats(&cache);
    t0 = now_s();
    k3_moe(moe_out, hn, &b.moe, &c, T, idx, wt, mscr);
    const double t_moe = now_s() - t0;
    dump_f32(f, "moe_out", moe_out, (int64_t)T * c.hidden, 1);
    fprintf(f, "}\n");
    int write_failed = ferror(f);
    if (fclose(f) != 0) write_failed = 1;

    printf("stage C  MoE (streamed): %.2f s for %d tokens (%.0f ms/token)\n\n",
           t_moe, T, t_moe * 1000 / T);
    k3_cache_report(&cache, "after one layer, cold");

    /* Second pass: same tokens, so every expert is already resident. The gap between
     * the two is exactly what caching buys. */
    k3_cache_reset_stats(&cache);
    t0 = now_s();
    k3_moe(moe_out, hn, &b.moe, &c, T, idx, wt, mscr);
    const double t_moe2 = now_s() - t0;
    printf("\nsecond pass, fully cached: %.2f s (%.0f ms/token), %.1fx faster\n",
           t_moe2, t_moe2 * 1000 / T, t_moe / t_moe2);
    k3_cache_report(&cache, "second pass");

    int finite = 1; double mx = 0.0;
    for (int64_t i = 0; i < (int64_t)T * c.hidden; i++) {
        if (!isfinite(moe_out[i])) { finite = 0; break; }
        if (fabs(moe_out[i]) > mx) mx = fabs(moe_out[i]);
    }
    printf("\noutput finite: %s, max |y| = %.6f\n", finite ? "YES" : "NO", mx);
    if (write_failed) fprintf(stderr, "cannot finish writing dump %s\n", dump_path);
    else printf("wrote %s for tools/verify_real_layer.py\n", dump_path);

    k3_cache_dump_hist(&cache, "expert_hist.json");
    k3_cache_dump_trace(&cache, "expert_trace.bin");
    k3_cache_free(&cache);
    k3_bind_free(&b);
    k3_st_close(&st);
    return finite && !write_failed ? 0 : 1;
}
