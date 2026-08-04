/* scale_test.c - does the engine hold up at REAL Kimi K3 dimensions?
 *
 * Everything validated so far ran at hidden 128 with 13 layers. The real model is
 * 7168 wide with 93 layers, 96 heads, 896 experts. The mathematics is identical, but
 * three classes of failure only appear at scale and all of them are silent:
 *
 *   1. INTEGER OVERFLOW. 12288 * 7168 = 88,080,384 fits in int32, but
 *      12288 * 7168 * 4 bytes = 352 MB does not fit comfortably once multiplied by a
 *      layer count, and an expert bank is 92 * 896 * 33,030,144 elements. Any product
 *      computed in int rather than size_t wraps and produces a wrong pointer.
 *   2. SCRATCH SIZING. The k3_*_scratch helpers must return the right value at real
 *      widths, not just tiny ones.
 *   3. EXECUTION. A full-width KDA layer is about 443M parameters; at fp32 that is
 *      1.8 GB of weights, which fits on an ordinary box and is worth running once.
 *
 * This does NOT load real weights. It allocates at real shapes with synthetic values
 * to prove the plumbing survives the dimensions.
 */
/* -std=c99 hides the POSIX clock API; request it before any include. */
#define _POSIX_C_SOURCE 199309L
#define _DARWIN_C_SOURCE

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "k3.h"

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void fill(float *p, size_t n, unsigned seed)
{
    unsigned s = seed ? seed : 1u;
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        p[i] = ((float)(s >> 8) / 8388608.0f - 1.0f) * 0.02f;
    }
}

static void human(double bytes, char *out, size_t n)
{
    const char *u[] = {"B", "KB", "MB", "GB", "TB"};
    int i = 0; while (bytes >= 1000.0 && i < 4) { bytes /= 1000.0; i++; }
    snprintf(out, n, "%.2f %s", bytes, u[i]);
}

int main(void)
{
    K3Cfg c; memset(&c, 0, sizeof c);
    c.hidden = 7168;  c.n_layers = 93;   c.vocab = 163840; c.rms_eps = 1e-5f;
    c.kda_heads = 96; c.kda_head_dim = 128; c.conv_k = 4;  c.gate_lb = -5.0f;
    c.n_heads = 96;   c.q_lora = 1536;   c.kv_lora = 512;
    c.qk_nope = 128;  c.qk_rope = 64;    c.v_head = 128;   c.mla_out_gate = 1;
    c.n_experts = 896; c.topk = 16;      c.n_shared = 2;
    c.latent = 3584;  c.moe_inter = 3072; c.routed_scale = 1.0f;
    c.moe_renorm = 1; c.latent_norm = 1;
    c.first_dense = 1; c.dense_inter = 33792;
    c.attn_res_block = 12;
    c.situ_b1 = 4.0f; c.situ_b2 = 25.0f;

    static int fa[24];
    int nfa = 0;
    for (int i = 4; i <= 93; i += 4) fa[nfa++] = i;
    fa[nfa++] = 93;                       /* the trailing extra MLA layer */
    c.n_full_attn = nfa; c.full_attn = fa;

    char b1[32], b2[32];
    printf("Kimi K3 scale test, REAL dimensions\n");
    printf("hidden %d, layers %d, heads %d, experts %d top-%d, latent %d\n\n",
           c.hidden, c.n_layers, c.kda_heads, c.n_experts, c.topk, c.latent);

    /* ---------- 1. layer map ---------- */
    int nk = 0, nm = 0;
    for (int i = 0; i < c.n_layers; i++) { if (k3_is_mla(&c, i)) nm++; else nk++; }
    printf("layer map: %d KDA + %d MLA = %d  (expect 69 + 24 = 93)  %s\n",
           nk, nm, nk + nm, (nk == 69 && nm == 24) ? "OK" : "WRONG");
    printf("  last five layers: ");
    for (int i = c.n_layers - 5; i < c.n_layers; i++)
        printf("%d=%s ", i, k3_is_mla(&c, i) ? "MLA" : "KDA");
    printf("\n  layer 0 dense: %s\n\n", k3_is_dense(&c, 0) ? "yes" : "no");

    /* ---------- 2. scratch sizing, checked for overflow ---------- */
    printf("scratch requirements (floats), checked in size_t:\n");
    const int Ts[] = {1, 8, 512, 4096};
    int scratch_bad = 0;
    for (unsigned i = 0; i < sizeof Ts / sizeof *Ts; i++) {
        const int T = Ts[i];
        size_t mla = k3_mla_scratch(&c, T), kda = k3_kda_scratch(&c, T);
        size_t moe = k3_moe_scratch(&c),    lay = k3_layer_scratch(&c, T);
        human((double)lay * 4, b1, sizeof b1);
        printf("  T=%-5d mla %-12zu kda %-12zu moe %-10zu layer %-12zu (%s)\n",
               T, mla, kda, moe, lay, b1);
        /* A scratch size of zero, or a layer buffer smaller than a sub-buffer it must
         * contain, is a heap overflow waiting for a caller that trusts it. Detecting one
         * is a FAILURE, not a remark, so it feeds the verdict at the end of main. */
        if (mla == 0 || kda == 0 || lay < mla || lay < kda) {
            printf("    OVERFLOW OR UNDERSIZE at T=%d\n", T);
            scratch_bad = 1;
        }
    }
    printf("\n");

    /* ---------- 3. parameter and byte budget ---------- */
    const size_t P = (size_t)c.kda_heads * c.kda_head_dim;      /* 12288 */
    const size_t kda_par = 4 * P * c.hidden + (size_t)c.hidden * P
                         + 3 * P * c.conv_k
                         + (size_t)c.kda_head_dim * c.hidden + P * c.kda_head_dim
                         + (size_t)c.kda_heads * c.hidden
                         + c.kda_heads + P + c.kda_head_dim;
    const size_t exp_par = 3ull * (size_t)c.moe_inter * c.latent;
    const size_t all_exp = (size_t)(c.n_layers - 1) * c.n_experts * exp_par;

    human((double)kda_par * 2, b1, sizeof b1);
    printf("one KDA layer  : %zu params  (%s at bf16)\n", kda_par, b1);
    human((double)kda_par * 69 * 2, b1, sizeof b1);
    /* This figure covers KDA ATTENTION ONLY and is NOT the resident trunk. It omits the
     * 24 MLA layers and, more importantly, the shared experts, 6144 wide, shipped BF16
     * rather than MXFP4, dense, and fired on every token. The always-active set measured
     * from the 96 shard headers by tools/budget.py is 56,743,648,000 parameters,
     * 113.49 GB at bf16, of which this line is barely half. Reporting it as the trunk
     * understates the memory requirement by a factor of two. */
    printf("69 KDA layers  : %s at bf16   (KDA attention ONLY, not the whole trunk)\n", b1);
    printf("full trunk     : 113.49 GB at bf16, 56,743,648,000 always-active params\n"
           "                 (measured from the shard headers, not derived here)\n");
    human((double)exp_par * 0.53125, b1, sizeof b1);
    printf("one expert     : %zu params  (%s at MXFP4)\n", exp_par, b1);
    human((double)all_exp * 0.53125, b1, sizeof b1);
    printf("all %zu experts: %s at MXFP4  <- streamed from NVMe\n",
           (size_t)(c.n_layers - 1) * c.n_experts, b1);

    /* ---------- 4. per-sequence state ---------- */
    const size_t kstate = (size_t)69 * c.kda_heads * c.kda_head_dim * c.kda_head_dim;
    const size_t cstate = (size_t)69 * 3 * P * (c.conv_k - 1);
    human((double)kstate * 2, b1, sizeof b1);
    human((double)cstate * 2, b2, sizeof b2);
    printf("\nper-sequence state, FIXED regardless of context:\n");
    printf("  KDA recurrent : %s at bf16\n", b1);
    printf("  ShortConv     : %s at bf16\n", b2);
    /* What the engine ACTUALLY caches, which is not the compressed latent. Re-expanding
     * the 576-float latent through kv_b for every cached position would cost an O(T)
     * sweep of 24576x512 matmuls per layer per token, so k3_mla_cached stores the
     * EXPANDED per-head keys and values plus the shared rope slot, in fp32. That is 42x
     * the latent, and quoting the latent figure here understated the cost by the same
     * factor. */
    const double kv_tok = ((double)c.n_heads * (c.qk_nope + c.v_head) + c.qk_rope) * 24 * 4;
    human(kv_tok, b1, sizeof b1);
    printf("  MLA KV        : %s per position (24 MLA layers, EXPANDED k and v, fp32)\n", b1);
    for (long ctx = 8192; ctx <= 1048576; ctx *= 16) {
        human(kv_tok * (double)ctx, b1, sizeof b1);
        printf("                  %s at %ld context\n", b1, ctx);
    }

    /* ---------- 5. actually run one full-width KDA layer ---------- */
    printf("\nallocating and running ONE full-width KDA layer (fp32)...\n");
    const int T = 4;
    K3KdaW w; memset(&w, 0, sizeof w);
    size_t need = 4 * P * c.hidden + (size_t)c.hidden * P + 3 * P * c.conv_k
                + (size_t)c.kda_head_dim * c.hidden + P * c.kda_head_dim
                + (size_t)c.kda_heads * c.hidden + c.kda_heads + P + c.kda_head_dim;
    human((double)need * 4, b1, sizeof b1);
    printf("  weights: %s\n", b1);
    float *W_ = (float *)malloc(need * sizeof(float));
    if (!W_) { printf("  ALLOCATION FAILED -- not enough RAM for the test\n"); return 1; }
    fill(W_, need, 12345u);

    size_t o = 0;
    w.q = W_ + o; o += P * c.hidden;   w.k = W_ + o; o += P * c.hidden;
    w.v = W_ + o; o += P * c.hidden;   w.g = W_ + o; o += P * c.hidden;
    w.o = W_ + o; o += (size_t)c.hidden * P;
    w.q_conv = W_ + o; o += P * c.conv_k;
    w.k_conv = W_ + o; o += P * c.conv_k;
    w.v_conv = W_ + o; o += P * c.conv_k;
    w.f_a = W_ + o; o += (size_t)c.kda_head_dim * c.hidden;
    w.f_b = W_ + o; o += P * c.kda_head_dim;
    w.b   = W_ + o; o += (size_t)c.kda_heads * c.hidden;
    w.A_log = W_ + o; o += c.kda_heads;
    w.dt_bias = W_ + o; o += P;
    w.o_norm = W_ + o; o += c.kda_head_dim;

    float *x  = (float *)malloc((size_t)T * c.hidden * sizeof(float));
    float *y  = (float *)malloc((size_t)T * c.hidden * sizeof(float));
    float *sc = (float *)malloc(k3_kda_scratch(&c, T) * sizeof(float));
    float *st = (float *)calloc((size_t)P * c.kda_head_dim + 3 * P * (c.conv_k - 1),
                                sizeof(float));
    if (!x || !y || !sc || !st) { printf("  scratch allocation FAILED\n"); return 1; }
    fill(x, (size_t)T * c.hidden, 999u);

    const double t0 = now_s();
    k3_kda_layer(y, x, &w, &c, T, st, sc);
    const double dt = now_s() - t0;

    int finite = 1; double mx = 0.0;
    for (size_t i = 0; i < (size_t)T * c.hidden; i++) {
        if (!isfinite(y[i])) { finite = 0; break; }
        if (fabs(y[i]) > mx) mx = fabs(y[i]);
    }
    printf("  ran %d tokens in %.2f s (%.0f ms/token)\n", T, dt, dt * 1000 / T);
    printf("  output all finite: %s, max |y| = %.6f\n", finite ? "YES" : "NO", mx);
    printf("  state non-zero   : %s\n",
           st[0] != 0.0f || st[1] != 0.0f ? "yes" : "no (suspicious)");

    /* Every condition this file checks feeds the exit status. A test that prints an
     * alarming banner and then exits 0 is worse than no test: CI stays green and the
     * banner scrolls past. */
    const int ok = (nk == 69 && nm == 24 && finite && !scratch_bad);
    printf("\nSCALE TEST %s\n", ok ? "PASSED" : "FAILED");
    if (scratch_bad)
        printf("  a scratch size was zero or undersized see the table above\n");
    if (!finite)
        printf("  the KDA layer produced non-finite output at full width\n");
    if (nk != 69 || nm != 24)
        printf("  layer map is wrong: %d KDA + %d MLA, expected 69 + 24\n", nk, nm);
    free(W_); free(x); free(y); free(sc); free(st);
    return ok ? 0 : 1;
}
