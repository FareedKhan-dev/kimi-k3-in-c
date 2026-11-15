/* bench_kernels.c - where does the arithmetic actually go?
 *
 * The engine's decode time at the memory floor splits roughly 36 s trunk read, 11 s
 * expert read, 10 s compute. The read paths are now within 10-20% of what the device can
 * deliver, so the only lossless win left is the compute. Before optimising it, measure
 * which kernel owns it -- guessing which loop is hot is how people spend a week making
 * something 2% faster.
 *
 * Dimensions here are the REAL ones, per token:
 *   bf16 trunk matmuls   the attention projections, latent down/up, shared experts and
 *                        the dense MLP. Sized from the actual layer shapes.
 *   MXFP4 expert matmuls 16 experts x 3 matrices x 92 layers, in latent space.
 *
 * Reports GFLOP/s and the projected per-token seconds for each, so the two can be
 * compared directly against the measured 10 s compute budget.
 */
#define _POSIX_C_SOURCE 199309L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "k3.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Exact bits of a whole output vector.
 *
 * Building this file with and without AVX2 and comparing these hashes is the ONLY real
 * proof that a vector path is bit-identical rather than merely close. A tolerance check
 * would happily pass a kernel that quietly reassociated the reduction, which is exactly
 * the mistake worth catching: this engine's claim is that its output equals the
 * reference, and "equals" has to mean equals.
 *
 * This only means anything while the inputs are FINITE. NaN compares unequal to itself
 * and carries a payload that propagates differently through scalar and vector code, so
 * a NaN input turns this from a proof into a guaranteed false alarm. See fillbf16. */
static void fnv(const char *label, const float *v, int n)
{
    unsigned long long h = 1469598103934665603ull;
    for (int k = 0; k < n; k++) {
        union { float f; unsigned u; } b; b.f = v[k];
        for (int t = 0; t < 4; t++) { h ^= (b.u >> (8 * t)) & 0xFFu; h *= 1099511628211ull; }
    }
    printf("             %s OUTPUT FNV1a = %016llx\n", label, h);
}

static void fillf(float *p, size_t n, unsigned s)
{
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        p[i] = ((float)(s >> 8) / 8388608.0f - 1.0f) * 0.05f;
    }
}
static void fillb(unsigned char *p, size_t n, unsigned s)
{
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        p[i] = (unsigned char)(s >> 13);
    }
}

/* bf16 weights that are FINITE, which fillb cannot promise.
 *
 * Random bytes reinterpreted as bf16 hit the all-ones exponent about one time in 256,
 * so a 7168-term row is NaN with probability 1 - (255/256)^7168, i.e. essentially
 * always. Every output of the bf16 section then comes out NaN, and hashing NaN compares
 * PAYLOADS rather than arithmetic: those propagate differently through libm's fma() and
 * through _mm256_fmadd_pd, so the two builds disagree unconditionally and the check
 * described above reports a bit-identity failure that is not there.
 *
 * Same xorshift and the same distribution as fillf, truncated to the top half, which is
 * what bf16 storage is. No intermediate float array: at 12288 x 7168 that would be a
 * 352 MB temporary for values consumed once. */
static void fillbf16(uint16_t *p, size_t n, unsigned s)
{
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        const float f = ((float)(s >> 8) / 8388608.0f - 1.0f) * 0.05f;
        unsigned u; memcpy(&u, &f, sizeof u);
        p[i] = (unsigned short)(u >> 16);
    }
}

int main(void)
{
    printf("kernel benchmark at REAL Kimi K3 dimensions\n");
#ifdef __AVX2__
    printf("built WITH AVX2\n\n");
#elif defined(__ARM_NEON) && defined(__aarch64__)
    printf("built WITH NEON\n\n");
#else
    printf("built WITHOUT AVX2 or NEON (scalar)\n\n");
#endif

    /* ---------- bf16 trunk matmul: KDA q_proj, 7168 -> 12288 ---------- */
    {
        const int in = 7168, out = 12288;
        uint16_t *W = (uint16_t *)malloc((size_t)in * out * sizeof(uint16_t));
        float *x = (float *)malloc((size_t)in * sizeof(float));
        float *y = (float *)malloc((size_t)out * sizeof(float));
        if (!W || !x || !y) { printf("alloc failed\n"); return 1; }
        fillbf16(W, (size_t)in * out, 12345u);
        fillf(x, in, 999u);

        k3_matmul_bf16(y, x, W, in, out);              /* warm */
        const int reps = 5;
        const double t0 = now_s();
        for (int r = 0; r < reps; r++) k3_matmul_bf16(y, x, W, in, out);
        const double dt = (now_s() - t0) / reps;
        const double gflop = 2.0 * in * out / 1e9;
        printf("bf16 matmul  %5d x %-5d  %7.2f ms  %8.1f GFLOP/s\n",
               out, in, dt * 1e3, gflop / dt);

        /* Per token the trunk is 56.74 G always-active params = 113.49 GFLOP of
         * multiply-add. Project from the rate just measured. */
        fnv("bf16 ", y, out);
        printf("             trunk is 56.74 G params/token -> %.2f s/token at this rate\n",
               2.0 * 56.74e9 / 1e9 / (gflop / dt));
        free(W); free(x); free(y);
    }

    /* ---------- MXFP4 expert matmul: w1, latent 3584 -> inter 3072 ---------- */
    {
        const int in = 3584, rows = 3072, group = K3_MXFP4_GROUP;
        const int pcols = in / 2, ngrp = in / group;
        unsigned char *pk = (unsigned char *)malloc((size_t)rows * pcols);
        unsigned char *sc = (unsigned char *)malloc((size_t)rows * ngrp);
        float *x = (float *)malloc((size_t)in * sizeof(float));
        float *y = (float *)malloc((size_t)rows * sizeof(float));
        if (!pk || !sc || !x || !y) { printf("alloc failed\n"); return 1; }
        fillb(pk, (size_t)rows * pcols, 777u);
        memset(sc, 127, (size_t)rows * ngrp);          /* scale 2^0, none skipped */
        fillf(x, in, 4242u);

        k3_matmul_mxfp4(y, x, pk, sc, in, rows, group);
        const int reps = 5;
        const double t0 = now_s();
        for (int r = 0; r < reps; r++) k3_matmul_mxfp4(y, x, pk, sc, in, rows, group);
        const double dt = (now_s() - t0) / reps;
        const double gflop = 2.0 * in * rows / 1e9;
        printf("\nMXFP4 matmul %5d x %-5d  %7.2f ms  %8.1f GFLOP/s\n",
               rows, in, dt * 1e3, gflop / dt);

        /* One expert is w1 + w3 (both 3072x3584) + w2 (3584x3072) = 3 of these.
         * 16 experts x 92 MoE layers per token. */
        fnv("mxfp4", y, rows);
        const double per_tok = dt * 3.0 * 16 * 92;
        printf("             16 experts x 3 mats x 92 layers -> %.2f s/token\n", per_tok);
        free(pk); free(sc); free(x); free(y);
    }

    /* ---------- Gated MLA attention: real head geometry, decode at long context ------
     * What scales with context is the attention loop itself, so the projections here
     * are deliberately tiny (hidden 256, q_lora 64, kv_lora 64, fp32) while the part that
     * matters is the released one: 96 heads, 192-wide q.k (128 nope + 64 shared rope),
     * 128-wide values, over a filled KV cache of P positions. Decode attention is linear
     * in P, so the per-layer time here scales directly to longer contexts. */
    {
        enum { H = 96, QN = 128, QR = 64, VH = 128, P = 4096 };
        const int kvd = QN + VH, qh = QN + QR;
        K3Cfg c; memset(&c, 0, sizeof c);
        c.n_heads = H; c.qk_nope = QN; c.qk_rope = QR; c.v_head = VH;
        c.hidden = 256; c.q_lora = 64; c.kv_lora = 64; c.rms_eps = 1e-6f;
        const int E = c.hidden, kvw = c.kv_lora + QR;
        float *q_a = (float *)malloc((size_t)c.q_lora * E * sizeof(float));
        float *q_b = (float *)malloc((size_t)H * qh * c.q_lora * sizeof(float));
        float *kv_a = (float *)malloc((size_t)kvw * E * sizeof(float));
        float *kv_b = (float *)malloc((size_t)H * kvd * c.kv_lora * sizeof(float));
        float *wo = (float *)malloc((size_t)E * H * VH * sizeof(float));
        float *wg = (float *)malloc((size_t)H * VH * E * sizeof(float));
        float *n1 = (float *)malloc((size_t)c.q_lora * sizeof(float));
        float *n2 = (float *)malloc((size_t)c.kv_lora * sizeof(float));
        float *kvc = (float *)malloc((size_t)P * H * kvd * sizeof(float));
        float *ropec = (float *)malloc((size_t)P * QR * sizeof(float));
        float *x = (float *)malloc((size_t)E * sizeof(float));
        float *y = (float *)malloc((size_t)E * sizeof(float));
        float *scr = (float *)malloc(k3_mla_scratch_cached(&c, 1, P, 1) * sizeof(float));
        if (!q_a || !q_b || !kv_a || !kv_b || !wo || !wg || !n1 || !n2 || !kvc || !ropec ||
            !x || !y || !scr) { printf("alloc failed\n"); return 1; }
        fillf(q_a, (size_t)c.q_lora * E, 11u);  fillf(q_b, (size_t)H * qh * c.q_lora, 12u);
        fillf(kv_a, (size_t)kvw * E, 13u);      fillf(kv_b, (size_t)H * kvd * c.kv_lora, 14u);
        fillf(wo, (size_t)E * H * VH, 15u);     fillf(wg, (size_t)H * VH * E, 16u);
        for (int i = 0; i < c.q_lora; i++) n1[i] = 1.0f;
        for (int i = 0; i < c.kv_lora; i++) n2[i] = 1.0f;
        fillf(kvc, (size_t)P * H * kvd, 17u);   fillf(ropec, (size_t)P * QR, 18u);
        fillf(x, E, 19u);
        K3MlaW w; memset(&w, 0, sizeof w);      /* wdt 0: fp32 weights */
        w.q_a = q_a; w.q_b = q_b; w.kv_a = kv_a; w.kv_b = kv_b; w.o = wo; w.g = wg;
        w.q_a_norm = n1; w.kv_a_norm = n2;

        /* One decode step at position P-1, attending over all P cached positions. It
         * rewrites only position P-1 of the cache, from the same input each time. */
        double t1 = 0.0, tn = 0.0;
        unsigned long long h1 = 0, hn = 0;
        int nth = 1;
        for (int pass = 0; pass < 2; pass++) {
#ifdef _OPENMP
            const int want = pass == 0 ? 1 : omp_get_num_procs();
            omp_set_num_threads(want);
            nth = want;
#else
            if (pass == 1) break;
#endif
            k3_mla_cached(y, x, &w, &c, 1, scr, kvc, ropec, P - 1, P);   /* warm */
            const int reps = 3;
            const double t0 = now_s();
            for (int r = 0; r < reps; r++)
                k3_mla_cached(y, x, &w, &c, 1, scr, kvc, ropec, P - 1, P);
            const double dt = (now_s() - t0) / reps;
            unsigned long long h = 1469598103934665603ull;
            for (int k = 0; k < E; k++) {
                union { float f; unsigned u; } b; b.f = y[k];
                for (int t = 0; t < 4; t++) { h ^= (b.u >> (8 * t)) & 0xFFu; h *= 1099511628211ull; }
            }
            if (pass == 0) { t1 = dt; h1 = h; } else { tn = dt; hn = h; }
        }
        printf("\nMLA attention, 96 heads, decode at %d positions\n", P);
        printf("             1 thread    %8.2f ms/layer -> %6.2f s/token over 24 MLA layers\n",
               t1 * 1e3, t1 * 24);
        if (tn > 0.0) {
            printf("             %-3d threads %8.2f ms/layer -> %6.2f s/token  (%.1fx)\n",
                   nth, tn * 1e3, tn * 24, t1 / tn);
            printf("             output FNV1a %016llx vs %016llx: %s\n", h1, hn,
                   h1 == hn ? "bit-identical" : "DIFFERENT");
            if (h1 != hn) return 1;
        }
        printf("             attention is linear in context at decode: x%d at %d positions\n",
               32768 / P, 32768);
        free(q_a); free(q_b); free(kv_a); free(kv_b); free(wo); free(wg); free(n1); free(n2);
        free(kvc); free(ropec); free(x); free(y); free(scr);
    }

    printf("\nmeasured compute budget at the floor is about 10 s/token; whichever line\n"
           "above dominates it is the one worth vectorising.\n");
    return 0;
}
