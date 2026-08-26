/* test_expert.c - load real routed experts from the real checkpoint, and time it.
 *
 * TWO QUESTIONS, BOTH ANSWERED WITH THE RELEASED WEIGHTS ON LOCAL DISK.
 *
 * 1. CORRECTNESS. test_ops already proves k3_mxfp4_dequant is bit-exact against
 *    fixtures/mxfp4.json, but that fixture was built from bytes pulled over HTTP by
 *    tools/fetch_mxfp4.py. It proves the dequantiser, not the reader. This program
 *    closes the last link: it reads the same expert out of a local shard through
 *    k3_st/k3_load and dumps both the raw packed bytes and the dequantised floats, so
 *    tools/verify_expert.py can check them against that independently fetched fixture.
 *    If the reader computes an offset wrong, the bytes differ and it shows here.
 *
 * 2. SPEED, WHICH DECIDES WHETHER THE PROJECT WORKS AT ALL. A decode step routes to
 *    top-16 experts in each of 92 MoE layers: 1,472 expert loads, 17.55 MB each,
 *    25.83 GB per token if nothing is cached. Whether that is 4 seconds or 400 is a
 *    property of the storage, not of the code, and it has to be measured rather than
 *    assumed. Three regimes are timed:
 *      sequential cold  - experts in file order, page cache dropped
 *      random cold      - experts in shuffled order, page cache dropped   <- the real one
 *      warm             - the same reads served from page cache           <- the cache ceiling
 *    Random cold is what actually happens, because routing does not respect file order.
 *
 * usage: test_expert <shard_dir> [layer] [n_experts]
 */
/* glibc hides sync() behind __USE_MISC, which _POSIX_C_SOURCE alone does not set, so
 * it needs _DEFAULT_SOURCE too. Without it sync() is implicitly declared as returning
 * int when it returns void. */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "k3.h"
#include "k3_load.h"

#ifdef _WIN32
/* sync() and /proc/sys/vm/drop_caches are both Linux-only; drop_caches() below is only
 * ever reached when have_root() has already confirmed drop_caches exists, which it
 * never does on Windows, so this body is dead code at runtime -- it only needs to
 * compile, not do anything. */
static inline void sync(void) { }
#endif

/* Non-zero if any correctness check failed; becomes the exit status. */
static int g_bad = 0;

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void drop_caches(void)
{
    /* Without this the second run reads from RAM and reports a storage bandwidth that
     * does not exist. Needs root; if it fails the timings are labelled warm. */
    sync();
    FILE *f = fopen("/proc/sys/vm/drop_caches", "w");
    if (f) { fputs("3\n", f); fclose(f); }
}

static int have_root(void) { FILE *f = fopen("/proc/sys/vm/drop_caches", "w");
                             if (f) { fclose(f); return 1; } return 0; }

/* xorshift, so the shuffle is reproducible across runs and machines. */
static unsigned rs = 12345u;
static unsigned rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: test_expert <shard_dir> [layer] [n]\n"); return 2; }
    const char *dir = argv[1];
    const int layer = argc > 2 ? atoi(argv[2]) : 1;
    int n = argc > 3 ? atoi(argv[3]) : 64;

    K3St s;
    double t0 = now_s();
    if (k3_st_open(&s, dir) != 0) return 1;
    printf("indexed %d tensors from %d shard(s) in %.2f s\n\n", s.nt, s.nshard, now_s() - t0);

    /* ---- how many experts does this layer actually have? ---- */
    int have = 0;
    for (int e = 0; e < 1024; e++) {
        char nm[256];
        snprintf(nm, sizeof nm,
                 "language_model.model.layers.%d.block_sparse_moe.experts.%d.w1.weight_packed",
                 layer, e);
        if (!k3_st_find(&s, nm)) break;
        have++;
    }
    printf("layer %d has %d routed experts present in this shard set\n", layer, have);
    if (have == 0) { fprintf(stderr, "no experts for layer %d here\n", layer); k3_st_close(&s); return 1; }
    if (n > have) n = have;

    /* ---- geometry and contiguity, checked across every expert ---- */
    K3ExpertRef r0;
    if (k3_expert_ref(&s, layer, 0, &r0) != 0) { k3_st_close(&s); return 1; }
    printf("expert 0 geometry:\n");
    static const char *W[3] = { "w1", "w2", "w3" };
    for (int i = 0; i < 3; i++)
        printf("  %s packed [%d][%d] -> logical [%d][%d], scales [%d][%d], group %d\n",
               W[i], r0.m[i].rows, r0.m[i].pcols, r0.m[i].rows, r0.m[i].pcols * 2,
               r0.m[i].rows, r0.m[i].scols, K3_MXFP4_GROUP);
    printf("  run: shard %d, offset %lld, %lld bytes, contiguous %s\n",
           r0.shard, (long long)r0.off, (long long)r0.nbytes, r0.contiguous ? "YES" : "NO");

    int64_t params = 0;
    for (int i = 0; i < 3; i++) params += k3_qmat_numel(&r0.m[i]);
    printf("  %lld parameters in %lld bytes = %.6f bytes/param (MXFP4 predicts %.6f)\n",
           (long long)params, (long long)r0.nbytes, (double)r0.nbytes / (double)params,
           0.5 + 1.0 / 32.0);

    int noncontig = 0, badgeom = 0;
    for (int e = 0; e < have; e++) {
        K3ExpertRef r;
        if (k3_expert_ref(&s, layer, e, &r) != 0) { badgeom++; continue; }
        if (!r.contiguous) noncontig++;
        if (r.nbytes != r0.nbytes) badgeom++;
    }
    printf("  across all %d experts: %d non-contiguous, %d with unexpected geometry%s\n\n",
           have, noncontig, badgeom, (noncontig || badgeom) ? "   <-- INVESTIGATE" : "");

    /* Stop here if any expert differs in size from expert 0.
     *
     * Everything below loads experts into a buffer sized for expert 0. Continuing past a
     * detected geometry anomaly would write a larger expert into that buffer, a heap
     * overflow, in the code path whose entire purpose was to warn about the anomaly. A
     * non-uniform expert bank is also a legitimate failure in its own right: the streaming
     * cache sizes every slot from one probe and assumes the rest match. */
    if (badgeom) {
        fprintf(stderr,
                "FAIL: %d of %d experts do not match expert 0's geometry.\n"
                "      The expert cache sizes all slots from a single probe, so a\n"
                "      non-uniform bank cannot be streamed safely. Refusing to load.\n",
                badgeom, have);
        return 1;
    }

    unsigned char *buf = (unsigned char *)malloc((size_t)r0.nbytes);
    if (!buf) { fprintf(stderr, "cannot allocate %lld bytes\n", (long long)r0.nbytes); return 1; }

    /* ---- correctness dump for expert 0, w1 ---- */
    if (k3_expert_load(&s, &r0, buf) != r0.nbytes) { fprintf(stderr, "load failed\n"); return 1; }
    {
        const int rows = 64;                       /* what fetch_mxfp4.py recorded */
        const int64_t wid = (int64_t)r0.m[0].pcols * 2;
        float *w1 = (float *)malloc((size_t)rows * wid * sizeof(float));
        k3_mxfp4_dequant(w1, buf + r0.m[0].p_off, buf + r0.m[0].s_off,
                         rows, r0.m[0].pcols, K3_MXFP4_GROUP);
        FILE *f = fopen("expert_dump.json", "w");
        fprintf(f, "{\"layer\":%d,\"expert\":0,\"rows\":%d,\"pcols\":%d,\"width\":%lld,",
                layer, rows, r0.m[0].pcols, (long long)wid);
        fprintf(f, "\"packed\":[");
        for (int64_t i = 0; i < (int64_t)rows * r0.m[0].pcols; i++)
            fprintf(f, "%s%u", i ? "," : "", buf[r0.m[0].p_off + i]);
        fprintf(f, "],\"scales\":[");
        for (int64_t i = 0; i < (int64_t)rows * r0.m[0].scols; i++)
            fprintf(f, "%s%u", i ? "," : "", buf[r0.m[0].s_off + i]);
        fprintf(f, "],\"bits\":[");
        for (int64_t i = 0; i < (int64_t)rows * wid; i++) {
            union { float f; uint32_t u; } b; b.f = w1[i];
            fprintf(f, "%s%u", i ? "," : "", b.u);
        }
        fprintf(f, "]}\n");
        fclose(f);
        double mn = 1e30, mx = -1e30, ss = 0.0; int64_t nz = 0;
        for (int64_t i = 0; i < (int64_t)rows * wid; i++) {
            if (w1[i] < mn) mn = w1[i];
            if (w1[i] > mx) mx = w1[i];
            ss += (double)w1[i] * w1[i];
            if (w1[i] != 0.0f) nz++;
        }
        printf("expert 0 w1, first %d rows dequantised:\n", rows);
        printf("  min %.6f  max %.6f  rms %.6f  nonzero %.1f%%\n",
               mn, mx, sqrt(ss / ((double)rows * wid)), 100.0 * nz / ((double)rows * wid));
        printf("  wrote expert_dump.json for tools/verify_expert.py\n\n");
        free(w1);
    }

    /* ---- the fused MXFP4 matmul must equal dequantise-then-multiply ----
     * k3_matmul_mxfp4 is what lets a cached expert stay 17.55 MB instead of 132 MB, so
     * it sits on the critical path of every routed token. It is a second, independent
     * implementation of the same arithmetic as k3_mxfp4_dequant followed by k3_matmul,
     * and the two must agree. They accumulate in a different ORDER (the fused version
     * sums within a 32-element group before applying the shared scale), so the
     * comparison is against double-precision rounding, not against zero. */
    {
        const int rows = r0.m[0].rows, wid = r0.m[0].pcols * 2;
        float *W = (float *)malloc((size_t)rows * wid * sizeof(float));
        float *x = (float *)malloc((size_t)wid * sizeof(float));
        float *ya = (float *)malloc((size_t)rows * sizeof(float));
        float *yb = (float *)malloc((size_t)rows * sizeof(float));
        if (W && x && ya && yb) {
            for (int i = 0; i < wid; i++) {
                rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5;
                x[i] = (float)(rs >> 8) / 8388608.0f - 1.0f;
            }
            double t1 = now_s();
            k3_mxfp4_dequant(W, buf + r0.m[0].p_off, buf + r0.m[0].s_off,
                             rows, r0.m[0].pcols, K3_MXFP4_GROUP);
            k3_matmul(ya, x, W, wid, rows);
            double t_deq = now_s() - t1;

            t1 = now_s();
            k3_matmul_mxfp4(yb, x, buf + r0.m[0].p_off, buf + r0.m[0].s_off,
                            wid, rows, K3_MXFP4_GROUP);
            double t_fus = now_s() - t1;

            double maxabs = 0.0, maxrel = 0.0, scale = 0.0;
            for (int i = 0; i < rows; i++) {
                const double d = fabs((double)ya[i] - (double)yb[i]);
                if (d > maxabs) maxabs = d;
                if (fabs(ya[i]) > scale) scale = fabs(ya[i]);
            }
            maxrel = scale > 0 ? maxabs / scale : 0.0;
            printf("fused MXFP4 matmul vs dequantise-then-matmul, w1 [%d][%d]:\n", rows, wid);
            /* This comparison is the ONLY correctness assertion in this program: it is
             * what proves the fused MXFP4 matmul agrees with dequantise-then-multiply on
             * released checkpoint bytes. The comparison must assert, not merely print, or a broken
             * nibble order or a mis-biased E8M0 scale printed "DISAGREE  <-- BUG" and the
             * process still exited 0. Record it so main can return it. */
            if (!(maxrel < 1e-6)) g_bad++;
            printf("  max abs diff %.3e, relative to max |y| %.3e   %s\n",
                   maxabs, maxrel, maxrel < 1e-6 ? "AGREE" : "DISAGREE  <-- BUG");
            printf("  dequant+matmul %.1f ms (plus %.1f MB materialised), fused %.1f ms\n",
                   t_deq * 1000.0, (double)rows * wid * 4 / 1e6, t_fus * 1000.0);
            printf("  fused reads %.2f MB instead of %.2f MB: %.1fx less memory traffic\n\n",
                   (double)(r0.m[0].p_bytes + r0.m[0].s_bytes) / 1e6,
                   (double)rows * wid * 4 / 1e6,
                   (double)rows * wid * 4 / (double)(r0.m[0].p_bytes + r0.m[0].s_bytes));
        }
        free(W); free(x); free(ya); free(yb);
    }

    /* ---- build the access orders ---- */
    int *seq = (int *)malloc((size_t)n * sizeof(int));
    int *shuf = (int *)malloc((size_t)have * sizeof(int));
    for (int i = 0; i < n; i++) seq[i] = i;
    for (int i = 0; i < have; i++) shuf[i] = i;
    for (int i = have - 1; i > 0; i--) { int j = (int)(rnd() % (unsigned)(i + 1));
                                         int t = shuf[i]; shuf[i] = shuf[j]; shuf[j] = t; }

    const int root = have_root();
    printf("page cache control: %s\n", root ? "available (cold timings are genuine)"
                                            : "UNAVAILABLE, run as root for cold timings");

    struct { const char *label; int *order; int cold; } runs[] = {
        { "sequential, cold", seq,  1 },
        { "random, cold    ", shuf, 1 },
        { "random, warm    ", shuf, 0 },
    };

    printf("\n%-18s %8s %10s %12s %12s\n", "ACCESS PATTERN", "EXPERTS", "SECONDS", "MB/s", "EXPERTS/s");
    printf("%s\n", "----------------------------------------------------------------");
    double rand_cold_mbs = 0.0;
    for (unsigned k = 0; k < sizeof runs / sizeof *runs; k++) {
        if (runs[k].cold) { if (!root) continue; drop_caches(); }
        double best_t = 0.0;
        int64_t bytes = 0;
        double t1 = now_s();
        for (int i = 0; i < n; i++) {
            K3ExpertRef r;
            if (k3_expert_ref(&s, layer, runs[k].order[i], &r) != 0) continue;
            bytes += k3_expert_load(&s, &r, buf);
        }
        best_t = now_s() - t1;
        double mbs = (double)bytes / 1e6 / best_t;
        printf("%-18s %8d %10.3f %12.0f %12.1f\n",
               runs[k].label, n, best_t, mbs, n / best_t);
        if (k == 1) rand_cold_mbs = mbs;
    }

    /* ---- what that means per token ---- */
    if (rand_cold_mbs > 0.0) {
        const double per_tok_gb = 1472.0 * (double)r0.nbytes / 1e9;   /* 92 layers x top-16 */
        printf("\nprojected, top-16 across 92 MoE layers, nothing cached:\n");
        printf("  %.2f GB of experts per token\n", per_tok_gb);
        printf("  at the measured random-cold %.0f MB/s: %.1f s/token (%.3f tok/s)\n",
               rand_cold_mbs, per_tok_gb * 1000.0 / rand_cold_mbs,
               rand_cold_mbs / (per_tok_gb * 1000.0));
        printf("  this is the FLOOR that caching and prefetch have to beat.\n");
    }

    free(buf); free(seq); free(shuf);
    k3_st_close(&s);
    if (g_bad) printf("\n%d CHECK(S) FAILED\n", g_bad);
    return g_bad ? 1 : 0;
}
