/* Does the BIT-EXACT batched kernel still deliver the amortisation?
 *
 * The isolated prototype reached 5.58x at T=16, but it kept ONE accumulator per output.
 * The real kernel must keep SIXTEEN PER ACTIVATION to preserve k3_matmul_bf16's reduction
 * tree -- at T=16 that is 256 doubles, 2 KB per output row, far past what stays in
 * registers. Exactness may have eaten the speedup, and the only way to know is to measure
 * the kernel that actually ships rather than the one that was convenient to prototype.
 *
 * Real KDA projection shape, several matrices cycled so the working set is well past the
 * 96 MB V-Cache -- the engine gets no reuse either, and a benchmark that fits in cache
 * reports a speed the engine never sees.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "k3.h"

#define IN   7168
#define OUT  12288
#define NMAT 6

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(void)
{
    static uint16_t *W[NMAT];
    const size_t wn = (size_t)IN * OUT;
    for (int m = 0; m < NMAT; m++) {
        W[m] = (uint16_t *)malloc(wn * sizeof(uint16_t));
        if (!W[m]) { fprintf(stderr, "OOM\n"); return 1; }
        for (size_t i = 0; i < wn; i++) W[m][i] = (uint16_t)(0x3F80 ^ (i & 0x3FF));
    }
    float *x = (float *)malloc((size_t)K3_BATCH_MAX * IN * sizeof(float));
    float *y = (float *)malloc((size_t)K3_BATCH_MAX * OUT * sizeof(float));
    if (!x || !y) { fprintf(stderr, "OOM\n"); return 1; }
    for (int i = 0; i < K3_BATCH_MAX * IN; i++) x[i] = (float)((i % 17) - 8) * 0.01f;

    const double wbytes = (double)wn * 2.0;
    printf("%d x %d bf16 = %.1f MB, %d matrices = %.2f GB working set\n",
           IN, OUT, wbytes / 1e6, NMAT, NMAT * wbytes / 1e9);
    printf("the SHIPPING kernel, which keeps 16 accumulators per activation\n\n");
    printf("%3s  %12s %12s  %9s %12s\n", "T", "serial x T", "batched", "speedup", "GB/s eff");

    int mi = 0;
    for (int T = 1; T <= K3_BATCH_MAX; T *= 2) {
        double t0 = now_s();
        for (int t = 0; t < T; t++) {
            k3_matmul_bf16(y + (size_t)t * OUT, x + (size_t)t * IN, W[mi], IN, OUT);
            mi = (mi + 1) % NMAT;
        }
        const double ts = now_s() - t0;

        t0 = now_s();
        k3_matmul_bf16_batch(y, OUT, x, IN, W[mi], IN, OUT, T);
        mi = (mi + 1) % NMAT;
        const double tb = now_s() - t0;

        printf("%3d  %12.4f %12.4f  %8.2fx %12.1f\n",
               T, ts, tb, ts / tb, wbytes / tb / 1e9);
        fflush(stdout);
    }
    for (int m = 0; m < NMAT; m++) free(W[m]);
    free(x); free(y);
    return 0;
}
