/* Is k3_matmul_bf16_batch bitwise identical to calling k3_matmul_bf16 T times?
 *
 * Not "close", not "within tolerance" -- IDENTICAL. The engine's correctness story rests
 * on the AVX2, NEON and scalar matmuls agreeing bit for bit (test_ops asserts it) and on
 * the full-model oracle reproducing the reference exactly. A batched kernel that changed
 * the last ulp would pass a tolerance test, fail the oracle, and be much harder to debug
 * than one that fails here.
 *
 * Shapes chosen to exercise the parts that are easy to get wrong:
 *   - `in` NOT a multiple of 16, so the ragged tail runs and must use the same order
 *   - `out` both above and below the OpenMP threshold of 64
 *   - T = 1 (must take the single-vector path), 2, 8, and K3_BATCH_MAX
 *   - denormal-ish and large exponents, since double accumulation order only shows up
 *     when magnitudes differ enough to make addition non-associative
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3.h"

static uint32_t rng_state = 0x1234567u;
static uint32_t rnd(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

/* Spread across exponents on purpose: if every value has a similar magnitude, summation
 * is effectively associative and a reordering bug stays invisible. */
static float spicy(void)
{
    const int e = (int)(rnd() % 19) - 9;
    const float m = (float)((double)(rnd() % 2000) / 1000.0 - 1.0);
    return (float)ldexp((double)m, e);
}

static int check(int in, int out, int T)
{
    uint16_t *W = (uint16_t *)malloc((size_t)in * out * sizeof(uint16_t));
    float *x  = (float *)malloc((size_t)in * T * sizeof(float));
    float *ya = (float *)malloc((size_t)out * T * sizeof(float));
    float *yb = (float *)malloc((size_t)out * T * sizeof(float));
    if (!W || !x || !ya || !yb) { fprintf(stderr, "OOM\n"); exit(1); }

    for (size_t i = 0; i < (size_t)in * out; i++) {
        union { float f; uint32_t u; } v;
        v.f = spicy();
        W[i] = (uint16_t)(v.u >> 16);          /* a real bf16: the top half of an f32 */
    }
    for (size_t i = 0; i < (size_t)in * T; i++) x[i] = spicy();

    for (int t = 0; t < T; t++)
        k3_matmul_bf16(ya + (size_t)t * out, x + (size_t)t * in, W, in, out);
    k3_matmul_bf16_batch(yb, out, x, in, W, in, out, T);

    /* memcmp, not a tolerance: NaN would compare unequal under ==, and a one-ulp drift is
     * exactly the failure this test exists to catch. */
    const int same = memcmp(ya, yb, (size_t)out * T * sizeof(float)) == 0;
    int first = -1;
    if (!same) {
        for (int i = 0; i < out * T; i++)
            if (memcmp(&ya[i], &yb[i], sizeof(float)) != 0) { first = i; break; }
    }
    printf("  %-5s in=%-6d out=%-6d T=%-3d", same ? "PASS" : "FAIL", in, out, T);
    if (!same) {
        printf("   first differing element %d: serial %.17g batched %.17g",
               first, (double)ya[first], (double)yb[first]);
    }
    printf("\n");

    free(W); free(x); free(ya); free(yb);
    return same;
}

int main(void)
{
    printf("batched bf16 matmul must be BITWISE identical to the serial kernel\n\n");
    int ok = 1;
    ok &= check(7168, 128, 1);          /* T=1 must take the single-vector path */
    ok &= check(7168, 128, 2);
    ok &= check(7168, 128, 8);
    ok &= check(7168, 128, K3_BATCH_MAX);
    ok &= check(1023, 96,  8);          /* in not a multiple of 16: ragged tail */
    ok &= check(1023, 33,  K3_BATCH_MAX); /* out below the OpenMP threshold */
    ok &= check(17,   65,  3);          /* tiny in, out just over the threshold */
    ok &= check(4096, 512, 16);
    printf("\n%s\n", ok ? "BATCHED MATMUL IS BIT-EXACT" : "BIT-EXACTNESS BROKEN");
    return ok ? 0 : 1;
}
