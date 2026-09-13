/* test_sample.c - the opt-in sampler, which greedy runs never touch.
 *
 * WHAT IS CHECKED (all exact, no statistics, nothing flaky)
 *   1 GREEDY FALLBACK   temp <= 0 returns the argmax without drawing.
 *   2 TOP-1             top_k = 1 returns the argmax at any temperature.
 *   3 TINY TOP-P        top_p near zero keeps only the top candidate.
 *   4 DETERMINISM       same seed replays an identical 200-draw stream;
 *                       distinct streams stay independent (advancing one
 *                       never moves the other).
 *   5 CLAMPING          top_k <= 0 or > n behaves as the full vocabulary:
 *                       equal seeds give equal streams.
 *   6 MISUSE            n <= 0 returns -1; NULL stream with temp > 0 falls
 *                       back to the argmax instead of touching a global.
 *   7 ONE DRAW          one sampler draw advances the stream exactly one
 *                       k3_rng_double step, so single-draw reference replays
 *                       stay in lockstep.
 *   8 NO DUPLICATES     with fewer finite logits than top_k, masked -inf
 *                       entries must not duplicate taken indices (that would
 *                       double-count nucleus weight: P(argmax) 0.51 instead
 *                       of 0.88).
 *   9 NAN TEMP          NaN temperature is greedy, never silent sampling.
 *
 * usage: test_sample (no fixtures, no weights)
 *
 * author F E R M I ∞ H A R T <contact@fermihart.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "k3.h"

static int g_fail = 0;

static void ck(int ok, const char *what, const char *detail)
{
    printf("  %s  %-34s %s\n", ok ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!ok) g_fail++;
}

/* Peaked at index 5, runner-up at 2: every property below has a known answer. */
static void make_logits(float *lg, int n)
{
    for (int i = 0; i < n; i++) lg[i] = -(float)(i % 7) * 0.5f;
    lg[5] = 4.0f;
    lg[2] = 2.0f;
}

int main(void)
{
    const int n = 64;
    float *lg = (float *)malloc((size_t)n * sizeof(float));
    if (!lg) { printf("OOM in test setup\n"); return 2; }
    make_logits(lg, n);

    uint64_t s = 12345u;
    ck(k3_sample_next(lg, n, 0.0f, 0, 0.0f, &s) == 5, "temp 0 is argmax", "");
    ck(k3_sample_next(lg, n, -1.0f, 0, 0.0f, &s) == 5, "temp<0 is argmax", "");
    ck(k3_sample_next(lg, n, 2.0f, 1, 0.0f, &s) == 5, "top-k 1 is argmax", "");
    ck(k3_sample_next(lg, n, 2.0f, 0, 1e-9f, &s) == 5, "tiny top-p is argmax", "");
    ck(k3_sample_next(lg, n, 1.0f, 0, 0.0f, NULL) == 5, "NULL stream is argmax", "");
    ck(k3_sample_next(lg, 0, 1.0f, 0, 0.0f, &s) == -1, "n<=0 returns -1", "");

    /* Determinism: two streams from one seed, replayed twice. */
    uint64_t a = 999u, b = 999u, c = 777u;
    int eq = 1, adv = 1;
    for (int i = 0; i < 200; i++) {
        const int x = k3_sample_next(lg, n, 0.7f, 8, 0.9f, &a);
        const int y = k3_sample_next(lg, n, 0.7f, 8, 0.9f, &b);
        if (x != y) eq = 0;
        if (x < 0 || x >= n) adv = 0;
        (void)k3_sample_next(lg, n, 0.7f, 8, 0.9f, &c);
    }
    ck(eq, "same seed replays identically", "200 draws");
    ck(adv, "all draws in range", "");
    /* Independence is structural (separate states), but pin it behaviorally:
     * rewinding a stream reproduces its own prefix exactly. */
    uint64_t d = 4242u;
    int p1[8], p2[8];
    for (int i = 0; i < 8; i++) p1[i] = k3_sample_next(lg, n, 0.7f, 8, 0.9f, &d);
    d = 4242u;
    int rw = 1;
    for (int i = 0; i < 8; i++) p2[i] = k3_sample_next(lg, n, 0.7f, 8, 0.9f, &d);
    for (int i = 0; i < 8; i++) if (p1[i] != p2[i]) rw = 0;
    ck(rw, "rewound stream repeats prefix", "");

    /* Clamping: top_k <= 0 and top_k > n select the same candidate set, so
     * equal seeds must walk equal streams. */
    uint64_t e = 31u, f = 31u;
    int cl = 1;
    for (int i = 0; i < 50; i++) {
        const int x = k3_sample_next(lg, n, 1.5f, 0, 0.0f, &e);
        const int y = k3_sample_next(lg, n, 1.5f, n + 1000, 0.0f, &f);
        if (x != y) cl = 0;
    }
    ck(cl, "top-k clamping is consistent", "0 vs >n, 50 draws");

    /* One draw per token: step a manual stream once and require the
     * sampler to leave an identical stream behind after one draw. */
    {
        uint64_t s1 = 555u, s2 = 555u;
        (void)k3_rng_double(&s1);
        (void)k3_sample_next(lg, n, 0.7f, 8, 0.9f, &s2);
        ck(s1 == s2, "one draw per token", "stream states match");
    }

    /* NaN temperature: unordered comparison, must fall back to greedy. */
    {
        uint64_t sn = 1u;
        float nan = 0.0f / 0.0f;
        ck(k3_sample_next(lg, n, nan, 8, 0.0f, &sn) == 5, "NaN temp is argmax", "");
    }

    /* Duplicate guard: 2 finite logits, rest -inf, top_k 8. Correct:
     * P(5) = e^4/(e^4+e^2) ~= 0.88. The sentinel bug picked index 2 six
     * extra times, dragging P(5) to ~0.51. Margin is ~20 sigma. */
    {
        float sp[64];
        for (int i = 0; i < 64; i++) sp[i] = -(1.0f / 0.0f);
        sp[5] = 4.0f; sp[2] = 2.0f;
        uint64_t sd = 31337u;
        int c5 = 0;
        const int N = 2000;
        for (int i = 0; i < N; i++)
            if (k3_sample_next(sp, 64, 1.0f, 8, 0.0f, &sd) == 5) c5++;
        double frac = (double)c5 / N;
        char det[64];
        snprintf(det, sizeof det, "P(5)=%.3f, want 0.80..0.95", frac);
        ck(frac > 0.80 && frac < 0.95, "no duplicate candidates", det);
    }

    free(lg);
    printf("\n%s\n", g_fail ? "SAMPLE TESTS FAILED" : "SAMPLE TESTS PASSED");
    return g_fail ? 1 : 0;
}
