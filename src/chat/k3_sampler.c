/* k3_sampler.c - PCG32 plus deterministic temperature/top-p selection. */
#include "k3_sampler.h"

#include <math.h>
#include <stdlib.h>

static uint64_t splitmix64(uint64_t *x)
{
    *x += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t z = *x;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static uint32_t pcg32(K3Sampler *s)
{
    uint64_t old = s->state;
    s->state = old * UINT64_C(6364136223846793005) + s->inc;
    uint32_t x = (uint32_t)(((old >> 18) ^ old) >> 27);
    uint32_t rot = (uint32_t)(old >> 59);
    return (x >> rot) | (x << ((-rot) & 31));
}

static double unit(K3Sampler *s)
{
    /* 53 unbiased bits, so no platform PRNG or libc rand() leaks into output. */
    uint64_t x = ((uint64_t)pcg32(s) << 21) ^ (pcg32(s) & UINT32_C(0x1fffff));
    return (double)x * (1.0 / 9007199254740992.0);
}

void k3_sampler_init(K3Sampler *s, double temperature, double top_p,
                     uint64_t seed, uint64_t turn)
{
    uint64_t x = seed ^ (turn * UINT64_C(0x9e3779b97f4a7c15));
    uint64_t state = splitmix64(&x), seq = splitmix64(&x);
    s->temperature = temperature; s->top_p = top_p;
    s->state = 0; s->inc = (seq << 1u) | 1u;
    (void)pcg32(s); s->state += state; (void)pcg32(s);
    s->ids = NULL; s->prob = NULL; s->cap = 0;
}

void k3_sampler_free(K3Sampler *s)
{
    free(s->ids); free(s->prob); s->ids = NULL; s->prob = NULL; s->cap = 0;
}

static int cmp_desc(const void *aa, const void *bb, void *ctx)
{
    const double *p = (const double *)ctx;
    int a = *(const int *)aa, b = *(const int *)bb;
    if (p[a] > p[b]) return -1;
    if (p[a] < p[b]) return 1;
    return a < b ? -1 : a > b;
}

/* qsort_r has incompatible BSD/GNU argument order. A narrow static comparator keeps
 * the sampler portable C99; chat uses one sampler on the foreground REPL thread. */
static const double *sort_prob;
static int cmp_desc_global(const void *aa, const void *bb) { return cmp_desc(aa, bb, (void *)sort_prob); }

int k3_sampler_next(K3Sampler *s, const float *logits, int n, int greedy, int *out)
{
    if (!logits || !out || n <= 0 || !(s->temperature > 0.0) || !(s->top_p > 0.0) || s->top_p > 1.0) return -1;
    int best = 0;
    for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    if (greedy) { *out = best; return 0; }
    if (n > s->cap) {
        int *ids = (int *)malloc((size_t)n * sizeof(*ids));
        double *prob = (double *)malloc((size_t)n * sizeof(*prob));
        if (!ids || !prob) { free(ids); free(prob); return -1; }
        free(s->ids); free(s->prob);
        s->ids = ids; s->prob = prob; s->cap = n;
    }
    const double max = logits[best] / s->temperature;
    double sum = 0.0;
    for (int i = 0; i < n; i++) { s->ids[i] = i; s->prob[i] = exp((double)logits[i] / s->temperature - max); sum += s->prob[i]; }
    if (!(sum > 0.0) || !isfinite(sum)) { *out = best; return 0; }
    for (int i = 0; i < n; i++) s->prob[i] /= sum;
    sort_prob = s->prob; qsort(s->ids, (size_t)n, sizeof(*s->ids), cmp_desc_global); sort_prob = NULL;
    double keep = 0.0; int nk = 0;
    while (nk < n && keep < s->top_p) keep += s->prob[s->ids[nk++]];
    if (!nk) nk = 1;
    double r = unit(s) * keep, acc = 0.0;
    for (int i = 0; i < nk; i++) { int id = s->ids[i]; acc += s->prob[id]; if (r < acc || i + 1 == nk) { *out = id; return 0; } }
    return -1;
}
