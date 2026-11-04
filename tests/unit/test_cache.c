/* test_cache.c - the streaming expert cache, which nothing else tests.
 *
 * WHY THIS FILE EXISTS
 *   Every op fixture and all three oracle gates drive the RESIDENT expert bank
 *   (K3MoeW.w1/w3/w2). The streaming path (K3MoeW.src -> K3Cache) is reached only when
 *   running the real 1.56 TB checkpoint. So the component the entire project depends on
 *   had no test at all, and it showed: a batch-prefetch bug that handed one slot to
 *   several experts simultaneously passed 22/22 fixtures and all three gates, and was
 *   caught only because the real model emitted token 65 where it should have emitted
 *   2494. A test that cannot fail is worse than no test; a path with no test is worse
 *   still.
 *
 * WHAT IS CHECKED
 *   1 IDENTITY      every expert read back through the cache is byte-identical to the
 *                   same expert read straight off disk. This is the check the aliasing
 *                   bug fails.
 *   2 EQUIVALENCE   the batch prefetch and the serial path return the SAME bytes. The
 *                   prefetch is an optimisation; if it changes a byte it is wrong.
 *   3 PRESSURE      with a cache far smaller than the working set, so eviction runs
 *                   constantly and slots are recycled aggressively. The bug only
 *                   appeared under pressure, because with a roomy cache pick_victim
 *                   returns genuinely free slots and the aliasing never happens.
 *   4 ACCOUNTING    requests, hits and prefetch_reads stay mutually consistent.
 *
 * usage: test_cache <fixture_dir> [n_experts]
 *        fixture_dir comes from tools/make_cache_fixture.py
 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3.h"
#include "k3_cache.h"
#include "k3_load.h"
#include "k3_st.h"

static int g_fail = 0;

static void ck(int ok, const char *what, const char *detail)
{
    printf("  %s  %-34s %s\n", ok ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!ok) g_fail++;
}

/* Read an expert straight from the store, bypassing the cache entirely. This is the
 * ground truth the cache is measured against. */
static unsigned char *direct_read(const K3St *st, int layer, int e, int64_t *nb)
{
    K3ExpertRef r;
    if (k3_expert_ref(st, layer, e, &r) != 0) return NULL;
    unsigned char *b = (unsigned char *)malloc((size_t)r.nbytes);
    if (!b) return NULL;
    if (k3_expert_load(st, &r, b) != r.nbytes) { free(b); return NULL; }
    *nb = r.nbytes;
    return b;
}

/* The three (packed, scale) pairs the cache hands out must point at bytes equal to the
 * direct read. Compare through the SAME offsets the kernels use, so a wrong pad or a
 * wrong slot base is caught, not just a wrong buffer. */
static int same_expert(const K3St *st, int layer, int e, const K3ExpertQ *q)
{
    int64_t nb = 0;
    unsigned char *truth = direct_read(st, layer, e, &nb);
    if (!truth) return 0;
    K3ExpertRef r;
    if (k3_expert_ref(st, layer, e, &r) != 0) { free(truth); return 0; }

    const unsigned char *got[6] = { q->p1, q->s1, q->p2, q->s2, q->p3, q->s3 };
    int ok = 1;
    for (int i = 0; i < 3 && ok; i++) {
        const unsigned char *tp = truth + r.m[i].p_off, *ts = truth + r.m[i].s_off;
        const int64_t pn = (int64_t)r.m[i].rows * r.m[i].pcols;
        const int64_t sn = (int64_t)r.m[i].rows * r.m[i].scols;
        if (memcmp(got[i * 2], tp, (size_t)pn) != 0) ok = 0;
        if (memcmp(got[i * 2 + 1], ts, (size_t)sn) != 0) ok = 0;
    }
    free(truth);
    return ok;
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "../fixtures/cache";
    const int NE = argc > 2 ? atoi(argv[2]) : 24;

    K3St st;
    if (k3_st_open(&st, dir) != 0) {
        fprintf(stderr, "TEST ABORTED: cannot open %s\n"
                        "  build it with: python3 tools/make_cache_fixture.py\n", dir);
        return 2;
    }
    printf("streaming expert cache, %d tensors from %d shard(s)\n\n", st.nt, st.nshard);

    K3Cfg c; memset(&c, 0, sizeof c);
    c.n_layers = 1; c.n_experts = NE; c.topk = 4;

    /* Size the budget by asking, not by arithmetic. The cache rounds a slot up to an
     * O_DIRECT-widened, page-aligned size, so for these deliberately tiny fixture
     * experts the requested budget and the usable slot count part company badly. Grow
     * until init accepts, then confirm PRESSURE remains: fewer slots than experts,
     * so eviction actually runs. A roomy cache hides slot-recycling bugs entirely. */
    K3ExpertRef probe;
    if (k3_expert_ref(&st, 0, 0, &probe) != 0) { fprintf(stderr, "no expert 0\n"); return 2; }
    K3Cache cache;
    int ok_init = 0;
    for (int64_t budget = probe.nbytes * 8; budget <= probe.nbytes * 4096; budget *= 2) {
        if (k3_cache_init(&cache, &st, &c, budget) == 0) { ok_init = 1; break; }
    }
    if (!ok_init) { fprintf(stderr, "cache init failed at every budget\n"); return 2; }
    { char b[80]; snprintf(b, sizeof b, "%d slots for %d experts, top-%d",
                           cache.nslot, NE, c.topk);
      ck(cache.nslot >= c.topk + 1 && cache.nslot < NE, "cache under pressure", b); }

    /* ---- 1+3: serial path, every expert, under eviction pressure ---- */
    int bad = 0;
    for (int pass = 0; pass < 3; pass++)
        for (int e = 0; e < NE; e++) {
            K3ExpertQ q;
            if (cache.src.get(&cache.src, 0, e, &q) != 0) { bad++; continue; }
            if (!same_expert(&st, 0, e, &q)) bad++;
        }
    { char b[64]; snprintf(b, sizeof b, "%d of %d reads wrong", bad, 3 * NE);
      ck(bad == 0, "serial reads are byte-exact", b); }

    /* ---- 2: batch prefetch must agree with the serial path ----
     * This is the check the aliasing bug fails. Ask for a whole top-k at once, with the
     * cache too small to hold the previous batch, then verify EVERY expert. */
    k3_cache_reset_stats(&cache);
    int bad2 = 0, batches = 0;
    if (!cache.src.getmany) {
        ck(0, "batch prefetch present", "getmany is NULL");
    } else {
        for (int start = 0; start + c.topk <= NE; start += c.topk) {
            int ids[16];
            for (int j = 0; j < c.topk; j++) ids[j] = start + j;
            cache.src.getmany(&cache.src, 0, ids, c.topk);
            batches++;
            for (int j = 0; j < c.topk; j++) {
                K3ExpertQ q;
                if (cache.src.get(&cache.src, 0, ids[j], &q) != 0) { bad2++; continue; }
                if (!same_expert(&st, 0, ids[j], &q)) bad2++;
            }
        }
        char b[96];
        snprintf(b, sizeof b, "%d batches of %d, %d wrong", batches, c.topk, bad2);
        ck(bad2 == 0, "batch prefetch is byte-exact", b);

        /* ACCOUNTING, checked HERE and only here. The loop above is exactly the pattern
         * k3_moe uses -- prefetch a top-k, then consume that same top-k -- and for that
         * pattern every prefetched expert is still resident when get() asks, so it is
         * recorded as a hit and prefetch_reads can never exceed hits. If it does, the
         * report's "true resident hit rate" underflows and the whole figure is a lie.
         * The mixed test below deliberately prefetches sets it never consumes, so the
         * invariant does NOT hold there and asserting it would be wrong. */
        char a[128];
        snprintf(a, sizeof a, "requests %llu, hits %llu, prefetch %llu",
                 (unsigned long long)(cache.hits + cache.misses),
                 (unsigned long long)cache.hits,
                 (unsigned long long)cache.prefetch_reads);
        ck(cache.prefetch_reads <= cache.hits, "prefetch_reads <= hits", a);
    }

    /* ---- 2b: interleaving the two paths must not corrupt either ---- */
    k3_cache_reset_stats(&cache);
    int bad3 = 0;
    for (int e = 0; e < NE; e++) {
        if (cache.src.getmany && (e % 2) == 0) {
            int ids[4];
            for (int j = 0; j < 4; j++) ids[j] = (e + j) % NE;
            cache.src.getmany(&cache.src, 0, ids, 4);
        }
        K3ExpertQ q;
        if (cache.src.get(&cache.src, 0, e, &q) != 0) { bad3++; continue; }
        if (!same_expert(&st, 0, e, &q)) bad3++;
    }
    { char b[64]; snprintf(b, sizeof b, "%d of %d wrong", bad3, NE);
      ck(bad3 == 0, "mixed batch and serial", b); }

    /* Linux's full device reports buffered write errors only on flush/close. */
    FILE *full = fopen("/dev/full", "wb");
    if (full) {
        fclose(full);
        ck(k3_cache_dump_hist(&cache, "/dev/full") != 0,
           "histogram flush failure is reported", "full device");
        ck(k3_cache_dump_trace(&cache, "/dev/full") != 0,
           "trace flush failure is reported", "full device");
    } else {
        printf("  SKIP  full-device cache output tests (no /dev/full)\n");
    }

    k3_cache_free(&cache);
    k3_st_close(&st);
    printf("\n%s\n", g_fail ? "CACHE TESTS FAILED" : "CACHE TESTS PASSED");
    return g_fail ? 1 : 0;
}
