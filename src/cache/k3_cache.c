/* k3_cache.c - see k3_cache.h. */
#define _POSIX_C_SOURCE 200809L

#include "k3_portable_io.h"   /* first: sets _DARWIN_C_SOURCE before any libc header;
                                * on Windows, supplies posix_memalign */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef _WIN32
#include <sys/mman.h>   /* MADV_HUGEPAGE; k3_portable_io.h no-ops it on Windows */
#endif

#include "k3_cache.h"

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Resolve a slot to the three (packed, scale) pairs the kernels want. */
static void fill_q(const K3Cache *c, int slot, K3ExpertQ *q)
{
    /* pad is where the expert really begins: an O_DIRECT read starts at the enclosing
     * 4096 boundary, which is at or before the expert's own offset. */
    const unsigned char *b = c->arena + (size_t)slot * c->slot_bytes + c->pad[slot];
    const K3ExpertRef *r = &c->ref[slot];
    q->p1 = b + r->m[0].p_off; q->s1 = b + r->m[0].s_off;
    q->p2 = b + r->m[1].p_off; q->s2 = b + r->m[1].s_off;
    q->p3 = b + r->m[2].p_off; q->s3 = b + r->m[2].s_off;
}

/* Least recently used unpinned slot. Linear, deliberately: a few hundred comparisons
 * against a 17.55 MB read is not where the time goes. */
/* key_of[] has THREE states, not two:
 *     >= 0            holds that key
 *     K3_SLOT_EMPTY   holds nothing, free to take
 *     K3_SLOT_INFLIGHT reserved by a batch prefetch whose read has not finished
 *
 * The third state exists because of a real bug. The batch prefetch marks a slot empty
 * before reading into it, so that a failed read cannot leave the slot claiming an expert
 * it does not hold. But the empty test below is a FAST PATH that returns immediately,
 * ahead of the pinned check and the LRU scan -- so the next expert in the same batch was
 * handed the SAME slot, several parallel reads wrote into one buffer, and the MoE
 * multiplied garbage. It cost one wrong token (65 instead of 2494) on the real model and
 * nothing at all in the fixtures, because no fixture exercises the streaming cache. */
static int pick_victim(K3Cache *c)
{
    int best = -1;
    uint64_t oldest = (uint64_t)-1;
    for (int i = 0; i < c->nslot; i++) {
        if (c->key_of[i] == K3_SLOT_INFLIGHT) continue;   /* being read into RIGHT NOW */
        if (c->key_of[i] == K3_SLOT_EMPTY) return i;      /* free, take it */
        if (c->pinned[i]) continue;
        if (c->used_at[i] < oldest) { oldest = c->used_at[i]; best = i; }
    }
    return best;
}

/* Bring (layer, expert) resident and return its slot, or -1. */
static int admit(K3Cache *c, int layer, int expert)
{
    const int32_t key = layer * c->n_experts + expert;
    int slot = c->slot_of[key];
    if (slot >= 0) {
        c->hits++;
        c->used_at[slot] = ++c->clock;
        return slot;
    }
    c->misses++;

    K3ExpertRef r;
    if (k3_expert_ref(c->st, layer, expert, &r) != 0) return -1;
    if (r.nbytes > c->slot_bytes) {
        fprintf(stderr, "k3_cache: L%d expert %d is %lld bytes, slot holds %lld\n",
                layer, expert, (long long)r.nbytes, (long long)c->slot_bytes);
        return -1;
    }

    slot = pick_victim(c);
    if (slot < 0) {
        fprintf(stderr, "k3_cache: every slot is pinned, cannot admit L%d expert %d\n",
                layer, expert);
        return -1;
    }
    if (c->key_of[slot] >= 0) { c->slot_of[c->key_of[slot]] = -1; c->evictions++; }

    const double t0 = now_s();
    int64_t pad = 0;
    const int64_t got = k3_expert_load_direct(c->st, &r,
                            c->arena + (size_t)slot * c->slot_bytes,
                            c->slot_bytes, &pad);
    c->load_seconds += now_s() - t0;
    if (got != r.nbytes) {
        fprintf(stderr, "k3_cache: short load of L%d expert %d (%lld of %lld)\n",
                layer, expert, (long long)got, (long long)r.nbytes);
        c->key_of[slot] = -1;
        return -1;
    }
    c->bytes_read += (uint64_t)got;

    c->ref[slot] = r;
    c->pad[slot] = (int32_t)pad;
    c->key_of[slot] = key;
    c->slot_of[key] = slot;
    c->used_at[slot] = ++c->clock;
    return slot;
}

/* Bring a whole top-k resident, with the reads issued CONCURRENTLY.
 *
 * The serial path admits one expert per call, so the drive sees a queue depth of one:
 * 17.55 MB, wait, repeat, 16 times per layer. NVMe needs depth to reach rated bandwidth,
 * so that pattern leaves most of the drive idle. This hands the whole set over at once.
 *
 * THREE PHASES, and the split is not cosmetic:
 *   1 SERIAL   resolve each miss and reserve it a slot. Slot allocation touches the LRU
 *              bookkeeping, which is shared mutable state and must not race.
 *   2 PARALLEL do the reads. Every read targets a distinct, already-assigned buffer and
 *              goes through pread, which takes its offset as an argument and so does not
 *              touch any shared file position. Nothing here is shared for writing.
 *   3 SERIAL   publish. A slot is registered to its key ONLY after its read succeeded.
 *
 * Phase 3 is where the danger was. Registering the key up front, then reading, would
 * leave a failed read with a slot that claims to hold an expert it does not -- and the
 * next request for that expert would count a HIT and multiply garbage. That exact bug
 * existed in the trunk ring and is why the order here is deliberate.
 */
static int cache_getmany(K3ExpertSrc *self, int layer, const int *ids, int n)
{
    K3Cache *c = (K3Cache *)self;
    if (n <= 0) return 0;

    typedef struct { int slot; int expert; K3ExpertRef r; int64_t got, pad; } Work;
    /* One entry per expert in a batch prefetch, so it is bounded by top-k. */
    Work w[K3_MAX_TOPK];
    int nw = 0;
    const int cap = (int)(sizeof w / sizeof *w);

    /* ---- phase 1: reserve, serially ---- */
    for (int i = 0; i < n && nw < cap; i++) {
        const int e = ids[i];
        if (e < 0 || e >= c->n_experts) continue;
        const int32_t key = layer * c->n_experts + e;
        if (c->slot_of[key] >= 0) continue;             /* already resident */

        int dup = 0;                                    /* the same id twice in one top-k */
        for (int j = 0; j < nw; j++) if (w[j].expert == e) { dup = 1; break; }
        if (dup) continue;

        K3ExpertRef r;
        if (k3_expert_ref(c->st, layer, e, &r) != 0) continue;
        if (r.nbytes > c->slot_bytes) continue;

        const int slot = pick_victim(c);
        if (slot < 0) break;
        if (c->key_of[slot] >= 0) { c->slot_of[c->key_of[slot]] = -1; c->evictions++; }
        /* INFLIGHT, not EMPTY. Marking it empty made pick_victim's fast path hand the
         * same slot to the next expert in this very batch. */
        c->key_of[slot] = K3_SLOT_INFLIGHT;
        c->used_at[slot] = ++c->clock;

        w[nw].slot = slot; w[nw].expert = e; w[nw].r = r; w[nw].got = -1; w[nw].pad = 0;
        nw++;
    }
    if (nw == 0) return 0;

    /* Issue in DISK-OFFSET order. Experts are not stored id-ordered inside a shard, so
     * sorting by where the bytes actually live turns a scattered set of seeks into a
     * mostly forward sweep. Insertion sort: nw is at most the top-k. */
    for (int i = 1; i < nw; i++) {
        Work t = w[i]; int j = i - 1;
        while (j >= 0 && (w[j].r.shard > t.r.shard ||
                         (w[j].r.shard == t.r.shard && w[j].r.off > t.r.off))) {
            w[j + 1] = w[j]; j--;
        }
        w[j + 1] = t;
    }

    /* ---- phase 2: read, concurrently ---- */
    const double t0 = now_s();
#ifdef _OPENMP
#   pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int i = 0; i < nw; i++) {
        int64_t pad = 0;
        const int64_t got = k3_expert_load_direct(
            c->st, &w[i].r, c->arena + (size_t)w[i].slot * c->slot_bytes,
            c->slot_bytes, &pad);
        w[i].got = got;
        w[i].pad = pad;
    }
    c->load_seconds += now_s() - t0;

    /* ---- phase 3: publish only what actually arrived ---- */
    int ok = 0;
    for (int i = 0; i < nw; i++) {
        if (w[i].got != w[i].r.nbytes) {
            fprintf(stderr, "k3_cache: short prefetch of L%d expert %d (%lld of %lld); "
                            "leaving the slot empty so it cannot be served as a hit\n",
                    layer, w[i].expert, (long long)w[i].got, (long long)w[i].r.nbytes);
            c->key_of[w[i].slot] = K3_SLOT_EMPTY;       /* release the reservation */
            continue;
        }
        const int32_t key = layer * c->n_experts + w[i].expert;
        c->ref[w[i].slot] = w[i].r;
        c->pad[w[i].slot] = (int32_t)w[i].pad;
        c->key_of[w[i].slot] = key;
        c->slot_of[key] = w[i].slot;
        c->used_at[w[i].slot] = ++c->clock;
        c->bytes_read += (uint64_t)w[i].got;
        c->prefetch_reads++;
        ok++;
    }
    return ok;
}

/* Is this expert already resident, i.e. would get() serve it with no disk read? Used by
 * the draft model's cache-only routing to propose tokens without any expert I/O; if it
 * is resident, fill_q hands back the same bytes get() would. */
static int cache_resident(K3ExpertSrc *self, int layer, int expert, K3ExpertQ *out)
{
    K3Cache *c = (K3Cache *)self;
    if (layer < 0 || layer >= c->n_layers || expert < 0 || expert >= c->n_experts)
        return 0;
    const int32_t key = layer * c->n_experts + expert;
    const int slot = c->slot_of[key];
    if (slot < 0) return 0;
    if (out) fill_q(c, slot, out);
    return 1;
}

static int cache_get(K3ExpertSrc *self, int layer, int expert, K3ExpertQ *out)
{
    K3Cache *c = (K3Cache *)self;          /* src is the first member, by contract */
    if (layer < 0 || layer >= c->n_layers || expert < 0 || expert >= c->n_experts) {
        fprintf(stderr, "k3_cache: out of range L%d expert %d\n", layer, expert);
        return -1;
    }
    c->hist[layer * c->n_experts + expert]++;

    /* Record the request before serving it. The trace must reflect what the MODEL
     * asked for, independent of what the cache happened to hold, or replaying it at a
     * different capacity would be meaningless. */
    if (c->ntrace + 2 > c->captrace) {
        int64_t nc = c->captrace ? c->captrace * 2 : (1 << 16);
        int32_t *nt = (int32_t *)realloc(c->trace, (size_t)nc * sizeof(int32_t));
        if (nt) { c->trace = nt; c->captrace = nc; }
    }
    if (c->ntrace + 2 <= c->captrace) {
        c->trace[c->ntrace++] = layer;
        c->trace[c->ntrace++] = expert;
    }

    const int slot = admit(c, layer, expert);
    if (slot < 0) return -1;
    fill_q(c, slot, out);
    return 0;
}

int k3_cache_init(K3Cache *c, const K3St *st, const K3Cfg *cfg, int64_t budget_bytes)
{
    memset(c, 0, sizeof *c);
    c->src.get = cache_get;
    c->src.resident = cache_resident;
    /* K3_NOPREFETCH=1 disables the batch path at runtime. An A/B between two BUILDS
     * compares two binaries; an A/B on one binary compares one decision, which is the
     * only way to attribute a timing difference to the prefetch rather than to the
     * compiler, the layout, or the weather. */
    c->src.getmany = getenv("K3_NOPREFETCH") ? NULL : cache_getmany;
    if (!c->src.getmany)
        fprintf(stderr, "k3_cache: batch prefetch DISABLED by K3_NOPREFETCH\n");
    c->src.ctx = c;
    c->st = st;
    c->n_layers = cfg->n_layers;
    c->n_experts = cfg->n_experts;

    /* Size a slot from the checkpoint rather than from arithmetic: find any expert and
     * ask how many bytes it actually occupies. */
    K3ExpertRef probe;
    int found = 0;
    for (int L = 0; L < cfg->n_layers && !found; L++) {
        if (k3_is_dense(cfg, L)) continue;
        if (k3_expert_ref(st, L, 0, &probe) == 0) found = 1;
    }
    if (!found) { fprintf(stderr, "k3_cache: no routed experts in this shard set\n"); return -1; }
    /* Room for an O_DIRECT read widened outward to 4096 boundaries at both ends. */
    /* Round the SLOT STRIDE up to the O_DIRECT alignment, not just the arena base.
     *
     * posix_memalign below aligns the arena, which aligns slot 0 and nothing else: slot
     * N starts at arena + N*slot_bytes, so every slot is aligned only if slot_bytes is
     * itself a multiple of K3_ST_ALIGN. On the real checkpoint an expert is 17,547,264
     * bytes, which is exactly 4284 * 4096, so this held BY COINCIDENCE and the engine
     * worked. With any other expert size -- another model, a repacked container, or the
     * few-KB experts in tests/fixtures/cache -- every O_DIRECT read into every slot
     * after the first returns 0 bytes and the cache silently serves nothing. The
     * fixture deliberately uses a non-conforming expert size so this is gated rather
     * than left to the real checkpoint's coincidence (tests/unit/test_cache.c). */
    c->slot_bytes = probe.nbytes + 2 * K3_ST_ALIGN;
    c->slot_bytes = (c->slot_bytes + K3_ST_ALIGN - 1) & ~(int64_t)(K3_ST_ALIGN - 1);

    c->nslot = (int)(budget_bytes / c->slot_bytes);
    if (c->nslot < cfg->topk + 1) {
        fprintf(stderr,
                "k3_cache: budget %.2f GB gives %d slots of %.2f MB, but top-%d needs at "
                "least %d. A cache smaller than one token's working set would evict an "
                "expert that is still being multiplied.\n",
                (double)budget_bytes / 1e9, c->nslot, (double)c->slot_bytes / 1e6,
                cfg->topk, cfg->topk + 1);
        return -1;
    }

    /* Page aligned so the arena can later be read into with O_DIRECT unchanged. */
    /* 2 MB aligned and hugepage-advised, for the same reason as the trunk arena: every
     * O_DIRECT expert read pins its destination pages, and a 17.55 MB slot on 4 KB pages
     * is 4,284 pins per read, 1,472 reads per token. See k3_trunk.c:k3_alloc_direct.
     * K3_NOHUGE=1 restores 4 KB so the two can be compared on one binary. */
    {
        const int huge = !getenv("K3_NOHUGE");
        const size_t al = huge ? (2u << 20) : 4096u;
        size_t want = (size_t)c->nslot * c->slot_bytes;
        if (huge) want = (want + al - 1) & ~(al - 1);
        if (posix_memalign((void **)&c->arena, al, want) != 0) {
            fprintf(stderr, "k3_cache: cannot allocate %.2f GB arena\n", (double)want / 1e9);
            return -1;
        }
#if defined(MADV_HUGEPAGE)
        if (huge) madvise(c->arena, want, MADV_HUGEPAGE);
#endif
    }
    if (0) {
        fprintf(stderr, "k3_cache: cannot allocate %.2f GB arena\n",
                (double)c->nslot * c->slot_bytes / 1e9);
        return -1;
    }

    const size_t nkey = (size_t)c->n_layers * c->n_experts;
    c->slot_of = (int32_t *)malloc(nkey * sizeof(int32_t));
    c->key_of  = (int32_t *)malloc((size_t)c->nslot * sizeof(int32_t));
    c->used_at = (uint64_t *)calloc((size_t)c->nslot, sizeof(uint64_t));
    c->pinned  = (unsigned char *)calloc((size_t)c->nslot, 1);
    c->ref     = (K3ExpertRef *)calloc((size_t)c->nslot, sizeof(K3ExpertRef));
    c->pad     = (int32_t *)calloc((size_t)c->nslot, sizeof(int32_t));
    c->hist    = (uint32_t *)calloc(nkey, sizeof(uint32_t));
    if (!c->slot_of || !c->key_of || !c->used_at || !c->pinned || !c->ref || !c->pad || !c->hist) {
        k3_cache_free(c); return -1;
    }
    for (size_t i = 0; i < nkey; i++) c->slot_of[i] = -1;
    for (int i = 0; i < c->nslot; i++) c->key_of[i] = -1;
    return 0;
}

void k3_cache_free(K3Cache *c)
{
    k3_aligned_free(c->arena); free(c->slot_of); free(c->key_of);
    free(c->used_at); free(c->pinned); free(c->ref); free(c->pad); free(c->hist);
    free(c->trace);
    memset(c, 0, sizeof *c);
}

int k3_cache_dump_trace(const K3Cache *c, const char *path)
{
    if (!c->trace || c->ntrace == 0) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    const size_t n = fwrite(c->trace, sizeof(int32_t), (size_t)c->ntrace, f);
    fclose(f);
    printf("wrote %s: %lld requests (%.1f KB)\n",
           path, (long long)(c->ntrace / 2), (double)c->ntrace * 4 / 1024.0);
    return n == (size_t)c->ntrace ? 0 : -1;
}

int k3_cache_pin(K3Cache *c, int layer, int expert, int pin)
{
    const int32_t key = layer * c->n_experts + expert;
    if (key < 0 || key >= c->n_layers * c->n_experts) return 0;
    const int slot = c->slot_of[key];
    if (slot < 0) return 0;
    c->pinned[slot] = pin ? 1 : 0;
    return 1;
}

int k3_cache_prefetch(K3Cache *c, int layer, int expert)
{
    return admit(c, layer, expert) >= 0 ? 0 : -1;
}

void k3_cache_reset_stats(K3Cache *c)
{
    c->hits = c->misses = c->evictions = c->bytes_read = 0;
    c->load_seconds = 0.0;
    /* prefetch_reads belongs to the same window as hits and misses.
     *
     * k3_cache_report derives the effective hit rate as (hits - prefetch_reads), so both
     * counters must cover the same interval. Resetting one without the other compares a
     * per-window numerator against a since-startup subtrahend, which drives the result
     * negative and clamps it to zero at every cache size. */
    c->prefetch_reads = 0;
}

void k3_cache_report(const K3Cache *c, const char *label)
{
    const uint64_t n = c->hits + c->misses;
    int resident = 0, pinned = 0;
    for (int i = 0; i < c->nslot; i++) { if (c->key_of[i] >= 0) resident++; if (c->pinned[i]) pinned++; }
    printf("cache [%s]\n", label ? label : "");
    printf("  slots        : %d of %.2f MB = %.2f GB arena (%d resident, %d pinned)\n",
           c->nslot, (double)c->slot_bytes / 1e6,
           (double)c->nslot * c->slot_bytes / 1e9, resident, pinned);
    printf("  requests     : %llu  hits %llu (%.2f%%)  misses %llu  evictions %llu\n",
           (unsigned long long)n, (unsigned long long)c->hits,
           n ? 100.0 * c->hits / n : 0.0,
           (unsigned long long)c->misses, (unsigned long long)c->evictions);
    /* The prefetch makes the raw hit rate above flattering: an expert the batch read
     * from disk moments earlier is resident by the time get() asks, so it counts as a
     * hit. Report what was actually served from RAM without touching the disk. */
    if (c->prefetch_reads) {
        const unsigned long long served = (c->hits > c->prefetch_reads)
                                        ? c->hits - c->prefetch_reads : 0;
        printf("  of those hits : %llu came from the batch prefetch, i.e. read from disk\n"
               "                  this token; TRUE resident hit rate %.2f%%\n",
               (unsigned long long)c->prefetch_reads, n ? 100.0 * served / n : 0.0);
    }
    printf("  read from disk: %.2f GB in %.2f s (%.0f MB/s while loading)\n",
           (double)c->bytes_read / 1e9, c->load_seconds,
           c->load_seconds > 0 ? (double)c->bytes_read / 1e6 / c->load_seconds : 0.0);
}

int k3_cache_dump_hist(const K3Cache *c, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "{\"n_layers\":%d,\"n_experts\":%d,\"counts\":{",
            c->n_layers, c->n_experts);
    int first = 1;
    for (int L = 0; L < c->n_layers; L++) {
        for (int e = 0; e < c->n_experts; e++) {
            const uint32_t v = c->hist[L * c->n_experts + e];
            if (!v) continue;                       /* sparse: most are zero */
            fprintf(f, "%s\"%d,%d\":%u", first ? "" : ",", L, e, v);
            first = 0;
        }
    }
    fprintf(f, "}}\n");
    fclose(f);
    return 0;
}
