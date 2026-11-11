/* k3_cache.h - the streaming expert cache. This is the part the project exists for.
 *
 * THE PROBLEM, IN MEASURED NUMBERS
 *   Routed experts are 1.45 TB of the 1.56 TB checkpoint. A decode step selects top-16
 *   in each of 92 MoE layers, so 1,472 experts, 17,547,264 bytes each: 25.83 GB of
 *   weights per token if nothing is cached. At the ~1.2 GB/s a commodity NVMe device
 *   sustains on cold random reads of this size, that alone is about 21 s/token. The
 *   cache exists to reduce it, and its hit rate is the single number that decides
 *   whether the whole approach works. Measured hit rates by budget, with the hardware
 *   named, are in docs/PERFORMANCE.md.
 *
 * WHY THE CACHE HOLDS MXFP4 AND NOT FLOATS
 *   Dequantised, one expert is 132 MB rather than 17.55 MB. Caching floats would cut
 *   the number of resident experts by 7.5x for no benefit, because k3_matmul_mxfp4
 *   consumes the packed form directly and a matrix-vector product is memory-bound:
 *   reading a seventh of the bytes is faster, not slower.
 *
 * REPLACEMENT POLICY
 *   LRU with pinning. Two things make
 *   plain LRU safe here:
 *     - k3_moe fetches an expert and immediately uses it, so a slot handed out cannot
 *       be evicted before use as long as capacity exceeds topk. The constructor
 *       enforces that rather than trusting it.
 *     - The victim search is a linear scan over slots. At a few hundred slots that is a
 *       few hundred comparisons against a 17.55 MB read; it is not worth a heap.
 *
 * THE HISTOGRAM IS NOT INSTRUMENTATION
 *   Which experts are hot is not knowable in advance and is the input to any sensible
 *   pinning strategy. The cache counts every request per (layer, expert), 82,432
 *   counters costing 330 KB, so a real workload can be profiled and the hot set pinned
 *   permanently. Without that measurement, pinning is guesswork.
 */
#ifndef K3_CACHE_H
#define K3_CACHE_H

#include "k3.h"
#include "k3_load.h"
#include "k3_st.h"

/* Slot states for key_of[]. EMPTY must stay -1: k3_cache_init memsets the array and
 * several places already test "< 0" meaning "not holding a key", which both sentinels
 * satisfy. INFLIGHT is distinct so pick_victim can refuse to hand out a slot whose read
 * has not landed yet. */
#define K3_SLOT_EMPTY     (-1)
#define K3_SLOT_INFLIGHT  (-2)

typedef struct {
    K3ExpertSrc  src;             /* MUST be first: pass &cache->src to K3MoeW */

    const K3St  *st;
    int          n_layers, n_experts;

    unsigned char *arena;         /* nslot * slot_bytes, page aligned          */
    int64_t      slot_bytes;
    int          nslot;

    int32_t     *slot_of;         /* [n_layers*n_experts] -> slot, or -1       */
    int32_t     *key_of;          /* [nslot] -> layer*n_experts+expert, or -1  */
    uint64_t    *used_at;         /* [nslot] LRU stamp                         */
    unsigned char *pinned;        /* [nslot] never evict while set             */
    K3ExpertRef *ref;             /* [nslot] geometry of the resident expert   */
    int32_t     *pad;             /* [nslot] where the payload starts in the slot;
                                   * non-zero only on the O_DIRECT path, where the
                                   * read is widened to aligned bounds             */

    uint64_t     clock;
    /* stats */
    uint64_t     hits, misses, evictions, bytes_read;
    /* Experts brought resident by the BATCH prefetch rather than by get().
     *
     * This has to be counted separately or the hit rate becomes a lie. A prefetched
     * expert is resident by the time get() asks for it, so get() records a hit -- but
     * the bytes still came off the disk during this token. Without this counter the
     * report would show the hit rate climbing while the I/O did not fall at all.
     * Effective hit rate is (hits - prefetch_reads) / requests. */
    uint64_t     prefetch_reads;
    double       load_seconds;
    uint32_t    *hist;            /* [n_layers*n_experts] request counts       */

    /* THE ACCESS TRACE, and why it is worth recording.
     * The question this project exists to answer is how much RAM Kimi K3 actually
     * needs, which is a question about hit rate versus cache size. Measuring that
     * directly would mean re-running the model once per cache size, and each run costs
     * real money on a machine big enough to hold the trunk.
     *
     * The routing decisions do not depend on the cache at all, so one run yields the
     * whole curve: record (layer, expert) in request order, then simulate any
     * replacement policy at any capacity offline. 1,472 requests per token, 8 bytes
     * each, is 12 KB per token. */
    int32_t     *trace;
    int64_t      ntrace, captrace;
} K3Cache;

/* budget_bytes is the arena size; it is rounded down to whole experts. Fails if that
 * leaves fewer than topk+1 slots, because a smaller cache cannot serve one token
 * without evicting an expert that is still in use. */
int  k3_cache_init(K3Cache *c, const K3St *st, const K3Cfg *cfg, int64_t budget_bytes);
void k3_cache_free(K3Cache *c);

/* Pin or unpin whatever slot currently holds this expert. Pinning a resident hot set
 * is the payoff from the histogram. Returns 0 if the expert was not resident. */
int  k3_cache_pin(K3Cache *c, int layer, int expert, int pin);

/* Load an expert without returning it, so a prefetcher can warm the cache. */
int  k3_cache_prefetch(K3Cache *c, int layer, int expert);

void k3_cache_reset_stats(K3Cache *c);
void k3_cache_report(const K3Cache *c, const char *label);

/* Write the request histogram as JSON, for offline analysis of the hot set. */
int  k3_cache_dump_hist(const K3Cache *c, const char *path);

/* Write the access trace as a flat binary file of int32 pairs (layer, expert) in
 * request order. tools/sim_cache.py replays it at any capacity. */
int  k3_cache_dump_trace(const K3Cache *c, const char *path);

/* Whether a window read as nchunk chunks of `chunk` bytes (the last one shorter, to
 * `len` total) holds its payload, which ends `need` bytes into the window. got[k] is
 * what chunk k returned.
 *
 * Summing got[] is not the same test. The window is the payload widened to whole
 * aligned blocks, so it carries up to K3_ST_ALIGN - 1 bytes of tail slack, and a sum
 * cannot tell "the last chunk ended at EOF in that slack" from "a middle chunk came
 * back short": a middle chunk losing no more than the slack still sums past `need`,
 * and the expert is published, and then served as a hit, with a hole in it. Only the
 * last chunk may be short, and only past `need`. */
static inline int k3_chunks_cover(const int64_t *got, int nchunk, int64_t chunk,
                                  int64_t len, int64_t need)
{
    for (int k = 0; k < nchunk; k++) {
        const int64_t base = (int64_t)k * chunk;
        const int64_t want = (k + 1 < nchunk) ? chunk
                           : (need - base > 0 ? need - base : 0);
        if (base >= len || got[k] < want) return 0;
    }
    return nchunk > 0;
}

#endif /* K3_CACHE_H */
