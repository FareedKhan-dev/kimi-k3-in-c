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
 * REPLACEMENT POLICY: S3-FIFO, NOT LRU
 *   The offline simulator (tools/sim_cache.py, replaying a real trace) put 25.5 points
 *   between what LRU achieves and what Belady's optimal policy achieves at 64 GB:
 *   36.24% against 61.74%. That gap is the largest single number in this project's own
 *   measurements, and it says the lever is the POLICY, not the size -- buying RAM moves
 *   the curve far less than closing that gap would.
 *
 *   K3's router is trained for flat expert usage, which is exactly the access pattern
 *   LRU handles worst: near-uniform popularity with weak, long-range reuse. LRU keeps
 *   whatever was touched most recently regardless of whether it will ever be touched
 *   again, so a one-off request evicts something with a real reuse history. Flat but
 *   NOT uniform is precisely what a frequency-aware policy captures.
 *
 *   S3-FIFO (Yang et al., SOSP 2023) is the right shape for this workload:
 *     - a SMALL FIFO takes every new admission, so one-hit wonders -- which dominate
 *       here -- are evicted quickly without ever polluting the main cache;
 *     - a MAIN FIFO holds objects that proved themselves by being touched while in the
 *       small queue, and reinserts them once each time they prove it again;
 *     - a GHOST queue remembers the KEYS recently evicted from the small queue, so an
 *       object that comes back promotes straight into the main queue instead of having
 *       to earn its way twice.
 *   Uniform object size makes this the friendliest possible case: every expert is
 *   exactly 17,547,264 bytes, so a slot is a slot and no cost-aware weighting is needed.
 *   The ghost queue is exact rather than a sketch, because the key space is only
 *   82,432 entries and a byte per key is 82 KB.
 *
 *   K3_CACHE_POLICY=lru restores the old policy on the same binary, which is the only
 *   way to attribute a hit-rate difference to the policy rather than to the run.
 *
 *   Two things still make eviction safe, and they are enforced rather than assumed:
 *     - the slots most recently handed out are never evicted, so an expert being
 *       multiplied cannot be overwritten underneath the kernel;
 *     - a slot whose read has not landed is never handed out.
 *
 * SPECULATIVE PREFETCH: THE PREVIOUS TOKEN'S ROUTING
 *   About 90% of expert requests in a real trace are repeats. When decode arrives at
 *   layer L for token t, layer L's top-16 for token t-1 is already known -- and it is a
 *   good guess, because that is what "90% repeats" means. So the moment the layer is
 *   entered, before the router has even run on the current hidden state, reads are
 *   issued for the PREVIOUS token's set on a background thread. By the time routing has
 *   actually happened, most of what it asks for is either resident or in flight.
 *
 *   A wrong guess costs bandwidth this engine is not otherwise using: it spends 40-77%
 *   of a token waiting on the disk. A right guess converts a blocking 17.55 MB read into
 *   a warm hit. And the two policies compose: a speculative expert that is never
 *   requested lands in the small FIFO and is evicted from there, which is exactly where
 *   a wrong guess belongs.
 *
 *   K3_NOSPEC=1 disables it. It is also refused automatically when the cache is too
 *   small to hold two tokens' worth of working set, because at that size speculation
 *   would evict what the current token still needs.
 *
 * THE HISTOGRAM IS NOT INSTRUMENTATION
 *   Which experts are hot is not knowable in advance and is the input to any sensible
 *   pinning strategy. The cache counts every request per (layer, expert), 82,432
 *   counters costing 330 KB, so a real workload can be profiled and the hot set pinned
 *   permanently. Without that measurement, pinning is guesswork.
 */
#ifndef K3_CACHE_H
#define K3_CACHE_H

#include <pthread.h>

#include "k3.h"
#include "k3_load.h"
#include "k3_st.h"

/* Slot states for key_of[]. EMPTY must stay -1: k3_cache_init memsets the array and
 * several places already test "< 0" meaning "not holding a key", which both sentinels
 * satisfy. INFLIGHT is distinct so the victim search can refuse to hand out a slot whose
 * read has not landed yet. */
#define K3_SLOT_EMPTY     (-1)
#define K3_SLOT_INFLIGHT  (-2)

/* Replacement policies. S3FIFO is the default; LRU is kept for A/B on one binary. */
enum { K3_POLICY_S3FIFO = 0, K3_POLICY_LRU = 1 };

/* Which S3-FIFO queue a slot is in. FREE means it holds nothing and is on the free
 * list, which is what makes a cold cache fill without any eviction work at all. */
enum { K3_Q_FREE = 0, K3_Q_SMALL = 1, K3_Q_MAIN = 2 };

typedef struct {
    K3ExpertSrc  src;             /* MUST be first: pass &cache->src to K3MoeW */

    const K3St  *st;
    int          n_layers, n_experts;

    unsigned char *arena;         /* nslot * slot_bytes, page aligned          */
    int64_t      slot_bytes;
    int          nslot;

    int32_t     *slot_of;         /* [n_layers*n_experts] -> slot, or -1       */
    int32_t     *inflight_of;     /* [n_layers*n_experts] -> slot being read into, or -1.
                                   * Separate from slot_of because an inflight expert is
                                   * NOT resident and must not be served, but a second
                                   * request for it must wait rather than read it twice. */
    int32_t     *key_of;          /* [nslot] -> layer*n_experts+expert, or -1  */
    uint64_t    *used_at;         /* [nslot] LRU stamp, used by K3_POLICY_LRU  */
    unsigned char *pinned;        /* [nslot] never evict while set             */
    K3ExpertRef *ref;             /* [nslot] geometry of the resident expert   */
    int32_t     *pad;             /* [nslot] where the payload starts in the slot;
                                   * non-zero only on the O_DIRECT path, where the
                                   * read is widened to aligned bounds             */

    /* ---- S3-FIFO ---- */
    int          policy;
    int32_t     *q_next;          /* [nslot] next slot in its queue, or -1     */
    unsigned char *q_of;          /* [nslot] K3_Q_FREE / SMALL / MAIN          */
    unsigned char *freq;          /* [nslot] 0..3, saturating                  */
    int32_t      free_head;
    int32_t      s_head, s_tail, s_len;
    int32_t      m_head, m_tail, m_len;
    int32_t      s_target;        /* small queue's share of the slots, ~10%    */
    int32_t     *ghost;           /* [nghost] ring of recently evicted keys    */
    unsigned char *ghost_mark;    /* [n_layers*n_experts] membership, exact    */
    int32_t      nghost, ghost_at, ghost_len;

    /* The slots most recently handed out, which eviction must not touch: k3_moe holds
     * the pointers from get() across three matmuls, and with a prefetch thread running
     * there is now something else that could take the slot in the meantime. */
    int32_t     *recent;          /* ring of slots handed out, -1 when unused  */
    int          recent_at, recent_n;
    /* Sized to top-k, and never larger than nslot-1. That is the SAME invariant the
     * constructor already enforces -- a cache smaller than one token's working set
     * cannot serve a token without evicting an expert still being multiplied -- just
     * made explicit instead of left to arithmetic. Larger than top-k and a small cache
     * protects every slot it has, which is not a slow cache but a broken one. */
    int          recent_cap;

    /* ---- speculative prefetch from the previous token's routing ---- */
    pthread_mutex_t mu;           /* guards EVERY field above; reads happen outside  */
    pthread_cond_t  cv;           /* a read landed, or the worker wants work         */
    pthread_t    spec_thread;
    int          spec_on;         /* 1 once the worker is running                    */
    int          spec_stop;
    int          spec_busy;       /* a request is queued or being served             */
    int          spec_layer;
    int          spec_ids[K3_MAX_TOPK];
    int          spec_n;
    int32_t     *last_topk;       /* [n_layers][K3_MAX_TOPK] previous token's set     */
    int32_t     *last_n;          /* [n_layers] how many of them are valid            */

    uint64_t     clock;
    /* stats */
    uint64_t     hits, misses, evictions, bytes_read;
    /* Speculation, kept separate from the batch prefetch: these bytes were read on a
     * guess, and how many of them were actually asked for afterwards is the only
     * number that says whether the guess is worth making. */
    uint64_t     spec_reads, spec_used;
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

#endif /* K3_CACHE_H */
