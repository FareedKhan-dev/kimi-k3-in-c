/* k3_trunk.h - stream the resident trunk, so RAM becomes a dial instead of a floor.
 *
 * WHY STREAM THE TRUNK AT ALL
 *   The engine holds 110 GB of trunk plus 4.70 GB of embed/lm_head. That is the floor
 *   that forces a 128 GB machine. Quantising it down is the obvious idea and it is the
 *   wrong one: Kimi K3's technical report section 4.1.4 says the experts are MXFP4 with
 *   quantisation-aware training "while all non-expert components (attention projections,
 *   latent MoE projections, shared experts, and MoE routers) remain in higher
 *   precision". That list IS this trunk. Measured on 31 real attention tensors
 *   (docs/data/trunk-quantisation.txt), post-hoc int4 costs 17.4% mean relative WEIGHT
 *   reconstruction error against 0.96% for int8 -- an ~18x gap, consistent across every
 *   tensor sampled. That is weight error, not output quality, but it is enough to rule
 *   out 4-bit on a trunk that was never trained for it.
 *
 *   Streaming costs zero error. The bytes are the checkpoint's own bytes.
 *
 * WHY IT IS AFFORDABLE
 *   Read bandwidth decides this, and it varies by an order of magnitude between a
 *   network volume and local NVMe. Measure the target device with tools/devbw.py
 *   before drawing conclusions. On local NVMe at a few GB/s the whole trunk costs
 *   tens of seconds per token against compute of the same order, so the read is not
 *   automatically the bottleneck. What makes it hideable is that, unlike expert routing,
 *   the trunk access order is FIXED: layer 0, 1, ... 92, every single token, so the next
 *   read is always known in advance.
 *
 *   It IS hidden now. k3_trunk_prefetch hands layer L+1 to a reader thread while the
 *   main thread computes on layer L, which is what the fixed walk order makes safe: the
 *   next layer is always known, so there is nothing to predict. Measured on the released
 *   checkpoint at the laptop preset, this took 71.75 s/token to 42.27 s/token, a 1.70x
 *   improvement, and it beat running the same model with four times the memory and no
 *   overlap.
 *
 *   The second ring slot it needs costs a full slot, 2.37 GB at the floor, so it is only
 *   taken when the trunk budget can pay for it. Below that the ring stays at one slot and
 *   reads are serial again; k3_trunk_open says so on stdout when that happens. Correctness
 *   does not depend on which path runs, and the emitted tokens are identical either way.
 *
 * WHY LRU WOULD BE THE WORST POSSIBLE POLICY HERE
 *   A cyclic sequential scan is the classic LRU pathology. With N < 93 slots, by the
 *   time the walk returns to layer 0 it is exactly the least recently used thing and has
 *   just been evicted, so the hit rate is ZERO no matter how much RAM is added. This
 *   cache therefore PINS a set of layers and streams the rest through a small ring:
 *   pin K layers and the hit rate is exactly K/93, deterministically, and every extra
 *   gigabyte buys its fair share. The expert cache does not use LRU either, but for the
 *   opposite reason: expert reuse is data-dependent, so it wants a frequency-aware
 *   policy rather than a deterministic one. See k3_cache.h.
 *
 * WHY THE PINNED SET IS CHOSEN LARGEST FIRST
 *   For the READ VOLUME it avoids, the choice is neutral: pinning any set totalling B
 *   bytes removes exactly B bytes of per-token traffic, whichever layers they are.
 *
 *   It is not neutral for the RING. The ring slot is uniform and must therefore be as
 *   large as the largest layer that still streams through it. Layers here range from
 *   1.15 GB to 2.34 GB against a 1.17 GB mean, and the 2.34 GB one is the dense layer 0.
 *   Pinning the fattest layers first drops the largest survivor, which shrinks the slot,
 *   which frees budget, which pins more -- so the two quantities are mutually dependent
 *   and are solved by iterating to a fixed point.
 *
 *   The effect is largest exactly where it matters most. At the laptop preset the trunk
 *   budget is 3.0 GB and a 2.34 GB slot is most of it; taking that layer out of the ring
 *   first is the difference between a ring that leaves nothing over and one that does.
 *   Set K3_PIN_PREFIX=1 to restore the old prefix order on the same binary, which is the
 *   only honest way to attribute a timing difference to this decision.
 *
 * LAYOUT
 *   tools/pack_trunk.py copies each layer's trunk, which is ONE contiguous run in its
 *   shard, into trunk.bin, and records offsets in trunk.json. So loading a layer is a
 *   single pread from local NVMe. The bytes are copied verbatim, so a tensor's position
 *   inside a slot is (its absolute shard offset - the run start).
 */
#ifndef K3_TRUNK_H
#define K3_TRUNK_H

#include "k3.h"
#include "k3_bind.h"

#define K3_TRUNK_ALIGN 4096   /* pack_trunk.py pads runs to this so O_DIRECT works */

typedef struct {
    char    *name;
    int64_t  off;          /* byte offset WITHIN the layer run */
    int64_t  nbytes;
    int      dtype;        /* K3Dtype */
} K3TrunkTensor;

typedef struct {
    int64_t  file_off;     /* offset in trunk.bin */
    int64_t  nbytes;
    K3TrunkTensor *t;
    int      nt;
} K3TrunkLayer;

typedef struct {
    int          fd;
    int          direct;        /* 1 when the file was opened O_DIRECT */
    int          n_layers;
    K3TrunkLayer *lay;

    /* Backs every K3TrunkTensor.name, so it must outlive the whole struct. Owned here
     * and freed by k3_trunk_close; do not free the parser arena separately. */
    char          *json_arena;

    /* Pinned layers get exact-size allocations; only the streaming ring is uniform.
     * Uniform slots everywhere would size EVERY slot for layer 0, whose dense MLP makes
     * it 2.34 GB against 1.27 GB for a normal layer, wasting about half the budget.
     *
     * WHICH layers are pinned is decided LARGEST FIRST, not as a prefix. See the note
     * on the selection loop in k3_trunk_open: the choice is byte-neutral for the read
     * volume it avoids and very much not neutral for the ring slot, which must be sized
     * to the largest layer still streaming. */
    unsigned char **pin;        /* [npin] one exact allocation per pinned layer */
    int32_t       *pin_of;      /* [n_layers] -> index into pin[], or -1        */
    unsigned char *arena;       /* [nslot] uniform ring slots                   */
    int64_t      slot_bytes;    /* raw run + the widen area                     */
    int64_t      widen_bytes;   /* of slot_bytes, the fp32 expansion area       */
    int          nslot;
    int          reader_started; /* 1 once the async reader thread is running; the
                                  * one-slot guard requires this to stay 0     */
    int          npin;          /* how many layers are pinned                   */
    int         *layer_of;      /* [nslot] which layer occupies each ring slot  */
    int32_t     *slot_of;       /* [n_layers], -1 when not resident             */
    int          ring;          /* next ring slot to reuse                      */
    int          walk;          /* layer the caller last fetched. Slots holding layers
                                 * just AHEAD of it are prefetched-but-unconsumed and
                                 * must not be evicted; see upcoming() in the .c    */

    /* One asynchronous reader keeps the spare ring slots busy. The worker never publishes
     * a layer name before its read succeeds; bind waits for completion before consuming
     * it, and never lets the slot it is using be evicted underneath it. */
    void         *io_state;
    /* The main thread's io_uring, when the platform has one. NULL means reads use pread,
     * which is always correct and simply shallower. See k3_uring.h. */
    void         *uring;

    /* stats */
    uint64_t     hits, misses;
    uint64_t     bytes_read;
    double       load_seconds;
} K3Trunk;

/* budget_bytes sizes the slot array. The largest layers are pinned, as many as the
 * budget allows after a small streaming ring. ring_want asks for that many ring slots
 * (0 selects the default of 2); it is a request, and the budget wins. Returns 0. */
int  k3_trunk_open(K3Trunk *tr, const char *dir, const K3Cfg *c, int64_t budget_bytes,
                   int ring_want);
void k3_trunk_close(K3Trunk *tr);

/* Make layer L resident and hand back the start of its run. The pointer stays valid
 * until the NEXT k3_trunk_fetch on this trunk and no longer: the ring reclaims slots,
 * and the only slot protected from the reader is the one most recently fetched.
 *
 * Split out of k3_trunk_bind so the read path can be exercised without a checkpoint --
 * tests/unit/test_trunk.c walks a synthetic trunk and checks that the bytes under this
 * pointer are the layer's own, which is the failure the ring exists to prevent and the
 * one that produces fluent wrong output rather than a crash. */
int  k3_trunk_fetch(K3Trunk *tr, int L, unsigned char **out);

/* Make layer L resident and point b's weight pointers at it. b must already have been
 * prepared by k3_bind_layer_mem, which resolves the layer's tensor shapes once. */
int  k3_trunk_bind(K3Trunk *tr, const K3Cfg *c, int L, K3LayerBind *b);

/* Hint that layers L, L+1, ... are coming, and queue asynchronous reads for as many of
 * them as the ring can hold at once. Pinned and resident layers are skipped rather than
 * ending the scan, so the reader looks past them. Safe to call at any time; with a
 * single-slot ring it is a no-op. */
void k3_trunk_prefetch(K3Trunk *tr, int L);

void k3_trunk_report(const K3Trunk *tr, const char *label);

#endif /* K3_TRUNK_H */
