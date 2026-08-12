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

/* ------------------------------------------------------------- S3-FIFO ------- *
 * Three queues over the same slots. Everything here runs under c->mu.
 *
 * key_of[] has THREE states, not two:
 *     >= 0            holds that key
 *     K3_SLOT_EMPTY   holds nothing, free to take
 *     K3_SLOT_INFLIGHT reserved by a read that has not finished
 *
 * The third state exists because of a real bug. A prefetch marked a slot empty before
 * reading into it, so that a failed read could not leave the slot claiming an expert it
 * did not hold. But the empty test was a FAST PATH that returned immediately, ahead of
 * the pinned check -- so the next expert in the same batch was handed the SAME slot,
 * several parallel reads wrote into one buffer, and the MoE multiplied garbage. It cost
 * one wrong token (65 instead of 2494) on the real model and nothing at all in the
 * fixtures. With a speculation thread there is now a second way to reach that state, so
 * the rule is enforced in one place for every policy.
 */
static void q_push_tail(K3Cache *c, int slot, int q)
{
    c->q_next[slot] = -1;
    c->q_of[slot] = (unsigned char)q;
    int32_t *head = (q == K3_Q_SMALL) ? &c->s_head : &c->m_head;
    int32_t *tail = (q == K3_Q_SMALL) ? &c->s_tail : &c->m_tail;
    int32_t *len  = (q == K3_Q_SMALL) ? &c->s_len  : &c->m_len;
    if (*tail < 0) { *head = *tail = (int32_t)slot; }
    else { c->q_next[*tail] = (int32_t)slot; *tail = (int32_t)slot; }
    (*len)++;
}

static int q_pop_head(K3Cache *c, int q)
{
    int32_t *head = (q == K3_Q_SMALL) ? &c->s_head : &c->m_head;
    int32_t *tail = (q == K3_Q_SMALL) ? &c->s_tail : &c->m_tail;
    int32_t *len  = (q == K3_Q_SMALL) ? &c->s_len  : &c->m_len;
    const int32_t s = *head;
    if (s < 0) return -1;
    *head = c->q_next[s];
    if (*head < 0) *tail = -1;
    c->q_next[s] = -1;
    c->q_of[s] = K3_Q_FREE;
    (*len)--;
    return (int)s;
}

static void ghost_add(K3Cache *c, int32_t key)
{
    if (key < 0 || c->nghost <= 0) return;
    if (c->ghost_mark[key]) return;
    if (c->ghost_len == c->nghost) {          /* FIFO: drop the oldest key */
        const int32_t old = c->ghost[c->ghost_at];
        if (old >= 0) c->ghost_mark[old] = 0;
    } else {
        c->ghost_len++;
    }
    c->ghost[c->ghost_at] = key;
    c->ghost_mark[key] = 1;
    c->ghost_at = (c->ghost_at + 1) % c->nghost;
}

/* A slot handed out to a caller in the current layer must not be evicted: k3_moe keeps
 * the K3ExpertQ pointers across three matmuls, and the speculation thread is running the
 * whole time. Only the last topk matter, so this is a fixed-size ring rather than a flag
 * that would have to be cleared by somebody. */
static int is_recent(const K3Cache *c, int slot)
{
    for (int i = 0; i < c->recent_n; i++) if (c->recent[i] == slot) return 1;
    return 0;
}

static void mark_recent(K3Cache *c, int slot)
{
    const int cap = c->recent_cap;
    if (cap <= 0) return;
    if (c->recent_n < cap) c->recent_n++;
    c->recent[c->recent_at] = (int32_t)slot;
    c->recent_at = (c->recent_at + 1) % cap;
}

static int evictable(const K3Cache *c, int slot)
{
    if (c->pinned[slot]) return 0;
    if (c->key_of[slot] == K3_SLOT_INFLIGHT) return 0;
    if (is_recent(c, slot)) return 0;
    return 1;
}

/* Least recently used evictable slot. Linear, deliberately: a few hundred comparisons
 * against a 17.55 MB read is not where the time goes. */
static int pick_victim_lru(K3Cache *c)
{
    int best = -1;
    uint64_t oldest = (uint64_t)-1;
    for (int i = 0; i < c->nslot; i++) {
        if (c->key_of[i] == K3_SLOT_INFLIGHT) continue;   /* being read into RIGHT NOW */
        if (c->key_of[i] == K3_SLOT_EMPTY) return i;      /* free, take it */
        if (!evictable(c, i)) continue;
        if (c->used_at[i] < oldest) { oldest = c->used_at[i]; best = i; }
    }
    return best;
}

/* S3-FIFO eviction. Evict from the small queue while it is over its share, otherwise
 * from the main one.
 *   small: freq > 0 promotes to main (the object proved itself); otherwise it goes, and
 *          its KEY is remembered in the ghost queue so a prompt return promotes straight
 *          into main next time.
 *   main:  freq > 0 buys one more lap, decrementing; otherwise it goes.
 * A slot that cannot be evicted right now (pinned, inflight, or just handed out) is put
 * back at the tail of its own queue and the search continues. The guard bounds that, so
 * a cache whose every slot is protected returns -1 instead of spinning. */
static int pick_victim_s3(K3Cache *c)
{
    if (c->free_head >= 0) {                  /* cold cache: no eviction needed */
        const int s = c->free_head;
        c->free_head = c->q_next[s];
        c->q_next[s] = -1;
        c->q_of[s] = K3_Q_FREE;
        return s;
    }
    /* bs / bm count slots bounced back into each queue because they could not be
     * evicted. Once a queue has bounced its whole length it has nothing to offer and the
     * search must move to the other one -- otherwise the small queue, whose head is the
     * expert the caller is using RIGHT NOW, is popped and pushed forever while the main
     * queue sits full of perfectly evictable slots. That is not a slow path, it is a
     * failed admission: pick_victim returns -1 and the MoE drops an expert. */
    int bs = 0, bm = 0;
    const int guard = 4 * c->nslot + 8;
    for (int step = 0; step < guard; step++) {
        int from;
        if (c->s_len > bs && c->s_len >= c->s_target) from = K3_Q_SMALL;
        else if (c->m_len > bm)                       from = K3_Q_MAIN;
        else if (c->s_len > bs)                       from = K3_Q_SMALL;
        else return -1;                       /* every slot is pinned, inflight or in use */

        const int v = q_pop_head(c, from);
        if (v < 0) { if (from == K3_Q_SMALL) bs = c->s_len; else bm = c->m_len; continue; }
        if (!evictable(c, v)) {
            q_push_tail(c, v, from);
            if (from == K3_Q_SMALL) bs++; else bm++;
            continue;
        }
        if (from == K3_Q_SMALL) {
            if (c->freq[v] > 0) { c->freq[v] = 0; q_push_tail(c, v, K3_Q_MAIN); continue; }
            ghost_add(c, c->key_of[v]);
            return v;
        }
        if (c->freq[v] > 0) { c->freq[v]--; q_push_tail(c, v, K3_Q_MAIN); continue; }
        return v;
    }
    return -1;
}

static int pick_victim(K3Cache *c)
{
    return c->policy == K3_POLICY_LRU ? pick_victim_lru(c) : pick_victim_s3(c);
}

/* Put a freshly loaded slot into the right queue. A key the ghost queue remembers went
 * straight through the small queue once already, so it starts in main. */
static void admit_queue(K3Cache *c, int slot, int32_t key)
{
    if (c->policy == K3_POLICY_LRU) return;
    c->freq[slot] = 0;
    if (key >= 0 && c->ghost_mark[key]) {
        c->ghost_mark[key] = 0;               /* the ring entry ages out on its own */
        q_push_tail(c, slot, K3_Q_MAIN);
    } else {
        q_push_tail(c, slot, K3_Q_SMALL);
    }
}

/* A hit. S3-FIFO counts touches with a saturating 2-bit counter rather than reordering
 * a list, which is the whole reason it is cheaper than LRU as well as better here. */
static void touch(K3Cache *c, int slot)
{
    c->used_at[slot] = ++c->clock;
    if (c->policy != K3_POLICY_LRU && c->freq[slot] < 3) c->freq[slot]++;
}

/* Take a slot out of whatever queue it is in, for reuse. The victim pickers already
 * popped it; this is for the LRU path, which has no queues, and for slots reclaimed
 * from the free list. */
static void detach(K3Cache *c, int slot)
{
    if (c->policy == K3_POLICY_LRU) return;
    if (c->q_of[slot] == K3_Q_FREE) return;
    /* Only reachable via pick_victim_lru, which does not maintain the queues. */
    const int q = c->q_of[slot];
    int32_t *head = (q == K3_Q_SMALL) ? &c->s_head : &c->m_head;
    int32_t *tail = (q == K3_Q_SMALL) ? &c->s_tail : &c->m_tail;
    int32_t *len  = (q == K3_Q_SMALL) ? &c->s_len  : &c->m_len;
    int32_t prev = -1;
    for (int32_t s = *head; s >= 0; prev = s, s = c->q_next[s]) {
        if (s != slot) continue;
        if (prev < 0) *head = c->q_next[s]; else c->q_next[prev] = c->q_next[s];
        if (*tail == s) *tail = prev;
        (*len)--;
        break;
    }
    c->q_next[slot] = -1;
    c->q_of[slot] = K3_Q_FREE;
}

/* Reserve a slot for `key` and mark it INFLIGHT. Caller holds c->mu.
 * Returns the slot, or -1 when nothing can be evicted right now. */
static int reserve(K3Cache *c, int32_t key, const K3ExpertRef *r)
{
    const int slot = pick_victim(c);
    if (slot < 0) return -1;
    detach(c, slot);
    if (c->key_of[slot] >= 0) { c->slot_of[c->key_of[slot]] = -1; c->evictions++; }
    /* INFLIGHT, not EMPTY. Marking it empty made the victim search's free-slot fast path
     * hand the same slot to the next expert in the very same batch. */
    c->key_of[slot] = K3_SLOT_INFLIGHT;
    c->inflight_of[key] = (int32_t)slot;
    c->ref[slot] = *r;
    c->used_at[slot] = ++c->clock;
    return slot;
}

/* Publish, or release, a reserved slot. Caller holds c->mu. */
static void publish(K3Cache *c, int32_t key, int slot, int64_t got, int64_t pad,
                    const K3ExpertRef *r, const char *what, int layer, int expert)
{
    c->inflight_of[key] = -1;
    if (got != r->nbytes) {
        fprintf(stderr, "k3_cache: short %s of L%d expert %d (%lld of %lld); leaving the "
                        "slot empty so it cannot be served as a hit\n",
                what, layer, expert, (long long)got, (long long)r->nbytes);
        c->key_of[slot] = K3_SLOT_EMPTY;
        if (c->policy != K3_POLICY_LRU) {     /* back on the free list, not in a queue */
            c->q_next[slot] = c->free_head;
            c->free_head = (int32_t)slot;
            c->q_of[slot] = K3_Q_FREE;
        }
        pthread_cond_broadcast(&c->cv);
        return;
    }
    c->pad[slot] = (int32_t)pad;
    c->key_of[slot] = key;
    c->slot_of[key] = (int32_t)slot;
    c->used_at[slot] = ++c->clock;
    c->bytes_read += (uint64_t)got;
    admit_queue(c, slot, key);
    pthread_cond_broadcast(&c->cv);
}

/* Bring (layer, expert) resident and return its slot, or -1. Caller must NOT hold the
 * mutex; this takes it, and drops it around the read so other threads can work. */
static int admit(K3Cache *c, int layer, int expert, int count_stats)
{
    const int32_t key = layer * c->n_experts + expert;
    K3ExpertRef r;

    pthread_mutex_lock(&c->mu);
    for (;;) {
        const int32_t slot = c->slot_of[key];
        if (slot >= 0) {
            if (count_stats) c->hits++;
            touch(c, slot);
            mark_recent(c, slot);
            pthread_mutex_unlock(&c->mu);
            return slot;
        }
        /* Someone else -- the speculation thread, or a batch prefetch -- is already
         * reading exactly this expert. Wait for it rather than reading it twice. */
        if (c->inflight_of[key] >= 0) {
            pthread_cond_wait(&c->cv, &c->mu);
            continue;
        }
        break;
    }
    if (count_stats) c->misses++;
    pthread_mutex_unlock(&c->mu);

    if (k3_expert_ref(c->st, layer, expert, &r) != 0) return -1;
    if (r.nbytes > c->slot_bytes) {
        fprintf(stderr, "k3_cache: L%d expert %d is %lld bytes, slot holds %lld\n",
                layer, expert, (long long)r.nbytes, (long long)c->slot_bytes);
        return -1;
    }

    pthread_mutex_lock(&c->mu);
    /* Recheck: the wait above dropped the lock, and so did the ref lookup. */
    if (c->slot_of[key] >= 0) {
        const int slot = c->slot_of[key];
        touch(c, slot);
        mark_recent(c, slot);
        pthread_mutex_unlock(&c->mu);
        return slot;
    }
    int slot = -1;
    for (;;) {
        if (c->inflight_of[key] >= 0 || c->slot_of[key] >= 0) {
            pthread_cond_wait(&c->cv, &c->mu);
            if (c->slot_of[key] >= 0) {
                slot = c->slot_of[key];
                touch(c, slot);
                mark_recent(c, slot);
                pthread_mutex_unlock(&c->mu);
                return slot;
            }
            continue;
        }
        slot = reserve(c, key, &r);
        if (slot >= 0) break;
        /* Nothing evictable. If a read is in flight it will free something; otherwise
         * every slot really is pinned and waiting would hang. */
        int any = 0;
        for (int i = 0; i < c->nslot && !any; i++) if (c->key_of[i] == K3_SLOT_INFLIGHT) any = 1;
        if (!any) {
            pthread_mutex_unlock(&c->mu);
            fprintf(stderr, "k3_cache: every slot is pinned or in use, cannot admit "
                            "L%d expert %d\n", layer, expert);
            return -1;
        }
        pthread_cond_wait(&c->cv, &c->mu);
    }
    pthread_mutex_unlock(&c->mu);

    const double t0 = now_s();
    int64_t pad = 0;
    const int64_t got = k3_expert_load_direct(c->st, &r,
                            c->arena + (size_t)slot * c->slot_bytes,
                            c->slot_bytes, &pad);
    const double dt = now_s() - t0;

    pthread_mutex_lock(&c->mu);
    c->load_seconds += dt;
    publish(c, key, slot, got, pad, &r, "load", layer, expert);
    const int ok = (got == r.nbytes);
    if (ok) mark_recent(c, slot);
    pthread_mutex_unlock(&c->mu);
    return ok ? slot : -1;
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
typedef struct { int slot; int expert; K3ExpertRef r; int64_t got, pad; } Work;

/* The three phases, shared by the caller-driven batch prefetch and the speculation
 * thread. `spec` only changes which counter the bytes land in. */
static int batch_load(K3Cache *c, int layer, const int *ids, int n, int spec)
{
    /* One entry per expert in a batch prefetch, so it is bounded by top-k. */
    Work w[K3_MAX_TOPK];
    int nw = 0;
    const int cap = (int)(sizeof w / sizeof *w);

    /* ---- phase 1: reserve, under the lock ---- */
    pthread_mutex_lock(&c->mu);
    for (int i = 0; i < n && nw < cap; i++) {
        const int e = ids[i];
        if (e < 0 || e >= c->n_experts) continue;
        const int32_t key = layer * c->n_experts + e;
        if (c->slot_of[key] >= 0) continue;             /* already resident */
        if (c->inflight_of[key] >= 0) continue;         /* somebody is reading it */

        int dup = 0;                                    /* the same id twice in one top-k */
        for (int j = 0; j < nw; j++) if (w[j].expert == e) { dup = 1; break; }
        if (dup) continue;

        K3ExpertRef r;
        if (k3_expert_ref(c->st, layer, e, &r) != 0) continue;
        if (r.nbytes > c->slot_bytes) continue;

        const int slot = reserve(c, key, &r);
        if (slot < 0) break;
        w[nw].slot = slot; w[nw].expert = e; w[nw].r = r; w[nw].got = -1; w[nw].pad = 0;
        nw++;
    }
    pthread_mutex_unlock(&c->mu);
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

    /* ---- phase 2: read, concurrently and WITHOUT the lock ---- */
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
    const double dt = now_s() - t0;

    /* ---- phase 3: publish only what actually arrived ---- */
    int ok = 0;
    pthread_mutex_lock(&c->mu);
    c->load_seconds += dt;
    for (int i = 0; i < nw; i++) {
        const int32_t key = layer * c->n_experts + w[i].expert;
        publish(c, key, w[i].slot, w[i].got, w[i].pad, &w[i].r,
                spec ? "speculation" : "prefetch", layer, w[i].expert);
        if (w[i].got != w[i].r.nbytes) continue;
        if (spec) c->spec_reads++; else c->prefetch_reads++;
        ok++;
    }
    pthread_mutex_unlock(&c->mu);
    return ok;
}

/* Remember this layer's routing so the NEXT token can be guessed from it, and hand the
 * batch over. k3_moe calls this with the whole top-k before consuming any of it. */
static int cache_getmany(K3ExpertSrc *self, int layer, const int *ids, int n)
{
    K3Cache *c = (K3Cache *)self;
    if (n <= 0) return 0;

    if (c->last_topk && layer >= 0 && layer < c->n_layers) {
        pthread_mutex_lock(&c->mu);
        int keep = n < K3_MAX_TOPK ? n : K3_MAX_TOPK;
        /* Count how many of the guessed set the router actually asked for, which is the
         * only honest measure of whether guessing is worth the bandwidth. */
        for (int i = 0; i < keep && c->last_n[layer] > 0; i++)
            for (int j = 0; j < c->last_n[layer]; j++)
                if (c->last_topk[(size_t)layer * K3_MAX_TOPK + j] == ids[i]) {
                    c->spec_used++; break;
                }
        for (int i = 0; i < keep; i++)
            c->last_topk[(size_t)layer * K3_MAX_TOPK + i] = (int32_t)ids[i];
        c->last_n[layer] = keep;
        pthread_mutex_unlock(&c->mu);
    }
    return batch_load(c, layer, ids, n, 0);
}

/* ------------------------------------------------- speculative prefetch ------ */
static void *spec_main(void *arg)
{
    K3Cache *c = (K3Cache *)arg;
    for (;;) {
        pthread_mutex_lock(&c->mu);
        while (!c->spec_busy && !c->spec_stop)
            pthread_cond_wait(&c->cv, &c->mu);
        if (c->spec_stop) { pthread_mutex_unlock(&c->mu); return NULL; }
        const int layer = c->spec_layer;
        int ids[K3_MAX_TOPK];
        const int n = c->spec_n;
        memcpy(ids, c->spec_ids, (size_t)n * sizeof(int));
        pthread_mutex_unlock(&c->mu);

        batch_load(c, layer, ids, n, 1);

        pthread_mutex_lock(&c->mu);
        c->spec_busy = 0;
        pthread_cond_broadcast(&c->cv);
        pthread_mutex_unlock(&c->mu);
    }
}

/* Called when the MoE enters a layer, before the router has run. Issues reads for the
 * set this layer wanted for the previous token. Returns without doing anything when
 * there is no history yet or the worker is still busy: a queue of stale guesses is worth
 * less than the bandwidth it would spend. */
static void cache_speculate(K3ExpertSrc *self, int layer)
{
    K3Cache *c = (K3Cache *)self;
    if (!c->spec_on || layer < 0 || layer >= c->n_layers) return;
    pthread_mutex_lock(&c->mu);
    /* A new layer means the previous layer's experts are finished with, so the
     * protection on the slots handed out for them can be dropped. */
    c->recent_n = 0; c->recent_at = 0;
    if (c->spec_busy || c->last_n[layer] <= 0) { pthread_mutex_unlock(&c->mu); return; }
    c->spec_layer = layer;
    c->spec_n = c->last_n[layer];
    for (int i = 0; i < c->spec_n; i++)
        c->spec_ids[i] = (int)c->last_topk[(size_t)layer * K3_MAX_TOPK + i];
    c->spec_busy = 1;
    pthread_cond_broadcast(&c->cv);
    pthread_mutex_unlock(&c->mu);
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
    pthread_mutex_lock(&c->mu);
    const int slot = c->slot_of[key];
    if (slot >= 0) {
        /* The caller is about to multiply out of this slot, so it must be protected
         * from eviction for the same reason get()'s result is. */
        mark_recent(c, slot);
        touch(c, slot);
    }
    pthread_mutex_unlock(&c->mu);
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
    pthread_mutex_lock(&c->mu);
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
    pthread_mutex_unlock(&c->mu);

    const int slot = admit(c, layer, expert, 1);
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
    /* Policy is a runtime choice so the two can be compared on ONE binary. Comparing
     * two builds compares two binaries; comparing one decision is the only way to
     * attribute a hit-rate difference to the policy. */
    {
        const char *pol = getenv("K3_CACHE_POLICY");
        c->policy = (pol && !strcmp(pol, "lru")) ? K3_POLICY_LRU : K3_POLICY_S3FIFO;
        if (c->policy == K3_POLICY_LRU)
            printf("expert cache: policy LRU (K3_CACHE_POLICY), not the default S3-FIFO\n");
    }
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
    c->inflight_of = (int32_t *)malloc(nkey * sizeof(int32_t));
    c->key_of  = (int32_t *)malloc((size_t)c->nslot * sizeof(int32_t));
    c->used_at = (uint64_t *)calloc((size_t)c->nslot, sizeof(uint64_t));
    c->pinned  = (unsigned char *)calloc((size_t)c->nslot, 1);
    c->ref     = (K3ExpertRef *)calloc((size_t)c->nslot, sizeof(K3ExpertRef));
    c->pad     = (int32_t *)calloc((size_t)c->nslot, sizeof(int32_t));
    c->hist    = (uint32_t *)calloc(nkey, sizeof(uint32_t));
    c->q_next  = (int32_t *)malloc((size_t)c->nslot * sizeof(int32_t));
    c->q_of    = (unsigned char *)calloc((size_t)c->nslot, 1);
    c->freq    = (unsigned char *)calloc((size_t)c->nslot, 1);
    c->recent_cap = cfg->topk < K3_MAX_TOPK ? cfg->topk : K3_MAX_TOPK;
    if (c->recent_cap > c->nslot - 1) c->recent_cap = c->nslot - 1;
    if (c->recent_cap < 1) c->recent_cap = 1;
    c->recent  = (int32_t *)malloc((size_t)c->recent_cap * sizeof(int32_t));
    c->ghost_mark = (unsigned char *)calloc(nkey, 1);
    c->last_topk  = (int32_t *)calloc((size_t)c->n_layers * K3_MAX_TOPK, sizeof(int32_t));
    c->last_n     = (int32_t *)calloc((size_t)c->n_layers, sizeof(int32_t));
    /* The ghost queue remembers as many keys as there are slots. Bigger than that and it
     * promotes objects whose eviction is no longer recent enough to mean anything. */
    c->nghost = c->nslot;
    c->ghost  = (int32_t *)malloc((size_t)c->nghost * sizeof(int32_t));
    if (!c->slot_of || !c->inflight_of || !c->key_of || !c->used_at || !c->pinned ||
        !c->ref || !c->pad || !c->hist || !c->q_next || !c->q_of || !c->freq ||
        !c->recent || !c->ghost_mark || !c->ghost || !c->last_topk || !c->last_n) {
        k3_cache_free(c); return -1;
    }
    for (size_t i = 0; i < nkey; i++) { c->slot_of[i] = -1; c->inflight_of[i] = -1; }
    for (int i = 0; i < c->nslot; i++) c->key_of[i] = -1;
    for (int i = 0; i < c->recent_cap; i++) c->recent[i] = -1;
    for (int i = 0; i < c->nghost; i++) c->ghost[i] = -1;

    /* Every slot starts on the free list, so a cold cache fills without ever running the
     * eviction machinery. The queues are empty until something is evicted. */
    c->s_head = c->s_tail = c->m_head = c->m_tail = -1;
    c->free_head = -1;
    for (int i = c->nslot - 1; i >= 0; i--) { c->q_next[i] = c->free_head; c->free_head = i; }
    /* 10% small queue, the ratio the S3-FIFO paper reports as insensitive across
     * workloads, with a floor so a tiny cache still has a small queue at all. */
    c->s_target = c->nslot / 10;
    if (c->s_target < 1) c->s_target = 1;

    pthread_mutex_init(&c->mu, NULL);
    pthread_cond_init(&c->cv, NULL);

    /* Speculation needs room for the guessed set AND the set the router actually picks,
     * on top of what the current layer is using. Below that it would evict what this
     * token still needs, turning a guess into a self-inflicted miss. */
    const int spec_min = 2 * cfg->topk + 8;
    if (getenv("K3_NOSPEC")) {
        printf("expert cache: speculative prefetch DISABLED by K3_NOSPEC\n");
    } else if (c->nslot < spec_min) {
        printf("expert cache: speculative prefetch OFF, %d slots is below the %d needed "
               "to hold\n              two tokens' working set at top-%d\n",
               c->nslot, spec_min, cfg->topk);
    } else if (pthread_create(&c->spec_thread, NULL, spec_main, c) == 0) {
        c->spec_on = 1;
        c->src.speculate = cache_speculate;
    } else {
        fprintf(stderr, "k3_cache: cannot start the speculation thread; continuing "
                        "without it\n");
    }
    return 0;
}

void k3_cache_free(K3Cache *c)
{
    if (c->spec_on) {
        pthread_mutex_lock(&c->mu);
        c->spec_stop = 1;
        pthread_cond_broadcast(&c->cv);
        pthread_mutex_unlock(&c->mu);
        pthread_join(c->spec_thread, NULL);
        c->spec_on = 0;
    }
    pthread_cond_destroy(&c->cv);
    pthread_mutex_destroy(&c->mu);
    k3_aligned_free(c->arena); free(c->slot_of); free(c->inflight_of); free(c->key_of);
    free(c->used_at); free(c->pinned); free(c->ref); free(c->pad); free(c->hist);
    free(c->q_next); free(c->q_of); free(c->freq); free(c->recent);
    free(c->ghost); free(c->ghost_mark); free(c->last_topk); free(c->last_n);
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
    pthread_mutex_lock(&c->mu);
    const int slot = c->slot_of[key];
    if (slot >= 0) c->pinned[slot] = pin ? 1 : 0;
    pthread_mutex_unlock(&c->mu);
    return slot >= 0;
}

int k3_cache_prefetch(K3Cache *c, int layer, int expert)
{
    /* Warming the cache is not a model request, so it must not move the hit rate. */
    return admit(c, layer, expert, 0) >= 0 ? 0 : -1;
}

void k3_cache_reset_stats(K3Cache *c)
{
    pthread_mutex_lock(&c->mu);
    c->hits = c->misses = c->evictions = c->bytes_read = 0;
    c->load_seconds = 0.0;
    c->spec_reads = c->spec_used = 0;
    /* prefetch_reads belongs to the same window as hits and misses.
     *
     * k3_cache_report derives the effective hit rate as (hits - prefetch_reads), so both
     * counters must cover the same interval. Resetting one without the other compares a
     * per-window numerator against a since-startup subtrahend, which drives the result
     * negative and clamps it to zero at every cache size. */
    c->prefetch_reads = 0;
    pthread_mutex_unlock(&c->mu);
}

void k3_cache_report(const K3Cache *c, const char *label)
{
    const uint64_t n = c->hits + c->misses;
    int resident = 0, pinned = 0;
    for (int i = 0; i < c->nslot; i++) { if (c->key_of[i] >= 0) resident++; if (c->pinned[i]) pinned++; }
    printf("cache [%s]\n", label ? label : "");
    printf("  policy       : %s%s\n",
           c->policy == K3_POLICY_LRU ? "LRU" : "S3-FIFO",
           c->policy == K3_POLICY_LRU ? "" : " (small/main/ghost)");
    if (c->policy != K3_POLICY_LRU)
        printf("                 small %d of %d target, main %d, ghost %d keys\n",
               c->s_len, c->s_target, c->m_len, c->ghost_len);
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
    if (c->spec_on) {
        /* A guess is worth making only if the bandwidth it spends comes back as hits.
         * spec_reads is what speculation actually pulled off the disk; spec_used counts
         * how many of the guessed ids the router then asked for. Printing the second
         * without the first would make a wasteful prefetcher look free. */
        printf("  speculation  : %llu experts read on the previous token's routing, "
               "%llu of the guessed\n                 set were then requested (%.1f%% of "
               "%llu requests)\n",
               (unsigned long long)c->spec_reads, (unsigned long long)c->spec_used,
               n ? 100.0 * (double)c->spec_used / (double)n : 0.0,
               (unsigned long long)n);
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
