/* k3_trunk.c - see k3_trunk.h for why the trunk is streamed rather than quantised. */
#define _GNU_SOURCE            /* O_DIRECT */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "k3_portable_io.h"   /* first: sets _DARWIN_C_SOURCE before any libc header */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>   /* MADV_HUGEPAGE; k3_portable_io.h no-ops both on Windows */
#endif
#include <pthread.h>

#include "json.h"
#include "k3_st.h"
#include "k3_trunk.h"
#include "k3_uring.h"

static int k3_alloc_direct(void **out, size_t bytes);   /* defined below */

/* THE READER, and the two invariants that keep a deeper ring from corrupting output.
 *
 * With one slot in flight the old design could stay informal: the worker held one
 * request, the main thread waited for exactly that request, and the round-robin could
 * not collide with the slot the caller was computing on because there were only ever two.
 * A ring of three or more, kept full, breaks all of these by construction, so the rules
 * are now explicit and enforced in one place, claim_slot_locked:
 *
 *   1. A slot being READ INTO is never handed out again. pending[s] names the layer
 *      whose read is in progress, and a claim skips any such slot. This is the same
 *      hazard K3_SLOT_INFLIGHT solves in the expert cache, and it cost a wrong token
 *      there before it was understood.
 *   2. The slot the CALLER IS CURRENTLY USING is never evicted. `held` names it. Without
 *      this the prefetcher wraps the ring and preads the next layer straight over the
 *      weights the main thread is multiplying: the read succeeds, no pointer changes,
 *      and the run finishes with fluent wrong tokens. That exact failure was measured on
 *      the released checkpoint when a single-slot ring was given a reader thread.
 *   3. A slot holding a layer the walk is ABOUT TO REACH is never evicted either. See
 *      upcoming(). This one costs bytes rather than correctness, which is why it went
 *      unnoticed: the layer is simply read a second time.
 *
 * Everything that touches layer_of, slot_of, ring, pending or the request queue holds
 * io->mu. The READS themselves happen outside it, which is the entire point: the worker
 * and the main thread can both have a transfer in flight.
 */
#define K3_TRUNK_MAXRING 8

typedef struct {
    pthread_t thread;
    int       running;
    pthread_mutex_t mu;
    pthread_cond_t  cv_work;    /* worker waits here for a request        */
    pthread_cond_t  cv_done;    /* main thread waits here for a read      */
    K3Trunk  *tr;
    int       stop;
    int       q[K3_TRUNK_MAXRING];      /* queued layers, FIFO            */
    int       qslot[K3_TRUNK_MAXRING];
    int       nq;
    int       pending[K3_TRUNK_MAXRING];/* [slot] layer being read, or -1 */
    int       held;                     /* slot the caller is using, or -1 */
    K3Uring  *ur;                       /* the worker's own ring          */
} K3TrunkIO;

static void *trunk_io_main(void *arg);

/* WHERE THE TIME IN A BIND ACTUALLY GOES.
 *
 * k3_trunk_report divides bytes_read by load_seconds, but load_seconds brackets ONLY the
 * pread loop. It therefore reports a DEVICE rate, and everything else the bind does --
 * widening bf16 tensors to fp32, resolving names, kernel page bookkeeping -- is invisible
 * to it while still being paid on every layer of every token. That residual is large
 * enough to change conclusions drawn from the device rate alone.
 *
 * These three counters close the gap by measurement rather than estimate: wall clock
 * around the whole of k3_trunk_bind, of which the widen loop is tracked separately, so
 * bind_wall - load_seconds - widen_wall is the remaining unattributed time. */
double k3_trunk_bind_wall = 0.0;    /* total wall inside k3_trunk_bind   */
double k3_trunk_widen_wall = 0.0;   /* of which, inside k3_bind_layer_mem */
long   k3_trunk_binds = 0;

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int dt_of(const char *s)
{
    if (!strcmp(s, "BF16")) return K3_DT_BF16;
    if (!strcmp(s, "F32"))  return K3_DT_F32;
    if (!strcmp(s, "U8"))   return K3_DT_U8;
    if (!strcmp(s, "F16"))  return K3_DT_F16;
    if (!strcmp(s, "I8R"))  return K3_DT_I8R;
    return K3_DT_UNKNOWN;
}

static char *slurp(const char *p, size_t *n)
{
    FILE *f = fopen(p, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)sz + 1);
    if (!b) { fclose(f); return NULL; }
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) { free(b); fclose(f); return NULL; }
    b[sz] = 0; fclose(f);
    if (n) *n = (size_t)sz;
    return b;
}

/* Resolver handed to k3_bind_layer_mem: linear over one layer's ~28 tensors, which is
 * nothing next to a 1.27 GB read. */
typedef struct { const K3TrunkLayer *L; } Finder;

static int find_in_layer(void *ctx, const char *name,
                         int64_t *off, int64_t *nbytes, int *dtype)
{
    const K3TrunkLayer *L = ((Finder *)ctx)->L;
    for (int i = 0; i < L->nt; i++)
        if (!strcmp(L->t[i].name, name)) {
            *off = L->t[i].off; *nbytes = L->t[i].nbytes; *dtype = L->t[i].dtype;
            return 0;
        }
    return -1;
}

int k3_trunk_open(K3Trunk *tr, const char *dir, const K3Cfg *c, int64_t budget_bytes,
                  int ring_want)
{
    memset(tr, 0, sizeof *tr);
    /* memset leaves fd == 0, which is stdin. Every failure path below returns without
     * opening the file, and a caller that then calls k3_trunk_close would close the
     * process's stdin. -1 is the only safe "no file" value. */
    tr->fd = -1;

    char p[1024];
    snprintf(p, sizeof p, "%s/trunk.json", dir);
    size_t jn = 0;
    char *txt = slurp(p, &jn);
    if (!txt) { fprintf(stderr, "k3_trunk: cannot read %s\n", p); return -1; }
    /* The parser arena backs every K3TrunkTensor.name, so it must outlive the whole
     * K3Trunk. It is owned by the struct and freed in k3_trunk_close. */
    char *arena = NULL;
    jval *root = json_parse(txt, &arena);
    tr->json_arena = arena;
    if (!root) { fprintf(stderr, "k3_trunk: %s is not valid JSON\n", p); free(txt); return -1; }

    jval *jl = json_get(root, "layers");
    if (!jl || jl->t != J_ARR) { fprintf(stderr, "k3_trunk: no layers array\n"); goto bad; }
    tr->n_layers = jl->len;
    tr->lay = (K3TrunkLayer *)calloc((size_t)tr->n_layers, sizeof(K3TrunkLayer));
    if (!tr->lay) goto bad;

    for (int i = 0; i < jl->len; i++) {
        jval *e = jl->kids[i];
        jval *v;
        K3TrunkLayer *L = &tr->lay[i];
        if ((v = json_get(e, "file_off")) && v->t == J_NUM) L->file_off = (int64_t)v->num;
        if ((v = json_get(e, "nbytes"))   && v->t == J_NUM) L->nbytes   = (int64_t)v->num;
        jval *ts = json_get(e, "tensors");
        if (!ts || ts->t != J_OBJ) { fprintf(stderr, "k3_trunk: layer %d has no tensors\n", i); goto bad; }
        L->nt = ts->len;
        L->t = (K3TrunkTensor *)calloc((size_t)L->nt, sizeof(K3TrunkTensor));
        if (!L->t) goto bad;
        for (int k = 0; k < ts->len; k++) {
            K3TrunkTensor *t = &L->t[k];
            /* keys live in the parser arena, which is kept for the process lifetime */
            t->name = ts->keys[k];
            jval *o = ts->kids[k];
            if ((v = json_get(o, "off"))    && v->t == J_NUM) t->off    = (int64_t)v->num;
            if ((v = json_get(o, "nbytes")) && v->t == J_NUM) t->nbytes = (int64_t)v->num;
            if ((v = json_get(o, "dtype"))  && v->t == J_STR) t->dtype  = dt_of(v->str);
        }
    }
    free(txt);                      /* arena holds the strings; txt itself is done */

    snprintf(p, sizeof p, "%s/trunk.bin", dir);
    /* O_DIRECT, because the trunk is the one thing the page cache CANNOT help with.
     * Each streamed layer is read once per token and never reused before eviction, so
     * buffering it only copies every byte twice and evicts whatever else the cgroup was
     * holding. Measured under a 32 GB cap: buffered reads collapsed to 1,878 MB/s
     * against 6,553 MB/s unconstrained.
     *
     * It requires offset, length and buffer all aligned. pack_trunk.py pads every run
     * to 4096 and the slots come from posix_memalign. If the filesystem refuses
     * O_DIRECT, fall back rather than fail: correctness does not depend on it. */
    tr->direct = 1;
    tr->fd = open(p, O_RDONLY | O_DIRECT);
    if (tr->fd >= 0 && k3_set_direct(tr->fd) != 0)
        tr->direct = 0;   /* Darwin refused F_NOCACHE: reads stay buffered but correct */
    if (tr->fd < 0) {
        tr->direct = 0;
        tr->fd = open(p, O_RDONLY);
    }
    if (tr->fd < 0) { fprintf(stderr, "k3_trunk: cannot open %s\n", p); return -1; }
    {
        jval *a = json_get(root, "align");
        const int64_t want = (a && a->t == J_NUM) ? (int64_t)a->num : 0;
        if (tr->direct && want != K3_TRUNK_ALIGN) {
            /* A trunk packed before the alignment change cannot be read with O_DIRECT:
             * its run offsets are arbitrary. Say so rather than fail every read. */
            fprintf(stderr, "k3_trunk: trunk.json reports align %lld, expected %d; "
                            "falling back to buffered reads (repack to enable O_DIRECT)\n",
                    (long long)want, K3_TRUNK_ALIGN);
            close(tr->fd);
            tr->direct = 0;
            tr->fd = open(p, O_RDONLY);
            if (tr->fd < 0) return -1;
        }
    }

    const size_t widen = k3_bind_widen_bytes(c);
    int64_t total = 0;
    for (int i = 0; i < tr->n_layers; i++) total += tr->lay[i].nbytes;

    /* Pin a PREFIX of layers, each in an exact-size allocation, then keep a small ring
     * of uniform slots for everything else. Uniform slots everywhere would size every
     * slot for layer 0 (2.34 GB, the dense MLP) and waste roughly half the budget. */
    tr->slot_of = (int32_t *)malloc((size_t)tr->n_layers * sizeof(int32_t));
    if (!tr->slot_of) return -1;
    for (int i = 0; i < tr->n_layers; i++) tr->slot_of[i] = -1;

    /* Ring slots: the layer being computed on, plus the reads in flight behind it.
     *
     * This is a REQUEST, not a guarantee. Each slot costs a full slot's worth of memory,
     * which at the floor is 2.37 GB, and the budget the caller asked for has to come
     * first: measured on the released checkpoint, taking the second slot unconditionally
     * moved the laptop preset from 8.78 GB to 11.12 GB peak RSS, a 27% overshoot of a
     * 3.0 GB trunk budget, and left the printed memory plan understating the real figure.
     * So slots are granted only when they fit, and reported when they are not. A single
     * slot is exactly what this file did before the asynchronous reader existed, so
     * falling back is always safe; it costs speed, not correctness.
     *
     * WHY MORE THAN TWO IS WORTH OFFERING. Two slots overlap one read with one layer's
     * compute, which is enough only while the two take about the same time. They do not:
     * a trunk read is 62.40 s of a 135.8 s token on the reference machine, so the reader
     * spends part of every layer idle waiting to be allowed to start the next one. A
     * third slot lets it run a layer further ahead and keeps the device continuously
     * busy. It is not free -- another 2.37 GB -- which is why it is a dial rather than a
     * default, and why the default stays at 2. */
    int RING_WANT = ring_want > 0 ? ring_want : 2;
    if (RING_WANT > K3_TRUNK_MAXRING) RING_WANT = K3_TRUNK_MAXRING;
    if (RING_WANT < 1) RING_WANT = 1;
    int RING = RING_WANT;

    /* Size the ring from the layers that will actually STREAM through it, and choose
     * which layers to pin LARGEST FIRST.
     *
     * Pinning used to be a PREFIX, layers 0..npin-1, and that interacted badly with the
     * ring in two compounding ways. The ring slot is uniform and must hold the largest
     * layer that still streams, so it was sized from the maximum over all UNPINNED
     * layers -- and prefix pinning takes layer 0 first, which at 2.34 GB is the largest
     * in the model (the only dense one, with a 33792-wide MLP). So the moment anything
     * was pinned at all, the budget was paying for a slot sized to a layer that no
     * longer used it, about 1.17 GB wasted at every budget above the floor.
     *
     * Largest first fixes both halves at once. For read volume the choice is neutral:
     * any pinned set totalling B bytes avoids exactly B bytes per token. For the ring it
     * is decisive, because dropping the biggest survivor is what shrinks the slot.
     *
     * The selection is greedy over layers sorted by size descending, and it keeps trying
     * SMALLER layers after a large one no longer fits -- so the budget is filled, not
     * merely walked. Ring size and pinned set remain mutually dependent (a smaller slot
     * frees budget, which pins more, which can shrink the slot again), so the whole
     * thing iterates to a fixed point. Selection is monotone in available budget, so the
     * loop converges rather than oscillating; it is bounded anyway.
     *
     * K3_PIN_PREFIX=1 restores the old order for an A/B on one binary. */
    const int pin_prefix = getenv("K3_PIN_PREFIX") != NULL;
    int *order = (int *)malloc((size_t)tr->n_layers * sizeof(int));
    unsigned char *chosen = (unsigned char *)calloc((size_t)tr->n_layers, 1);
    if (!order || !chosen) { free(order); free(chosen); return -1; }
    for (int i = 0; i < tr->n_layers; i++) order[i] = i;
    if (!pin_prefix) {
        /* Insertion sort by nbytes DESCENDING, ties broken by layer index ascending so
         * the selection is deterministic and a rerun pins the same set. 93 elements. */
        for (int i = 1; i < tr->n_layers; i++) {
            const int t = order[i];
            int j = i - 1;
            while (j >= 0 && tr->lay[order[j]].nbytes < tr->lay[t].nbytes) {
                order[j + 1] = order[j]; j--;
            }
            order[j + 1] = t;
        }
    }

    int64_t ring_slot = 0, spent = 0;
    int npin = 0;
    for (int pass = 0; pass < 8; pass++) {
        /* Largest UNPINNED layer: what the ring slot has to hold. */
        int64_t big = 0;
        for (int i = 0; i < tr->n_layers; i++)
            if (!chosen[i] && tr->lay[i].nbytes > big) big = tr->lay[i].nbytes;
        if (big == 0) big = tr->lay[tr->n_layers - 1].nbytes;   /* all pinned */
        int64_t rs = (big + K3_TRUNK_ALIGN - 1) & ~(int64_t)(K3_TRUNK_ALIGN - 1);
        rs += (int64_t)widen;
        rs = (rs + 4095) & ~(int64_t)4095;

        /* The ring itself must fit the budget before any layer is pinned. Drop to one
         * slot rather than overshoot. */
        RING = RING_WANT;
        while (RING > 1 && (int64_t)RING * rs > budget_bytes) RING--;

        int64_t sp = (int64_t)RING * rs;
        int np = 0;
        memset(chosen, 0, (size_t)tr->n_layers);
        for (int i = 0; i < tr->n_layers; i++) {
            const int L = order[i];
            const int64_t need = tr->lay[L].nbytes + (int64_t)widen;
            if (sp + need > budget_bytes) {
                /* A prefix pin stops at the first layer that does not fit, because the
                 * order is the walk order and skipping one would leave a hole in it.
                 * Largest-first has no such constraint: keep going and take the smaller
                 * ones that still fit. */
                if (pin_prefix) break;
                continue;
            }
            sp += need;
            chosen[L] = 1;
            np++;
        }
        if (rs == ring_slot && np == npin) { ring_slot = rs; spent = sp; break; }
        ring_slot = rs; npin = np; spent = sp;
    }

    tr->npin = npin;
    tr->nslot = RING;
    tr->slot_bytes = ring_slot;

    tr->pin_of = (int32_t *)malloc((size_t)tr->n_layers * sizeof(int32_t));
    tr->pin = (unsigned char **)calloc((size_t)(npin ? npin : 1), sizeof(unsigned char *));
    if (!tr->pin || !tr->pin_of) { free(order); free(chosen); return -1; }
    for (int i = 0; i < tr->n_layers; i++) tr->pin_of[i] = -1;
    {
        int k = 0;
        for (int L = 0; L < tr->n_layers && k < npin; L++) {
            if (!chosen[L]) continue;
            const size_t need = (size_t)((tr->lay[L].nbytes + K3_TRUNK_ALIGN - 1)
                                         & ~(int64_t)(K3_TRUNK_ALIGN - 1)) + widen;
            if (k3_alloc_direct((void **)&tr->pin[k], need) != 0) {
                fprintf(stderr, "k3_trunk: cannot allocate %.2f GB for pinned layer %d\n",
                        (double)need / 1e9, L);
                free(order); free(chosen);
                return -1;
            }
            tr->pin_of[L] = k++;
        }
    }
    free(order); free(chosen);
    if (k3_alloc_direct((void **)&tr->arena, (size_t)RING * (size_t)ring_slot) != 0) {
        fprintf(stderr, "k3_trunk: cannot allocate the %.2f GB streaming ring\n",
                (double)RING * ring_slot / 1e9);
        return -1;
    }
    tr->layer_of = (int *)malloc((size_t)RING * sizeof(int));
    for (int i = 0; i < RING; i++) tr->layer_of[i] = -1;
    tr->widen_bytes = (int64_t)widen;

    /* The io state always exists, because its mutex is what serialises the slot
     * bookkeeping even in the single-threaded case. The reader THREAD is started only
     * when there are at least two slots, and that is a correctness requirement rather
     * than an optimisation.
     *
     * k3_trunk_prefetch claims a slot for the incoming layer. With one slot that is
     * necessarily the slot k3_trunk_bind just returned to the caller, so the worker would
     * pread layer L+1 straight over layer L's bytes while the caller is still computing
     * on them. Nothing detects it: the read succeeds, no bound pointer changes, and the
     * run completes and emits fluent, wrong tokens.
     *
     * Measured on the released checkpoint. With one slot and the reader running, the same
     * prompt that gives 17374 20829 10 427 414 1008 606 142957 instead produced
     * 32609 2329 146429 2539 11 152834 44449 7569, with no diagnostic of any kind.
     *
     * `held` now enforces that invariant directly rather than leaving it to arithmetic,
     * so the single-slot case would in fact refuse to evict -- but the reader is still
     * withheld, because a ring that can never accept a prefetch has nothing for it to do
     * and would simply block. With running == 0, k3_trunk_prefetch returns immediately,
     * which is exactly the synchronous path this file had before the reader existed. */
    {
        K3TrunkIO *io = (K3TrunkIO *)calloc(1, sizeof *io);
        if (!io) return -1;
        io->tr = tr;
        io->held = -1;
        for (int i = 0; i < K3_TRUNK_MAXRING; i++) io->pending[i] = -1;
        pthread_mutex_init(&io->mu, NULL);
        pthread_cond_init(&io->cv_work, NULL);
        pthread_cond_init(&io->cv_done, NULL);
        tr->io_state = io;
        /* One ring per reading thread: an io_uring is not safe to drive from two
         * threads at once, and the whole point is that both can have reads in flight. */
        tr->uring = k3_uring_new(16);
        if (RING >= 2) {
            io->ur = k3_uring_new(16);
            if (pthread_create(&io->thread, NULL, trunk_io_main, io) != 0) {
                fprintf(stderr, "k3_trunk: cannot start asynchronous reader\n");
                k3_uring_free(io->ur);
                pthread_cond_destroy(&io->cv_work);
                pthread_cond_destroy(&io->cv_done);
                pthread_mutex_destroy(&io->mu);
                free(io);
                tr->io_state = NULL;
                return -1;
            }
            tr->reader_started = 1;
            io->running = 1;
        }
    }

    printf("trunk stream: %.2f GB packed, %d/%d layers PINNED (%.2f GB), "
           "ring %d x %.2f GB\n",
           (double)total / 1e9, npin, tr->n_layers,
           (double)(spent - (int64_t)RING * ring_slot) / 1e9,
           RING, (double)ring_slot / 1e9);
    {   /* Say which layers were pinned and what it bought, because "10/93 pinned" reads
         * the same whichever ten they were and the ring slot is the whole point. */
        int64_t biggest = 0, biggest_unpinned = 0;
        for (int i = 0; i < tr->n_layers; i++) {
            if (tr->lay[i].nbytes > biggest) biggest = tr->lay[i].nbytes;
            if (tr->pin_of[i] < 0 && tr->lay[i].nbytes > biggest_unpinned)
                biggest_unpinned = tr->lay[i].nbytes;
        }
        printf("              pinned %s; the ring slot is sized to the largest UNPINNED "
               "layer (%.2f GB\n              of a %.2f GB maximum)\n",
               npin == 0 ? "nothing"
                         : (getenv("K3_PIN_PREFIX") ? "as a PREFIX (K3_PIN_PREFIX)"
                                                    : "LARGEST FIRST"),
               (double)biggest_unpinned / 1e9, (double)biggest / 1e9);
    }
    printf("              reads use %s, %s\n",
           tr->direct ? "O_DIRECT (page cache bypassed)" : "buffered I/O",
           tr->uring
             ? (k3_uring_sqpoll((const K3Uring *)tr->uring)
                  ? "io_uring with SQPOLL" : "io_uring")
             : "pread at queue depth 1");
    if (tr->uring)
        printf("              up to %u requests in flight per layer read: an NVMe device "
               "does not reach\n              its rated bandwidth at depth one\n",
               k3_uring_depth((const K3Uring *)tr->uring));
    if (RING < RING_WANT)
        printf("              ring held at %d slot%s: %d slots need %.2f GB and the "
               "trunk budget is %.2f GB.\n"
               "              %s Raise --trunk-gb above %.2f GB for the "
               "%d asked for.\n",
               RING, RING == 1 ? "" : "s", RING_WANT,
               (double)RING_WANT * ring_slot / 1e9,
               (double)budget_bytes / 1e9,
               RING == 1 ? "Reads are NOT overlapped with compute."
                         : "Fewer reads stay in flight.",
               (double)RING_WANT * ring_slot / 1e9, RING_WANT);
    printf("              deterministic hit rate %.1f%% (a cyclic scan defeats LRU, so "
           "a pinned SET is used instead)\n", 100.0 * npin / tr->n_layers);
    return 0;
bad:
    free(txt);
    return -1;
}

void k3_trunk_close(K3Trunk *tr)
{
    K3TrunkIO *io = (K3TrunkIO *)tr->io_state;
    if (io) {
        if (io->running) {
            pthread_mutex_lock(&io->mu);
            io->stop = 1;
            pthread_cond_broadcast(&io->cv_work);
            pthread_mutex_unlock(&io->mu);
            pthread_join(io->thread, NULL);
        }
        k3_uring_free(io->ur);
        pthread_cond_destroy(&io->cv_work);
        pthread_cond_destroy(&io->cv_done);
        pthread_mutex_destroy(&io->mu);
        free(io);
    }
    k3_uring_free((K3Uring *)tr->uring);
    if (tr->fd >= 0) close(tr->fd);
    if (tr->pin) { for (int i = 0; i < tr->npin; i++) k3_aligned_free(tr->pin[i]); free(tr->pin); }
    k3_aligned_free(tr->arena); free(tr->layer_of); free(tr->slot_of); free(tr->pin_of);
    if (tr->lay) { for (int i = 0; i < tr->n_layers; i++) free(tr->lay[i].t); free(tr->lay); }
    free(tr->json_arena);   /* every K3TrunkTensor.name points into this */
    memset(tr, 0, sizeof *tr);
    tr->fd = -1;            /* see k3_trunk_open: 0 is stdin, not "closed" */
}

/* Read one layer's run into dst. */

/* Allocate an O_DIRECT target on a 2 MB boundary and ask for transparent hugepages.
 *
 * WHY THIS IS NOT COSMETIC. Every O_DIRECT read must pin its destination pages in the
 * kernel (get_user_pages) for the duration of the transfer. A 2.37 GB ring slot backed by
 * 4 KB pages is 578,000 pages pinned and released PER READ, and the trunk is read 93
 * times per token: about 53.8 million pin operations, at a few hundred nanoseconds each.
 * That is on the order of ten seconds per token spent in the kernel doing page
 * bookkeeping, none of which appears in the engine's own I/O timer -- which brackets only
 * the pread loop and therefore reports a device rate that looks like the disk is
 * saturated while a third of the token is unaccounted for.
 *
 * Backing the same buffer with 2 MB pages cuts the count by 512x. The allocation is
 * otherwise identical, so this is lossless and cannot change a single output bit.
 *
 * K3_NOHUGE=1 restores 4 KB alignment so the two can be A/B compared on ONE binary,
 * which is the only way to attribute a timing difference to this decision rather than to
 * the compiler or the weather. */
static int k3_alloc_direct(void **out, size_t bytes)
{
    const int huge = !getenv("K3_NOHUGE");
    const size_t align = huge ? (2u << 20) : 4096u;
    /* Round the LENGTH up too: madvise only covers whole pages, so a 2 MB-aligned start
     * with a ragged tail leaves the last stretch on 4 KB pages. */
    const size_t len = huge ? ((bytes + align - 1) & ~(align - 1)) : bytes;
    if (posix_memalign(out, align, len) != 0) return -1;
#if defined(MADV_HUGEPAGE)
    if (huge) madvise(*out, len, MADV_HUGEPAGE);   /* advisory: failure is not an error */
#endif
    return 0;
}

/* Read one layer's run. `ur` may be NULL, in which case this is the pread loop it has
 * always been; when present the transfer is split across the ring so several requests
 * are outstanding at once. The two are interchangeable byte for byte.
 *
 * Timing and byte counts come back through out-parameters rather than being added to
 * the K3Trunk here: two threads call this concurrently now, and += on a shared double is
 * a data race whose symptom is a plausible-looking throughput figure. */
static int load_run_to(K3Trunk *tr, int L, unsigned char *dst, K3Uring *ur,
                       double *secs, int64_t *bytes)
{
    const K3TrunkLayer *lay = &tr->lay[L];
    const double t0 = now_s();
    int64_t got = 0;

    if (ur) {
        got = k3_uring_read(ur, tr->fd, dst, lay->nbytes, lay->file_off);
        if (got == lay->nbytes) {
            *secs = now_s() - t0;
            *bytes = got;
            return 0;
        }
        got = 0;   /* io_uring refused this transfer: fall through to pread */
    }
    while (got < lay->nbytes) {
        const int64_t remain = lay->nbytes - got;
        const int64_t req = remain < K3_PREAD_MAX ? remain : K3_PREAD_MAX;
        ssize_t r = pread(tr->fd, dst + got, (size_t)req, (off_t)(lay->file_off + got));
        if (r <= 0) {
            fprintf(stderr, "k3_trunk: short read on layer %d\n", L);
            *secs = now_s() - t0; *bytes = got;
            return -1;
        }
        got += r;
    }
    *secs = now_s() - t0;
    *bytes = got;
    return 0;
}

/* Fold one completed read into the shared counters. Caller holds io->mu. */
static void tally_locked(K3Trunk *tr, double secs, int64_t bytes)
{
    tr->load_seconds += secs;
    tr->bytes_read += (uint64_t)bytes;
}

/* Is this slot holding a layer the walk is ABOUT to need?
 *
 * The third protection, after "being read into" and "in use by the caller". A layer the
 * prefetcher already fetched and published, but that the walk has not reached yet, is
 * neither of those -- so a claim for a further-ahead layer would evict it, and the walk
 * would arrive a moment later and read it AGAIN. That is invisible in the token stream
 * and shows up only as bytes: measured on the synthetic trunk at ring depth 2, 6.23 MB
 * read against a 4.72 MB bound, and on the released checkpoint as 491 GB read from a
 * 29.81 GB trunk over ten passes.
 *
 * It bites hardest immediately after a PINNED layer, because those hold no ring slot, so
 * `held` is -1 and every slot looks claimable.
 *
 * The window is bounded by the ring itself: a layer more than nslot ahead cannot have
 * been prefetched by this pass, so a slot holding one is stale from the PREVIOUS pass --
 * layers 91 and 92 still sitting there when the walk has wrapped to layer 0 -- and must
 * stay evictable, or a two-slot ring would deadlock at the start of every pass. */
static int upcoming(const K3Trunk *tr, int s)
{
    const int L = tr->layer_of[s];
    return L > tr->walk && L <= tr->walk + tr->nslot;
}

/* Take a ring slot for layer L, evicting whatever it held. Caller holds io->mu.
 * Returns -1 when every slot is either being read into or in use by the caller, which
 * is a normal transient rather than an error: wait on cv_done and retry. */
static int claim_slot_locked(K3Trunk *tr, K3TrunkIO *io, int L)
{
    for (int i = 0; i < tr->nslot; i++) {
        const int s = (tr->ring + i) % tr->nslot;
        if (io->pending[s] >= 0) continue;      /* a read is landing in it  */
        if (s == io->held) continue;            /* the caller is reading it */
        if (upcoming(tr, s)) continue;          /* the walk needs it next   */
        tr->ring = (s + 1) % tr->nslot;
        if (tr->layer_of[s] >= 0) tr->slot_of[tr->layer_of[s]] = -1;
        tr->layer_of[s] = -1;                   /* EMPTY before the read, not after */
        io->pending[s] = L;
        return s;
    }
    return -1;
}

static int pending_slot_locked(const K3Trunk *tr, const K3TrunkIO *io, int L)
{
    for (int s = 0; s < tr->nslot; s++) if (io->pending[s] == L) return s;
    return -1;
}

static void *trunk_io_main(void *arg)
{
    K3TrunkIO *io = (K3TrunkIO *)arg;
    K3Trunk *tr = io->tr;
    for (;;) {
        pthread_mutex_lock(&io->mu);
        while (io->nq == 0 && !io->stop)
            pthread_cond_wait(&io->cv_work, &io->mu);
        if (io->stop) { pthread_mutex_unlock(&io->mu); return NULL; }
        const int L = io->q[0], slot = io->qslot[0];
        for (int i = 1; i < io->nq; i++) { io->q[i - 1] = io->q[i]; io->qslot[i - 1] = io->qslot[i]; }
        io->nq--;
        pthread_mutex_unlock(&io->mu);

        double secs = 0.0; int64_t bytes = 0;
        const int rc = load_run_to(tr, L, tr->arena + (size_t)slot * tr->slot_bytes,
                                   io->ur, &secs, &bytes);

        pthread_mutex_lock(&io->mu);
        tally_locked(tr, secs, bytes);
        io->pending[slot] = -1;
        if (rc == 0) {
            /* Publish ONLY after the read succeeded. Registering the layer up front
             * would let a failed read serve a slot that claims to hold weights it does
             * not, and the next bind would count it as a hit and multiply garbage.
             *
             * Deliberately NOT counted as a miss here. hits and misses are per BIND, and
             * a bind that waits for a prefetch is one bind, not one miss plus one hit;
             * k3_trunk_bind owns that classification so the two always sum to the bind
             * count. Counting here as well inflated the hit rate by the prefetch rate,
             * which is precisely the number the rate is meant to expose. */
            tr->layer_of[slot] = L;
            tr->slot_of[L] = slot;
        }
        pthread_cond_broadcast(&io->cv_done);
        pthread_mutex_unlock(&io->mu);
    }
}

int k3_trunk_fetch(K3Trunk *tr, int L, unsigned char **out)
{
    if (L < 0 || L >= tr->n_layers) return -1;
    unsigned char *base;
    K3TrunkIO *io = (K3TrunkIO *)tr->io_state;

    /* Where the walk is. Everything at or behind this has been consumed and its slot may
     * be reused; a short way ahead of it is what the prefetcher has in flight and must
     * not lose. Updated under the lock because claim_slot_locked reads it. */
    pthread_mutex_lock(&io->mu);
    tr->walk = L;
    pthread_mutex_unlock(&io->mu);

    if (tr->pin_of[L] >= 0) {
        base = tr->pin[tr->pin_of[L]];
        if (tr->slot_of[L] < 0) {            /* first touch: load once, keep forever */
            double secs = 0.0; int64_t bytes = 0;
            /* tr->uring, NOT io->ur. An io_uring is a shared-memory protocol between one
             * submitter and the kernel; driving one ring from two threads interleaves
             * their head and tail updates, and the symptom is not corruption but a HANG:
             * one thread reaps the other's completion, and the other waits forever for a
             * completion that has already been consumed. That is exactly what handing
             * the reader's ring to the main thread here did, and it reproduced as an
             * intermittent stall in tests/unit/test_trunk.c at ring depths 2 and 3. */
            const int rc = load_run_to(tr, L, base, tr->uring, &secs, &bytes);
            pthread_mutex_lock(&io->mu);
            tally_locked(tr, secs, bytes);
            if (rc == 0) { tr->slot_of[L] = L; tr->misses++; }
            pthread_mutex_unlock(&io->mu);
            if (rc != 0) return -1;
        } else {
            tr->hits++;
        }
        pthread_mutex_lock(&io->mu);
        io->held = -1;                       /* no ring slot is in use */
        pthread_cond_broadcast(&io->cv_done);
        pthread_mutex_unlock(&io->mu);
    } else {
        int slot;
        pthread_mutex_lock(&io->mu);
        io->held = -1;    /* release the previous layer's slot before asking for one */
        pthread_cond_broadcast(&io->cv_done);
        /* Classify this bind ONCE, on the first look. Resident already means no bytes
         * moved for it; anything else means bytes did, whether this thread read them or
         * it waited for the reader to finish. hits + misses therefore equals the bind
         * count, which is what makes the printed rate mean anything. */
        int first = 1;
        for (;;) {
            slot = tr->slot_of[L];
            if (slot >= 0) { if (first) tr->hits++; break; }
            if (first) { tr->misses++; first = 0; }
            const int pend = pending_slot_locked(tr, io, L);
            if (pend >= 0) {                 /* the reader is already fetching it */
                pthread_cond_wait(&io->cv_done, &io->mu);
                continue;
            }
            slot = claim_slot_locked(tr, io, L);
            if (slot >= 0) {                 /* fetch it on this thread */
                pthread_mutex_unlock(&io->mu);
                double secs = 0.0; int64_t bytes = 0;
                const int rc = load_run_to(tr, L,
                                           tr->arena + (size_t)slot * tr->slot_bytes,
                                           tr->uring, &secs, &bytes);
                pthread_mutex_lock(&io->mu);
                tally_locked(tr, secs, bytes);
                io->pending[slot] = -1;
                if (rc == 0) { tr->layer_of[slot] = L; tr->slot_of[L] = slot; }
                pthread_cond_broadcast(&io->cv_done);
                if (rc != 0) {
                    /* A bind that fails moved no usable bytes and published nothing,
                     * so it is not a cache event: hits + misses counts COMPLETED binds,
                     * and test_trunk's truncated case holds the reader to that. The
                     * miss was charged on the first look above; take it back. */
                    tr->misses--;
                    pthread_mutex_unlock(&io->mu);
                    return -1;
                }
                break;
            }
            pthread_cond_wait(&io->cv_done, &io->mu);   /* every slot is busy */
        }
        io->held = slot;    /* nothing may evict this until the next bind */
        pthread_mutex_unlock(&io->mu);
        base = tr->arena + (size_t)slot * tr->slot_bytes;
    }
    *out = base;
    return 0;
}

int k3_trunk_bind(K3Trunk *tr, const K3Cfg *c, int L, K3LayerBind *b)
{
    const double t_bind0 = now_s();
    k3_trunk_binds++;
    unsigned char *base = NULL;
    if (k3_trunk_fetch(tr, L, &base) != 0) return -1;

    Finder f; f.L = &tr->lay[L];
    K3MemSrc src; src.find = find_in_layer; src.ctx = &f;
    unsigned char *widen = base + (((tr->lay[L].nbytes + K3_TRUNK_ALIGN - 1)
                                    & ~(int64_t)(K3_TRUNK_ALIGN - 1)));
    /* Pinned layers own exactly nbytes + widen; ring slots own slot_bytes. */
    const size_t cap = (size_t)tr->widen_bytes;
    const double tw = now_s();
    const int rc = k3_bind_layer_mem(c, L, b, base, &src, widen, cap, NULL);
    const double tnow = now_s();
    k3_trunk_widen_wall += tnow - tw;
    k3_trunk_bind_wall  += tnow - t_bind0;
    return rc;
}

/* Queue the upcoming layers, as many as the ring can hold at once.
 *
 * The walk order is fixed 0..92 on every token, so there is nothing to predict and a
 * hint is never wrong -- which is what makes running SEVERAL layers ahead safe rather
 * than speculative. Depth is bounded by the ring itself: claim_slot_locked refuses the
 * slot in use and any slot already being read into, so a ring of R slots can carry at
 * most R-1 outstanding reads and this loop simply stops when it runs out.
 *
 * Pinned and already-resident layers are SKIPPED rather than ending the scan, so a run
 * of pinned layers ahead does not stall the prefetcher: it looks past them to the next
 * layer that actually needs reading. */
void k3_trunk_prefetch(K3Trunk *tr, int L)
{
    K3TrunkIO *io = (K3TrunkIO *)tr->io_state;
    if (!io || !io->running || L < 0) return;

    pthread_mutex_lock(&io->mu);
    int queued = 0;
    for (int n = L; n < tr->n_layers && io->nq < tr->nslot; n++) {
        if (tr->pin_of[n] >= 0) continue;
        if (tr->slot_of[n] >= 0) continue;
        if (pending_slot_locked(tr, io, n) >= 0) continue;
        const int slot = claim_slot_locked(tr, io, n);
        if (slot < 0) break;                 /* no free slot: the ring is full */
        io->q[io->nq] = n;
        io->qslot[io->nq] = slot;
        io->nq++;
        queued++;
    }
    if (queued) pthread_cond_signal(&io->cv_work);
    pthread_mutex_unlock(&io->mu);
}

void k3_trunk_report(const K3Trunk *tr, const char *label)
{
    const uint64_t n = tr->hits + tr->misses;
    printf("trunk [%s]\n", label ? label : "");
    printf("  pinned %d/%d layers, ring %d slots\n", tr->npin, tr->n_layers, tr->nslot);
    printf("  binds %llu, hits %llu (%.1f%%), reads %llu\n",
           (unsigned long long)n, (unsigned long long)tr->hits,
           n ? 100.0 * tr->hits / n : 0.0, (unsigned long long)tr->misses);
    printf("  read %.2f GB in %.2f s (%.0f MB/s)\n",
           (double)tr->bytes_read / 1e9, tr->load_seconds,
           tr->load_seconds > 0 ? (double)tr->bytes_read / 1e6 / tr->load_seconds : 0.0);
    /* The rate above is a DEVICE rate: load_seconds brackets the pread loop alone. The
     * breakdown below is the wall clock actually spent inside k3_trunk_bind, so the
     * difference between them is per-bind overhead rather than disk time.
     *
     * Reporting the widen step separately is what distinguishes a slow device from
     * excessive work per bind, two causes with the same symptom and different fixes. */
    {
        /* load_seconds is DEVICE time and, with more than one ring slot, some of it
         * happens on the reader thread while the main thread is computing. Subtracting it
         * from bind wall clock then goes negative by exactly the amount of overlap
         * achieved, which is how the previous form of this line reported the feature
         * working as "other -157.06" and a read share of 207%. Overlapped time is a
         * result, not unattributed overhead, so it is named rather than subtracted. */
        const double serial = tr->load_seconds + k3_trunk_widen_wall;
        const double overlapped = serial - k3_trunk_bind_wall;
        if (overlapped > 0.0) {
            printf("  bind wall %.2f s over %ld binds; read %.2f + widen %.2f = %.2f s of "
                   "device work,\n"
                   "                    of which %.2f s (%.0f%%) overlapped compute on the "
                   "reader thread\n",
                   k3_trunk_bind_wall, k3_trunk_binds, tr->load_seconds,
                   k3_trunk_widen_wall, serial, overlapped,
                   serial > 0.0 ? 100.0 * overlapped / serial : 0.0);
        } else {
            const double other = k3_trunk_bind_wall - serial;
            printf("  bind wall %.2f s over %ld binds  =  read %.2f + widen %.2f + other %.2f\n",
                   k3_trunk_bind_wall, k3_trunk_binds, tr->load_seconds,
                   k3_trunk_widen_wall, other);
            if (k3_trunk_bind_wall > 0.0)
                printf("                    shares:      read %.0f%%  widen %.0f%%  other %.0f%%\n",
                       100.0 * tr->load_seconds / k3_trunk_bind_wall,
                       100.0 * k3_trunk_widen_wall / k3_trunk_bind_wall,
                       100.0 * other / k3_trunk_bind_wall);
        }
    }
}
