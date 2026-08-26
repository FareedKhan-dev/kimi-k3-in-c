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

static int k3_alloc_direct(void **out, size_t bytes);   /* defined below */

typedef struct {
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    K3Trunk *tr;
    int stop;
    int busy;
    int done;
    int layer;
    int slot;
    int result;
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

int k3_trunk_open(K3Trunk *tr, const char *dir, const K3Cfg *c, int64_t budget_bytes)
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

    /* Two ring slots: the layer being computed on, plus one asynchronous read in flight.
     *
     * This is a REQUEST, not a guarantee. The second slot costs a full slot's worth of
     * memory, which at the floor is 2.37 GB, and the budget the caller asked for has to
     * come first: measured on the released checkpoint, taking the second slot
     * unconditionally moved the laptop preset from 8.78 GB to 11.12 GB peak RSS, a 27%
     * overshoot of a 3.0 GB trunk budget, and left the printed memory plan understating
     * the real figure. So it is granted only when it fits, and reported when it does not.
     * A single slot is exactly what this file did before the asynchronous reader existed,
     * so falling back is always safe; it costs speed, not correctness. */
    const int RING_WANT = 2;
    int RING = RING_WANT;

    /* Size the ring from the layers that will actually STREAM through it.
     *
     * Pinning is a PREFIX: layers 0..npin-1 are held resident and never touch the ring,
     * so only npin..n_layers-1 ever occupy a slot. Sizing the slot from the maximum over
     * ALL layers therefore reserves room for layer 0 -- which at 2.34 GB is the largest
     * in the model, being the only dense one with a 33792-wide MLP, and which prefix
     * pinning pins FIRST whenever anything is pinned at all. That wasted about 1.17 GB
     * for nothing at every budget above the floor.
     *
     * Ring size and pin count are mutually dependent: a smaller ring frees budget, which
     * pins more layers, which can shrink the ring again. Iterate to a fixed point. It
     * converges in two or three passes and is monotone, so the loop is bounded. At the
     * floor, where npin is 0, this correctly changes nothing: every layer streams and the
     * ring must still hold the biggest of them. */
    int64_t ring_slot = 0, spent = 0;
    int npin = 0;
    for (int pass = 0; pass < 4; pass++) {
        int64_t big = 0;
        for (int i = npin; i < tr->n_layers; i++)
            if (tr->lay[i].nbytes > big) big = tr->lay[i].nbytes;
        if (big == 0) big = tr->lay[tr->n_layers - 1].nbytes;   /* all pinned */
        int64_t rs = (big + K3_TRUNK_ALIGN - 1) & ~(int64_t)(K3_TRUNK_ALIGN - 1);
        rs += (int64_t)widen;
        rs = (rs + 4095) & ~(int64_t)4095;

        /* The ring itself must fit the budget before any layer is pinned. The loop below
         * only ever tested ADDITIONAL pinned layers against it, so RING * rs was spent
         * whether or not it fitted. Drop to one slot rather than overshoot. */
        RING = RING_WANT;
        while (RING > 1 && (int64_t)RING * rs > budget_bytes) RING--;

        int64_t sp = (int64_t)RING * rs;
        int np = 0;
        while (np < tr->n_layers) {
            const int64_t need = tr->lay[np].nbytes + (int64_t)widen;
            if (sp + need > budget_bytes) break;
            sp += need;
            np++;
        }
        if (np >= tr->n_layers) np = tr->n_layers;
        if (rs == ring_slot && np == npin) { ring_slot = rs; spent = sp; break; }
        ring_slot = rs; npin = np; spent = sp;
    }

    tr->npin = npin;
    tr->nslot = RING;
    tr->slot_bytes = ring_slot;

    tr->pin = (unsigned char **)calloc((size_t)(npin ? npin : 1), sizeof(unsigned char *));
    if (!tr->pin) return -1;
    for (int i = 0; i < npin; i++) {
        const size_t need = (size_t)((tr->lay[i].nbytes + K3_TRUNK_ALIGN - 1)
                                     & ~(int64_t)(K3_TRUNK_ALIGN - 1)) + widen;
        if (k3_alloc_direct((void **)&tr->pin[i], need) != 0) {
            fprintf(stderr, "k3_trunk: cannot allocate %.2f GB for pinned layer %d\n",
                    (double)need / 1e9, i);
            return -1;
        }
    }
    if (k3_alloc_direct((void **)&tr->arena, (size_t)RING * (size_t)ring_slot) != 0) {
        fprintf(stderr, "k3_trunk: cannot allocate the %.2f GB streaming ring\n",
                (double)RING * ring_slot / 1e9);
        return -1;
    }
    tr->layer_of = (int *)malloc((size_t)RING * sizeof(int));
    for (int i = 0; i < RING; i++) tr->layer_of[i] = -1;
    tr->widen_bytes = (int64_t)widen;

    /* The reader is started ONLY when there are at least two slots, and this is a
     * correctness requirement rather than an optimisation.
     *
     * k3_trunk_prefetch claims tr->ring for the incoming layer. With one slot, tr->ring
     * is necessarily the slot k3_trunk_bind just returned to the caller, so the worker
     * preads layer L+1 straight over layer L's bytes while the caller is still computing
     * on them. Nothing detects it: the read succeeds, no bound pointer changes, and the
     * run completes and emits fluent, wrong tokens.
     *
     * Measured on the released checkpoint. With one slot and the reader running, the same
     * prompt that gives 17374 20829 10 427 414 1008 606 142957 instead produced
     * 32609 2329 146429 2539 11 152834 44449 7569, with no diagnostic of any kind.
     *
     * With io_state NULL, trunk_io_wait returns 0 and k3_trunk_prefetch returns
     * immediately, which is exactly the synchronous path this file had before the reader
     * existed. */
    if (RING >= 2) {
        K3TrunkIO *io = (K3TrunkIO *)calloc(1, sizeof *io);
        if (!io) return -1;
        io->tr = tr;
        pthread_mutex_init(&io->mu, NULL);
        pthread_cond_init(&io->cv, NULL);
        tr->io_state = io;
        if (pthread_create(&io->thread, NULL, trunk_io_main, io) != 0) {
            fprintf(stderr, "k3_trunk: cannot start asynchronous reader\n");
            pthread_cond_destroy(&io->cv);
            pthread_mutex_destroy(&io->mu);
            free(io);
            tr->io_state = NULL;
            return -1;
        }
    } else {
        tr->io_state = NULL;
    }

    printf("trunk stream: %.2f GB packed, %d/%d layers PINNED (%.2f GB), "
           "ring %d x %.2f GB\n",
           (double)total / 1e9, npin, tr->n_layers,
           (double)(spent - (int64_t)RING * ring_slot) / 1e9,
           RING, (double)ring_slot / 1e9);
    printf("              reads use %s\n",
           tr->direct ? "O_DIRECT (page cache bypassed)" : "buffered I/O");
    if (RING < RING_WANT)
        printf("              ring held at %d slot: a second slot needs %.2f GB and the "
               "trunk budget is %.2f GB,\n"
               "              so reads are NOT overlapped with compute. Raise --trunk-gb "
               "above %.2f GB to enable it.\n",
               RING, (double)RING_WANT * ring_slot / 1e9,
               (double)budget_bytes / 1e9,
               (double)RING_WANT * ring_slot / 1e9);
    printf("              deterministic hit rate %.1f%% (a cyclic scan defeats LRU, so "
           "a pinned prefix is used instead)\n", 100.0 * npin / tr->n_layers);
    return 0;
bad:
    free(txt);
    return -1;
}

void k3_trunk_close(K3Trunk *tr)
{
    K3TrunkIO *io = (K3TrunkIO *)tr->io_state;
    if (io) {
        pthread_mutex_lock(&io->mu);
        io->stop = 1;
        pthread_cond_signal(&io->cv);
        pthread_mutex_unlock(&io->mu);
        pthread_join(io->thread, NULL);
        pthread_cond_destroy(&io->cv);
        pthread_mutex_destroy(&io->mu);
        free(io);
    }
    if (tr->fd >= 0) close(tr->fd);
    if (tr->pin) { for (int i = 0; i < tr->npin; i++) k3_aligned_free(tr->pin[i]); free(tr->pin); }
    k3_aligned_free(tr->arena); free(tr->layer_of); free(tr->slot_of);
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

static int load_run(K3Trunk *tr, int L, unsigned char *dst)
{
    const K3TrunkLayer *lay = &tr->lay[L];
    const double t0 = now_s();
    int64_t got = 0;
    while (got < lay->nbytes) {
        const int64_t remain = lay->nbytes - got;
        const int64_t req = remain < K3_PREAD_MAX ? remain : K3_PREAD_MAX;
        ssize_t r = pread(tr->fd, dst + got, (size_t)req, (off_t)(lay->file_off + got));
        if (r <= 0) { fprintf(stderr, "k3_trunk: short read on layer %d\n", L); return -1; }
        got += r;
    }
    tr->load_seconds += now_s() - t0;
    tr->bytes_read += (uint64_t)got;
    return 0;
}

static void *trunk_io_main(void *arg)
{
    K3TrunkIO *io = (K3TrunkIO *)arg;
    for (;;) {
        pthread_mutex_lock(&io->mu);
        while (!io->busy && !io->stop)
            pthread_cond_wait(&io->cv, &io->mu);
        if (io->stop) {
            pthread_mutex_unlock(&io->mu);
            return NULL;
        }
        const int L = io->layer;
        const int slot = io->slot;
        K3Trunk *tr = io->tr;
        pthread_mutex_unlock(&io->mu);

        const int rc = load_run(tr, L, tr->arena + (size_t)slot * tr->slot_bytes);

        pthread_mutex_lock(&io->mu);
        io->result = rc;
        io->done = 1;
        io->busy = 0;
        pthread_cond_broadcast(&io->cv);
        pthread_mutex_unlock(&io->mu);
    }
}

static int trunk_io_wait(K3Trunk *tr, int L)
{
    K3TrunkIO *io = (K3TrunkIO *)tr->io_state;
    if (!io) return 0;
    pthread_mutex_lock(&io->mu);
    if ((io->busy || io->done) && io->layer == L) {
        while (!io->done && !io->stop)
            pthread_cond_wait(&io->cv, &io->mu);
        const int rc = io->result;
        const int slot = io->slot;
        if (!io->stop && rc == 0) {
            tr->layer_of[slot] = L;
            tr->slot_of[L] = slot;
            tr->misses++;
        }
        io->done = 0;
        pthread_mutex_unlock(&io->mu);
        return rc == 0 ? 1 : -1;
    }
    pthread_mutex_unlock(&io->mu);
    return 0;
}

int k3_trunk_bind(K3Trunk *tr, const K3Cfg *c, int L, K3LayerBind *b)
{
    if (L < 0 || L >= tr->n_layers) return -1;
    const double t_bind0 = now_s();
    k3_trunk_binds++;
    unsigned char *base;

    if (L < tr->npin) {
        base = tr->pin[L];
        if (tr->slot_of[L] < 0) {            /* first touch: load once, keep forever */
            if (load_run(tr, L, base) != 0) return -1;
            tr->slot_of[L] = L;
            tr->misses++;
        } else {
            tr->hits++;
        }
    } else {
        int slot = -1;
        const int prefetched = trunk_io_wait(tr, L);
        if (prefetched < 0) return -1;
        if (prefetched > 0) {
            slot = tr->slot_of[L];
        } else {
            for (int i = 0; i < tr->nslot; i++)
                if (tr->layer_of[i] == L) { slot = i; break; }
            if (slot >= 0) {
                tr->hits++;
            } else {
                slot = tr->ring;
                tr->ring = (tr->ring + 1) % tr->nslot;
                if (tr->layer_of[slot] >= 0) tr->slot_of[tr->layer_of[slot]] = -1;
                /* Mark the slot EMPTY before reading into it, not after. */
                tr->layer_of[slot] = -1;
                if (load_run(tr, L, tr->arena + (size_t)slot * tr->slot_bytes) != 0) return -1;
                tr->layer_of[slot] = L;
                tr->misses++;
            }
        }
        base = tr->arena + (size_t)slot * tr->slot_bytes;
    }

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

void k3_trunk_prefetch(K3Trunk *tr, int L)
{
    if (L < 0 || L >= tr->n_layers || L < tr->npin) return;
    for (int i = 0; i < tr->nslot; i++) if (tr->layer_of[i] == L) return;

    K3TrunkIO *io = (K3TrunkIO *)tr->io_state;
    if (!io) return;
    pthread_mutex_lock(&io->mu);
    if (io->busy || tr->slot_of[L] >= 0) {
        pthread_mutex_unlock(&io->mu);
        return;
    }
    const int slot = tr->ring;
    tr->ring = (tr->ring + 1) % tr->nslot;
    if (tr->layer_of[slot] >= 0) tr->slot_of[tr->layer_of[slot]] = -1;
    tr->layer_of[slot] = -1;
    io->layer = L;
    io->slot = slot;
    io->done = 0;
    io->busy = 1;
    pthread_cond_signal(&io->cv);
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
