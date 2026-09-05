/* k3_uring.c - see k3_uring.h.
 *
 * The whole of io_uring that this engine needs is: set up a ring, put IORING_OP_READ
 * entries in the submission queue, tell the kernel, and reap completions. Nothing here
 * is generic; it is a fixed-size ring used by one thread at a time to read one
 * contiguous run, and it deliberately does not try to be more.
 *
 * MEMORY ORDERING IS NOT DECORATION. The rings are shared memory between this process
 * and the kernel, and the only thing that makes them safe is the acquire/release pairing
 * on the head and tail indices: the kernel must not see a tail advance before the SQE it
 * describes is written, and this process must not read a CQE before observing the head
 * that published it. Plain loads and stores here work on x86 and corrupt silently on
 * arm64.
 */
#define _GNU_SOURCE
#include "k3_uring.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#include <errno.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <linux/io_uring.h>

/* The three io_uring syscalls have no glibc wrappers on every supported distribution,
 * so they are issued directly. This is the entire dependency surface. */
static int sys_io_uring_setup(unsigned entries, struct io_uring_params *p)
{
    return (int)syscall(__NR_io_uring_setup, entries, p);
}

static int sys_io_uring_enter(int fd, unsigned to_submit, unsigned min_complete,
                              unsigned flags)
{
    return (int)syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags,
                        (void *)NULL, (size_t)0);
}

struct K3Uring {
    int       fd;
    unsigned  entries;
    int       sqpoll;

    /* submission queue */
    void     *sq_ptr;   size_t sq_sz;
    unsigned *sq_head, *sq_tail, *sq_mask, *sq_flags;
    unsigned *sq_array;
    struct io_uring_sqe *sqes;  size_t sqes_sz;

    /* completion queue */
    void     *cq_ptr;   size_t cq_sz;
    unsigned *cq_head, *cq_tail, *cq_mask;
    struct io_uring_cqe *cqes;
};

static void uring_unmap(K3Uring *u)
{
    if (u->sqes && u->sqes != MAP_FAILED) munmap(u->sqes, u->sqes_sz);
    if (u->cq_ptr && u->cq_ptr != MAP_FAILED && u->cq_ptr != u->sq_ptr)
        munmap(u->cq_ptr, u->cq_sz);
    if (u->sq_ptr && u->sq_ptr != MAP_FAILED) munmap(u->sq_ptr, u->sq_sz);
    u->sqes = NULL; u->cq_ptr = NULL; u->sq_ptr = NULL;
}

K3Uring *k3_uring_new(unsigned entries)
{
    if (getenv("K3_NOURING")) return NULL;
    if (entries < 2) entries = 2;
    if (entries > 256) entries = 256;

    K3Uring *u = (K3Uring *)calloc(1, sizeof *u);
    if (!u) return NULL;

    struct io_uring_params p;
    memset(&p, 0, sizeof p);
    /* SQPOLL takes a kernel thread that spins on the submission queue, which removes a
     * syscall per submit and costs a core. Opt-in, and if the kernel refuses (it needs
     * privileges on older releases) fall straight back to a plain ring rather than
     * giving up on io_uring altogether. */
    const int want_sqpoll = getenv("K3_SQPOLL") != NULL;
    if (want_sqpoll) { p.flags = IORING_SETUP_SQPOLL; p.sq_thread_idle = 2000; }
    int fd = sys_io_uring_setup(entries, &p);
    if (fd < 0 && want_sqpoll) {
        memset(&p, 0, sizeof p);
        fd = sys_io_uring_setup(entries, &p);
    }
    if (fd < 0) { free(u); return NULL; }
    u->fd = fd;
    u->entries = p.sq_entries;
    u->sqpoll = (p.flags & IORING_SETUP_SQPOLL) ? 1 : 0;

    u->sq_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    u->cq_sz = p.cq_off.cqes  + p.cq_entries * sizeof(struct io_uring_cqe);
    /* Since 5.4 the kernel can back both rings with one mapping and says so with
     * IORING_FEAT_SINGLE_MMAP. Handle both, because getting this wrong maps a second
     * region that overlaps the first and every index read is garbage. */
    if (p.features & IORING_FEAT_SINGLE_MMAP) {
        if (u->cq_sz > u->sq_sz) u->sq_sz = u->cq_sz;
        u->cq_sz = u->sq_sz;
    }
    u->sq_ptr = mmap(NULL, u->sq_sz, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
    if (u->sq_ptr == MAP_FAILED) { close(fd); free(u); return NULL; }
    u->cq_ptr = (p.features & IORING_FEAT_SINGLE_MMAP)
              ? u->sq_ptr
              : mmap(NULL, u->cq_sz, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
    if (u->cq_ptr == MAP_FAILED) { uring_unmap(u); close(fd); free(u); return NULL; }

    u->sqes_sz = p.sq_entries * sizeof(struct io_uring_sqe);
    u->sqes = (struct io_uring_sqe *)mmap(NULL, u->sqes_sz, PROT_READ | PROT_WRITE,
                                          MAP_SHARED | MAP_POPULATE, fd,
                                          IORING_OFF_SQES);
    if (u->sqes == MAP_FAILED) { uring_unmap(u); close(fd); free(u); return NULL; }

    unsigned char *sq = (unsigned char *)u->sq_ptr;
    unsigned char *cq = (unsigned char *)u->cq_ptr;
    u->sq_head  = (unsigned *)(sq + p.sq_off.head);
    u->sq_tail  = (unsigned *)(sq + p.sq_off.tail);
    u->sq_mask  = (unsigned *)(sq + p.sq_off.ring_mask);
    u->sq_flags = (unsigned *)(sq + p.sq_off.flags);
    u->sq_array = (unsigned *)(sq + p.sq_off.array);
    u->cq_head  = (unsigned *)(cq + p.cq_off.head);
    u->cq_tail  = (unsigned *)(cq + p.cq_off.tail);
    u->cq_mask  = (unsigned *)(cq + p.cq_off.ring_mask);
    u->cqes     = (struct io_uring_cqe *)(cq + p.cq_off.cqes);
    return u;
}

void k3_uring_free(K3Uring *u)
{
    if (!u) return;
    uring_unmap(u);
    if (u->fd >= 0) close(u->fd);
    free(u);
}

unsigned k3_uring_depth(const K3Uring *u) { return u ? u->entries : 0u; }
int      k3_uring_sqpoll(const K3Uring *u) { return u ? u->sqpoll : 0; }

/* One chunk of a split read. `done` tracks partial completions so a short read is
 * resubmitted at the right offset instead of being mistaken for a failure. */
typedef struct { int64_t off, len, done; unsigned char *buf; } Chunk;

/* 8 MB per request: large enough that per-request overhead is irrelevant against a
 * 1.27 GB layer, small enough that a ring of 8 covers 64 MB and keeps the device busy
 * through the whole run. It is a multiple of 4096, so an O_DIRECT-aligned request stays
 * aligned after the split -- which is a correctness property, not a tuning choice. */
#define K3_URING_CHUNK (8 << 20)

int64_t k3_uring_read(K3Uring *u, int fd, void *buf, int64_t nbytes, int64_t off)
{
    if (!u || nbytes <= 0) return 0;

    const int64_t nchunk = (nbytes + K3_URING_CHUNK - 1) / K3_URING_CHUNK;
    Chunk *ck = (Chunk *)malloc((size_t)nchunk * sizeof(Chunk));
    if (!ck) return -1;
    for (int64_t i = 0; i < nchunk; i++) {
        const int64_t a = i * (int64_t)K3_URING_CHUNK;
        ck[i].off = off + a;
        ck[i].len = (nbytes - a < K3_URING_CHUNK) ? nbytes - a : K3_URING_CHUNK;
        ck[i].done = 0;
        ck[i].buf = (unsigned char *)buf + a;
    }

    int64_t next = 0;           /* next chunk never yet written into the SQ */
    int     posted = 0;         /* chunks in the SQ or in the kernel, not yet reaped */
    int64_t got = 0;
    int     failed = 0;
    /* Chunks needing resubmission after a short read, as a simple stack. */
    int64_t *again = (int64_t *)malloc((size_t)nchunk * sizeof(int64_t));
    int      nagain = 0;
    if (!again) { free(ck); return -1; }

    while (!failed && (next < nchunk || nagain > 0 || posted > 0)) {
        /* ---- fill the submission queue ---- */
        unsigned tail = __atomic_load_n(u->sq_tail, __ATOMIC_RELAXED);
        unsigned head = __atomic_load_n(u->sq_head, __ATOMIC_ACQUIRE);
        unsigned queued = 0;
        while ((tail - head) < u->entries && (next < nchunk || nagain > 0)) {
            const int64_t idx = nagain > 0 ? again[--nagain] : next++;
            struct io_uring_sqe *sqe = &u->sqes[tail & *u->sq_mask];
            memset(sqe, 0, sizeof *sqe);
            sqe->opcode    = IORING_OP_READ;
            sqe->fd        = fd;
            sqe->off       = (unsigned long long)(ck[idx].off + ck[idx].done);
            sqe->addr      = (unsigned long long)(uintptr_t)(ck[idx].buf + ck[idx].done);
            sqe->len       = (unsigned)(ck[idx].len - ck[idx].done);
            sqe->user_data = (unsigned long long)idx;
            u->sq_array[tail & *u->sq_mask] = tail & *u->sq_mask;
            tail++;
            queued++;
            posted++;
        }
        if (queued) {
            /* RELEASE: the SQEs above must be visible to the kernel before the tail
             * that publishes them. */
            __atomic_store_n(u->sq_tail, tail, __ATOMIC_RELEASE);
        }

        /* ---- hand them to the kernel ----
         * to_submit is recomputed from the RING, not from what this iteration happened
         * to queue. io_uring_enter returns how many SQEs it actually consumed and is
         * entitled to consume fewer than asked; passing `queued` and moving on leaves
         * the remainder sitting in the submission queue forever, and the next call --
         * with nothing new to submit -- then waits for a completion that was never going
         * to arrive. That is a hang, not a slowdown, and it reproduced as an intermittent
         * stall in tests/unit/test_trunk.c before this was written this way.
         *
         * min_complete likewise counts only what the KERNEL already holds. Waiting on a
         * request that is still sitting unsubmitted in the ring is the same deadlock in
         * a different disguise. */
        if (!u->sqpoll) {
            head = __atomic_load_n(u->sq_head, __ATOMIC_ACQUIRE);
            const unsigned to_submit = tail - head;
            const int in_kernel = posted - (int)to_submit;
            if (to_submit || in_kernel > 0) {
                int r;
                do {
                    r = sys_io_uring_enter(u->fd, to_submit,
                                           in_kernel > 0 ? 1u : 0u,
                                           IORING_ENTER_GETEVENTS);
                } while (r < 0 && errno == EINTR);
                if (r < 0) { failed = 1; break; }
            }
        } else {
            /* The polling thread parks itself after sq_thread_idle and sets NEED_WAKEUP;
             * skipping the syscall then would leave the submissions unnoticed. */
            const unsigned f = __atomic_load_n(u->sq_flags, __ATOMIC_ACQUIRE);
            if (f & IORING_SQ_NEED_WAKEUP) {
                int r;
                do { r = sys_io_uring_enter(u->fd, 0, 0, IORING_ENTER_SQ_WAKEUP); }
                while (r < 0 && errno == EINTR);
                if (r < 0) { failed = 1; break; }
            }
        }

        /* ---- reap ---- */
        unsigned chead = __atomic_load_n(u->cq_head, __ATOMIC_RELAXED);
        const unsigned ctail = __atomic_load_n(u->cq_tail, __ATOMIC_ACQUIRE);
        if (chead == ctail && posted > 0 && u->sqpoll) continue;  /* spin for SQPOLL */
        while (chead != ctail) {
            const struct io_uring_cqe *cqe = &u->cqes[chead & *u->cq_mask];
            const int64_t idx = (int64_t)cqe->user_data;
            const int res = cqe->res;
            chead++;
            posted--;
            if (res < 0) {
                if (res == -EINTR || res == -EAGAIN) { again[nagain++] = idx; continue; }
                failed = 1;
                continue;
            }
            if (res == 0) { failed = 1; continue; }   /* EOF short of the run */
            ck[idx].done += res;
            got += res;
            if (ck[idx].done < ck[idx].len) again[nagain++] = idx;
        }
        __atomic_store_n(u->cq_head, chead, __ATOMIC_RELEASE);
    }

    free(ck); free(again);
    if (failed) return -1;
    return got;
}

#else  /* not Linux: io_uring does not exist, and the caller falls back to pread */

struct K3Uring { int unused; };
K3Uring *k3_uring_new(unsigned entries) { (void)entries; return NULL; }
void     k3_uring_free(K3Uring *u) { (void)u; }
unsigned k3_uring_depth(const K3Uring *u) { (void)u; return 0u; }
int      k3_uring_sqpoll(const K3Uring *u) { (void)u; return 0; }
int64_t  k3_uring_read(K3Uring *u, int fd, void *buf, int64_t nbytes, int64_t off)
{ (void)u; (void)fd; (void)buf; (void)nbytes; (void)off; return -1; }

#endif
