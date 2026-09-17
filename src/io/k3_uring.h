/* k3_uring.h - queue depth for the trunk reader, without a new dependency.
 *
 * THE PROBLEM THIS SOLVES, IN THIS ENGINE'S OWN NUMBERS
 *   The trunk reader issues one pread at a time and waits for it. That is a queue depth
 *   of ONE, and an NVMe device does not reach its rated bandwidth at depth one: it
 *   reaches it by having several requests outstanding so the controller can keep its
 *   channels busy. tools/devbw.py already probes the target device at QD16 for exactly
 *   this reason, and the gap between the two numbers is what this file exists to close.
 *   Measured on the reference machine, trunk reads ran at 5908 MB/s inside a 135.8 s
 *   token while the same device probed materially higher at depth.
 *
 *   The fix is not "more threads". A layer run is ONE contiguous region, so it can be
 *   split into chunks and all the chunks submitted at once from a single thread. That is
 *   what io_uring is for.
 *
 * WHY RAW SYSCALLS RATHER THAN liburing
 *   liburing is a build dependency for something this project can express in about two
 *   hundred lines: three syscalls, two mmaps and a pair of ring indices. The repository
 *   carries exactly one third-party file (a JSON parser) and this is not worth being the
 *   second, especially since it must degrade to pread anyway on macOS, on older kernels,
 *   and inside containers that block io_uring outright.
 *
 * EVERYTHING HERE IS OPTIONAL AND FALLS BACK
 *   k3_uring_new returns NULL if io_uring is unavailable for any reason, and the caller
 *   is required to cope by using pread -- which is always correct, just shallower. No
 *   output bit depends on which path runs: these are reads of the same bytes from the
 *   same offsets into the same buffer.
 *
 * K3_NOURING=1  forces the pread path on a binary that has io_uring, so the two can be
 *               A/B compared without rebuilding.
 * K3_SQPOLL=1   asks the kernel to poll the submission queue from its own thread, which
 *               removes the io_uring_enter syscall from the submit path. It costs a
 *               kernel thread spinning, so it is opt-in rather than default, and setup
 *               falls back to a normal ring if the kernel refuses.
 */
#ifndef K3_URING_H
#define K3_URING_H

#include <stdint.h>

typedef struct K3Uring K3Uring;

/* Create a ring with room for `entries` outstanding requests. Returns NULL when
 * io_uring is unavailable, disabled, or refused by the kernel: that is a normal
 * outcome, not an error, and the caller must fall back to pread. */
K3Uring *k3_uring_new(unsigned entries);
void     k3_uring_free(K3Uring *u);

/* Read exactly nbytes at file offset off into buf, with the transfer split across as
 * many concurrent requests as the ring allows. Returns bytes read, or -1.
 *
 * Short reads are retried at the right offset rather than treated as failure, so this
 * behaves like the pread loop it replaces. With O_DIRECT the caller must still satisfy
 * alignment: this splits on a 4096 multiple so an aligned request stays aligned. */
int64_t  k3_uring_read(K3Uring *u, int fd, void *buf, int64_t nbytes, int64_t off);

/* Requests in flight per read, and whether the kernel accepted SQPOLL. For reporting. */
unsigned k3_uring_depth(const K3Uring *u);
int      k3_uring_sqpoll(const K3Uring *u);

#endif /* K3_URING_H */
