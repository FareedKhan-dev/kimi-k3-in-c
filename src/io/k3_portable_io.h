/* k3_portable_io.h - shims for the Linux-only I/O calls the readers use.
 *
 * The engine asks for two things Linux gives it and Darwin does not spell the same way:
 *
 *   O_DIRECT       bypass the page cache on the trunk and expert reads. Darwin's
 *                  equivalent is not an open() flag but fcntl(F_NOCACHE) after the
 *                  fact, so O_DIRECT is defined to 0 here (open() is unaffected) and
 *                  k3_set_direct() applies the real thing to the returned descriptor.
 *
 *   posix_fadvise  a page-cache prefetch hint with no Darwin equivalent. Callers
 *                  already treat it as advisory -- the one call site returns early on
 *                  the direct path because the hint has nothing to populate there -- so
 *                  the shim is a no-op that keeps the buffered path compiling.
 *
 * Both call sites fall back to buffered reads when the direct path is unavailable, so
 * neither shim changes what the engine computes, only how fast it reads.
 */
#ifndef K3_PORTABLE_IO_H
#define K3_PORTABLE_IO_H

/* The readers define _POSIX_C_SOURCE, which hides Darwin's non-standard fcntl commands
 * (F_NOCACHE among them) from <fcntl.h>. _DARWIN_C_SOURCE puts them back. It must be
 * set before the first libc header is pulled in, so this header is included first. */
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif

#include <fcntl.h>

#if defined(__APPLE__)

/* Darwin rejects a single pread above ~1 GiB with EINVAL. embed_tokens, lm_head and
 * trunk layer 0 are ~2.3 GiB each; k3_st_read and k3_trunk cap each syscall. */
#define K3_IO_READ_CHUNK (64 << 20)

/* Not an open() flag on Darwin: defining it to 0 leaves open() semantics untouched. */
#ifndef O_DIRECT
#define O_DIRECT 0
#endif

#ifndef POSIX_FADV_WILLNEED
#define POSIX_FADV_WILLNEED 3
#endif

static inline int posix_fadvise(int fd, off_t off, off_t len, int advice)
{
    (void)fd; (void)off; (void)len; (void)advice;
    return 0;   /* advisory only; the buffered path is correct without it */
}

/* Darwin's O_DIRECT equivalent, applied after open(). Failure is not fatal: the caller
 * keeps the descriptor and reads through the page cache instead. */
static inline int k3_set_direct(int fd)
{
    if (fd < 0) return -1;
    return fcntl(fd, F_NOCACHE, 1);
}

#else   /* Linux and friends: O_DIRECT on open() already did it */

static inline int k3_set_direct(int fd) { (void)fd; return 0; }

#endif

#endif /* K3_PORTABLE_IO_H */
