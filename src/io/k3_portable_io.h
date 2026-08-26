/* k3_portable_io.h - shims for the Linux-only I/O calls the readers use.
 *
 * The engine asks for things Linux gives it and the other two platforms do not spell
 * the same way:
 *
 *   O_DIRECT       bypass the page cache on the trunk and expert reads. Darwin's
 *                  equivalent is not an open() flag but fcntl(F_NOCACHE) after the
 *                  fact, so O_DIRECT is defined to 0 there (open() is unaffected) and
 *                  k3_set_direct() applies the real thing to the returned descriptor.
 *                  Windows is the opposite of Darwin: unbuffered I/O (FILE_FLAG_NO_
 *                  BUFFERING) can ONLY be set at CreateFile time, same constraint as
 *                  Linux's real O_DIRECT, but nothing in the MinGW runtime maps an
 *                  open() flag onto it -- so open() itself is intercepted below.
 *
 *   posix_fadvise  a page-cache prefetch hint with no Darwin or Windows equivalent.
 *                  Callers already treat it as advisory -- the one call site returns
 *                  early on the direct path because the hint has nothing to populate
 *                  there -- so the shim is a no-op that keeps the buffered path
 *                  compiling.
 *
 *   pread          positioned read. Native on Linux and Darwin. The MinGW runtime has
 *                  no equivalent, so Windows gets one built on ReadFile's OVERLAPPED
 *                  Offset/OffsetHigh fields -- a true positioned read that does not
 *                  touch a shared file-pointer, unlike SetFilePointerEx + ReadFile,
 *                  which would race when the trunk reader thread and the expert-cache
 *                  prefetch threads pread() the same fd concurrently.
 *
 *   posix_memalign Native on Linux and Darwin. Windows gets a thin wrapper over
 *                  _aligned_malloc, which takes its (size, align) arguments in the
 *                  opposite order.
 *
 * All four call sites fall back to buffered reads (or, for pread/posix_memalign, have
 * no fallback because the shim IS the implementation) when the direct path is
 * unavailable, so none of this changes what the engine computes, only how fast it
 * reads -- except on Windows, where pread and posix_memalign are load-bearing rather
 * than a speed path, since the codebase has no buffered-only alternative to either.
 *
 * A fifth difference is not an open() flag at all: pread() itself returns EINVAL on
 * Darwin for a single request of 2^31 bytes or more, where Linux either succeeds or
 * returns a short read that a retry loop already handles. The embedding table (2.35 GB)
 * and a packed trunk layer (up to 2.37 GB) both exceed that in one contiguous span.
 * K3_PREAD_MAX is the per-syscall cap every such read loop chunks against; on platforms
 * without the limit it just turns one syscall into a few, which costs nothing measurable
 * against a multi-gigabyte transfer.
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
#include <stdint.h>

#define K3_PREAD_MAX ((int64_t)1 << 30)   /* 1 GiB per syscall; see file header */

#if defined(__APPLE__)

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

#elif defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#ifndef O_DIRECT
#define O_DIRECT 0x40000000   /* sentinel bit; intercepted by k3_win_open() below */
#endif

#ifndef POSIX_FADV_WILLNEED
#define POSIX_FADV_WILLNEED 3
#endif

static inline int posix_fadvise(int fd, long long off, long long len, int advice)
{
    (void)fd; (void)off; (void)len; (void)advice;
    return 0;   /* advisory only; no Windows equivalent */
}

/* True positioned read: OVERLAPPED with only Offset/OffsetHigh set, on a handle NOT
 * opened with FILE_FLAG_OVERLAPPED, completes synchronously and does not touch any
 * shared file-pointer state -- safe for concurrent callers on the same fd. */
static inline long long k3_pread(int fd, void *buf, size_t count, long long offset)
{
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }

    OVERLAPPED ov;
    memset(&ov, 0, sizeof ov);
    ov.Offset = (DWORD)(offset & 0xFFFFFFFFu);
    ov.OffsetHigh = (DWORD)((unsigned long long)offset >> 32);

    DWORD got = 0;
    if (!ReadFile(h, buf, (DWORD)count, &got, &ov)) {
        if (GetLastError() == ERROR_HANDLE_EOF) return 0;
        errno = EIO;
        return -1;
    }
    return (long long)got;
}
#define pread(fd, buf, count, offset) k3_pread((fd), (buf), (count), (offset))

/* O_DIRECT has to be intercepted at open() itself: unlike Darwin's post-hoc fcntl,
 * Windows has no way to add FILE_FLAG_NO_BUFFERING to an already-open handle. The
 * bridge back to a plain int fd -- so every other pread()/close() call site in the
 * codebase stays untouched -- goes through _open_osfhandle.
 *
 * Every reader call site in this codebase passes O_RDONLY, so the access mode below
 * matters only to a caller outside the streaming path, such as a test that opens a
 * fixture file for writing: CreateFileA needs GENERIC_WRITE for that, or a later
 * ftruncate/write on the resulting handle fails with access denied even though the
 * open() call itself succeeded. */
static inline int k3_win_open(const char *path, int flags, ...)
{
    DWORD fileFlags = FILE_ATTRIBUTE_NORMAL;
    if (flags & O_DIRECT) fileFlags |= FILE_FLAG_NO_BUFFERING;

    DWORD access = GENERIC_READ;
    int crtFlags = _O_RDONLY;
    if ((flags & (O_WRONLY | O_RDWR)) == O_WRONLY) {
        access = GENERIC_WRITE;
        crtFlags = _O_WRONLY;
    } else if (flags & O_RDWR) {
        access = GENERIC_READ | GENERIC_WRITE;
        crtFlags = _O_RDWR;
    }

    HANDLE h = CreateFileA(path, access,
                            FILE_SHARE_READ | FILE_SHARE_WRITE,
                            NULL, OPEN_EXISTING, fileFlags, NULL);
    if (h == INVALID_HANDLE_VALUE) { errno = ENOENT; return -1; }

    int fd = _open_osfhandle((intptr_t)h, crtFlags | _O_BINARY);
    if (fd < 0) { CloseHandle(h); errno = EMFILE; return -1; }
    return fd;
}
#define open(path, flags, ...) k3_win_open((path), (flags))

/* GCC/Clang recognise the name posix_memalign as a built-in for optimisation purposes
 * (constant folding, alias analysis) even though MinGW's headers declare no such
 * function and its runtime provides no such symbol -- confirmed directly: a call to
 * it with no other declaration in scope fails with "implicit declaration", not a
 * link error, meaning nothing backs the built-in's assumed semantics. This
 * definition is therefore not optional the way the -Wshadow warning below implies;
 * it is the only real implementation on this platform. The pragma silences the
 * warning without silencing -Wshadow project-wide. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
static inline int posix_memalign(void **out, size_t align, size_t len)
{
    *out = _aligned_malloc(len, align);
    return *out ? 0 : ENOMEM;
}
#pragma GCC diagnostic pop

/* On real POSIX, a posix_memalign'd pointer is safe to pass to plain free() -- that is
 * the entire point of the interface. _aligned_malloc has no such guarantee: pairing it
 * with free() instead of _aligned_free() corrupts the heap (observed directly: Windows
 * terminates the process with STATUS_HEAP_CORRUPTION). k3_aligned_free() exists so the
 * two call sites that free a posix_memalign'd arena (k3_cache.c, k3_trunk.c) can free it
 * correctly on every platform without special-casing Windows at the call site itself. */
#define k3_aligned_free(p) _aligned_free(p)

/* madvise(MADV_HUGEPAGE) is a Linux transparent-hugepage hint; every caller already
 * treats it as advisory ("failure is not an error" -- k3_trunk.c), so a no-op is the
 * correct port here, not a functional gap. The alternative, VirtualAlloc with
 * MEM_LARGE_PAGES, needs SeLockMemoryPrivilege and allocations sized to an exact
 * large-page multiple -- a real feature to build for a hint the code already
 * tolerates losing. */
#define madvise(addr, len, advice) ((void)0)
#define MADV_HUGEPAGE 0

/* O_DIRECT already forced open()-time behavior above; nothing left to do post-open. */
static inline int k3_set_direct(int fd) { (void)fd; return 0; }

#else   /* Linux and friends: O_DIRECT on open() already did it */

static inline int k3_set_direct(int fd) { (void)fd; return 0; }

#endif

/* Linux and Darwin: posix_memalign's contract already makes plain free() safe. */
#ifndef k3_aligned_free
#define k3_aligned_free(p) free(p)
#endif

#endif /* K3_PORTABLE_IO_H */
