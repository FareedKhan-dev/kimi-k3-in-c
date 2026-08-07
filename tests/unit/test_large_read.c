/* test_large_read.c - regression for Darwin's ~1 GiB pread limit.
 *
 * WHY THIS FILE EXISTS
 *   embed_tokens and lm_head are ~2.35 GiB each; trunk layer 0 is ~2.34 GiB. On macOS a
 *   single pread above ~1 GiB returns EINVAL. k3_st_read chunks on __APPLE__ only; Linux
 *   keeps one pread per iteration. The safetensors fixtures are all smaller than 1 GiB, so
 *   test_st never exercises the path that broke hybrid NAS + local-trunk runs.
 *
 * WHAT IS CHECKED
 *   1 PROBE       on __APPLE__, one pread of 2 GiB fails with EINVAL
 *   2 FULL SPAN   k3_st_read of a sparse file ~2.06 GiB (embed-sized) is byte-exact at
 *                 the head, 1 GiB, 2 GiB, and the tail
 *   3 CROSS 1 GiB k3_st_read whose [off, off+nbytes) straddles the 1 GiB line
 *   4 PROBE       on Linux, one pread of 2 GiB succeeds (the path Darwin rejects)
 *
 * usage: test_large_read [workdir]
 *        workdir defaults to build/large_read (under the current directory)
 */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "k3_st.h"

/* embed_tokens is ~2.35 GiB. Measured on Darwin: a single pread succeeds at 1 GiB and
 * fails with EINVAL by 2 GiB, so the full-span case must be embed-sized or the test
 * would pass even with the broken single-pread loop. */
#define ONEBG   (1024ULL * 1024 * 1024)
#define CHUNK   (64ULL * 1024 * 1024)
#define FSIZE   (2ULL * ONEBG + CHUNK)

static int g_fail = 0;

static void ck(int ok, const char *what, const char *detail)
{
    printf("  %s  %-34s %s\n", ok ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!ok) g_fail++;
}

static int write_marker(int fd, off_t off, const char *tag)
{
    return pwrite(fd, tag, 4, off) == 4 ? 0 : -1;
}

static int make_sparse(const char *path)
{
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return -1;
    if (ftruncate(fd, (off_t)FSIZE) != 0) { close(fd); return -1; }
    if (write_marker(fd, 0, "HEAD") != 0 ||
        write_marker(fd, (off_t)ONEBG, "MID1") != 0 ||
        write_marker(fd, (off_t)(2 * ONEBG), "MID2") != 0 ||
        write_marker(fd, (off_t)(FSIZE - 4), "TAIL") != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static int check_markers(const unsigned char *buf, int64_t base_off, int64_t nbytes)
{
    if (base_off <= 0 && nbytes >= 4 && memcmp(buf, "HEAD", 4) != 0) return 0;
    if (base_off <= (int64_t)ONEBG && base_off + nbytes > (int64_t)ONEBG + 4 &&
        memcmp(buf + ((int64_t)ONEBG - base_off), "MID1", 4) != 0)
        return 0;
    if (base_off <= (int64_t)(2 * ONEBG) && base_off + nbytes > (int64_t)(2 * ONEBG) + 4 &&
        memcmp(buf + ((int64_t)(2 * ONEBG) - base_off), "MID2", 4) != 0)
        return 0;
    if (base_off + nbytes >= (int64_t)FSIZE &&
        memcmp(buf + ((int64_t)FSIZE - 4 - base_off), "TAIL", 4) != 0)
        return 0;
    return 1;
}

#if defined(__APPLE__)
static void probe_darwin_pread_limit(const char *dir)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/probe.bin", dir);
    const int64_t psz = (int64_t)(2ULL * ONEBG);   /* embed is ~2.3 GiB; 2 GiB fails here */
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        ck(0, "Darwin pread probe", "open failed");
        return;
    }
    if (ftruncate(fd, (off_t)psz) != 0) {
        close(fd);
        ck(0, "Darwin pread probe", "ftruncate failed");
        return;
    }
    void *buf = NULL;
    if (posix_memalign(&buf, 4096, (size_t)psz) != 0) {
        close(fd);
        ck(0, "Darwin pread probe", "alloc failed");
        return;
    }
    errno = 0;
    ssize_t r = pread(fd, buf, (size_t)psz, 0);
    const int rejected = (r < 0 && errno == EINVAL);
    free(buf);
    close(fd);
    unlink(path);
    ck(rejected, "Darwin pread probe",
       rejected ? ">2 GiB single pread rejected as expected" : "expected EINVAL, did not get it");
}
#else
static void probe_linux_pread_limit(const char *dir)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/probe.bin", dir);
    const int64_t psz = (int64_t)(2ULL * ONEBG);
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        ck(0, "Linux pread probe", "open failed");
        return;
    }
    if (ftruncate(fd, (off_t)psz) != 0) {
        close(fd);
        ck(0, "Linux pread probe", "ftruncate failed");
        return;
    }
    void *buf = NULL;
    if (posix_memalign(&buf, 4096, (size_t)psz) != 0) {
        close(fd);
        ck(0, "Linux pread probe", "alloc failed");
        return;
    }
    errno = 0;
    ssize_t r = pread(fd, buf, (size_t)psz, 0);
    const int ok = (r == (ssize_t)psz);
    free(buf);
    close(fd);
    unlink(path);
    ck(ok, "Linux pread probe",
       ok ? ">2 GiB single pread succeeded as expected" : "expected full read, pread failed");
}
#endif

static int read_via_k3_st(int fd, int64_t off, int64_t nbytes, unsigned char *buf)
{
    K3St s;
    memset(&s, 0, sizeof s);
    s.nshard = 1;
    s.fd = &fd;

    K3Tensor t;
    memset(&t, 0, sizeof t);
    t.name = "large_read.test";
    t.shard = 0;
    t.off = off;
    t.nbytes = nbytes;

    return k3_st_read(&s, &t, buf) == nbytes;
}

static void test_full_span(const char *path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        ck(0, "k3_st_read full span", "open failed");
        return;
    }
    unsigned char *buf = (unsigned char *)malloc((size_t)FSIZE);
    if (!buf) {
        close(fd);
        ck(0, "k3_st_read full span", "alloc failed");
        return;
    }
    const int ok = read_via_k3_st(fd, 0, (int64_t)FSIZE, buf) &&
                   check_markers(buf, 0, (int64_t)FSIZE);
    close(fd);
    free(buf);
    ck(ok, "k3_st_read full span",
       ok ? "HEAD / MID1 / MID2 / TAIL intact above 2 GiB" : "read or markers wrong");
}

static void test_cross_boundary(const char *path)
{
    const int64_t off = (int64_t)ONEBG - 4096;
    const int64_t nbytes = 8192;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        ck(0, "k3_st_read cross 1 GiB", "open failed");
        return;
    }
    unsigned char *buf = (unsigned char *)malloc((size_t)nbytes);
    if (!buf) {
        close(fd);
        ck(0, "k3_st_read cross 1 GiB", "alloc failed");
        return;
    }
    const int ok = read_via_k3_st(fd, off, nbytes, buf) &&
                   check_markers(buf, off, nbytes);
    close(fd);
    free(buf);
    ck(ok, "k3_st_read cross 1 GiB",
       ok ? "straddling read intact" : "read or markers wrong");
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "build/large_read";
    char path[1024];
    snprintf(path, sizeof path, "%s/sparse.bin", dir);

    if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "cannot create %s\n", dir);
        return 1;
    }

    printf("large read regression (%s)\n", dir);
    printf("  file size : %.2f GiB (sparse)\n\n", (double)FSIZE / (1024.0 * 1024 * 1024));

    int wfd = make_sparse(path);
    if (wfd < 0) {
        fprintf(stderr, "cannot create %s\n", path);
        return 1;
    }
    close(wfd);

#if defined(__APPLE__)
    probe_darwin_pread_limit(dir);
#else
    probe_linux_pread_limit(dir);
#endif
    test_full_span(path);
    test_cross_boundary(path);

    printf("\n%s\n", g_fail ? "LARGE READ TESTS FAILED" : "LARGE READ TESTS PASSED");
    return g_fail ? 1 : 0;
}
