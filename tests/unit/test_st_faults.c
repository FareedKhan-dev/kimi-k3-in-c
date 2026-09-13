/* test_st_faults.c - partial and corrupt shards must fail loudly.
 *
 * WHY THIS FILE EXISTS
 *   The download script warns that a partial checkpoint "does not fail loudly,
 *   it produces wrong tokens". The reader is the last line of defense, and the
 *   happy-path test (test_st.c) never feeds it a broken file. Each case below
 *   builds a mutant of the real fixture shards and asserts the failure is
 *   LOUD: open refuses, or a read comes back short. What is NOT asserted is
 *   content corruption with intact structure (flipped data bytes, lying but
 *   in-range offsets): those read back full and silent, which is exactly why
 *   trunk manifests exist -- this test documents the boundary of what the
 *   reader itself can catch.
 *
 * usage: test_st_faults <fixture_st_dir> <work_dir>
 *
 * author F E R M I ∞ H A R T <contact@fermihart.com>
 * SPDX-License-Identifier: Apache-2.0
 */
#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>

#include "k3_st.h"

static int g_fail = 0;

static void ck(int ok, const char *what, const char *detail)
{
    printf("  %s  %-34s %s\n", ok ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!ok) g_fail++;
}

static int endswith(const char *s, const char *suf)
{
    size_t a = strlen(s), b = strlen(suf);
    return a >= b && strcmp(s + a - b, suf) == 0;
}

static unsigned char *slurp(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    unsigned char *b = (unsigned char *)malloc((size_t)sz ? (size_t)sz : 1);
    if (!b) { fclose(f); return NULL; }
    if (sz && fread(b, 1, (size_t)sz, f) != (size_t)sz) {
        free(b); fclose(f); return NULL;
    }
    fclose(f);
    *n = (size_t)sz;
    return b;
}

static int writeall(const char *path, const unsigned char *b, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    const int ok = n == 0 || fwrite(b, 1, n, f) == n;
    fclose(f);
    return ok ? 0 : -1;
}

/* Copy every .safetensors from src to dst, then mutate file `which` with fn.
 * Mutations are small on purpose: each must break exactly one contract. */
static int make_case(const char *src, const char *dst, const char *mutname,
                     size_t keep)
{
    /* mkdir, not system(): the path is ours and the shell adds nothing. */
    mkdir(dst, 0755);
    DIR *d = opendir(src);
    if (!d) return -1;
    struct dirent *e;
    int found = 0;
    while ((e = readdir(d)) != NULL) {
        if (!endswith(e->d_name, ".safetensors")) continue;
        char sp[1024], dp[1024];
        snprintf(sp, sizeof sp, "%s/%s", src, e->d_name);
        snprintf(dp, sizeof dp, "%s/%s", dst, e->d_name);
        size_t n = 0;
        unsigned char *b = slurp(sp, &n);
        if (!b) { closedir(d); return -1; }
        if (mutname && strcmp(e->d_name, mutname) == 0) {
            if (keep < n) {
                unsigned char *t = (unsigned char *)malloc(keep ? keep : 1);
                if (!t) { free(b); closedir(d); return -1; }
                memcpy(t, b, keep);
                free(b);
                b = t;
                n = keep;
            }
            found = 1;
        }
        const int rc = writeall(dp, b, n);
        free(b);
        if (rc != 0) { closedir(d); return -1; }
    }
    closedir(d);
    return found || !mutname ? 0 : -1;
}

static uint64_t rdle64(const unsigned char *b)
{
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | b[i];
    return v;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: test_st_faults <fixture_st_dir> <work_dir>\n");
        return 2;
    }
    const char *fix = argv[1];
    const char *work = argv[2];
    mkdir(work, 0755);   /* case dirs nest one level below; make_case makes those */

    char first[1024] = "";
    {
        /* First shard name, for targeted single-file mutations. */
        DIR *d = opendir(fix);
        if (!d) { fprintf(stderr, "cannot open %s\n", fix); return 2; }
        struct dirent *e;
        while ((e = readdir(d)) != NULL)
            if (endswith(e->d_name, ".safetensors")) {
                snprintf(first, sizeof first, "%s", e->d_name);
                break;
            }
        closedir(d);
        if (!first[0]) { fprintf(stderr, "no shards in %s\n", fix); return 2; }
    }

    /* Control: an unmutated copy opens. Without this, every refusal below
     * could be the harness failing instead of the reader refusing. */
    {
        char d[1024];
        snprintf(d, sizeof d, "%s/ctl", work);
        K3St s;
        int ok = make_case(fix, d, NULL, 0) == 0 && k3_st_open(&s, d) == 0 && s.nt > 0;
        ck(ok, "control unmutated copy opens", ok ? "" : "harness or reader broken");
        if (ok) k3_st_close(&s);
    }

    /* T1: file cut to 4 bytes (shorter than the 8-byte header length). */
    {
        char d[1024];
        snprintf(d, sizeof d, "%s/t1", work);
        K3St s;
        memset(&s, 0, sizeof s);
        const int built = make_case(fix, d, first, 4) == 0;
        const int rc = built ? k3_st_open(&s, d) : 0;
        ck(built && rc != 0, "4-byte file refused at open", "");
        if (rc == 0) k3_st_close(&s);
    }

    /* T2: header length overwritten with all-ones (impossible size). */
    {
        char d[1024], p[2048];
        snprintf(d, sizeof d, "%s/t2", work);
        K3St s;
        memset(&s, 0, sizeof s);
        int ok = make_case(fix, d, NULL, 0) == 0;
        snprintf(p, sizeof p, "%s/%s", d, first);
        FILE *f = ok ? fopen(p, "r+b") : NULL;
        if (f) {
            static const unsigned char ff[8] = {255, 255, 255, 255, 255, 255, 255, 255};
            ok = fwrite(ff, 1, 8, f) == 8;
            fclose(f);
        } else {
            ok = 0;
        }
        const int rc = ok ? k3_st_open(&s, d) : 0;
        ck(ok && rc != 0, "impossible header length refused", "");
        if (rc == 0 && ok) k3_st_close(&s);
    }

    /* T3: structural JSON byte flipped ('{' at offset 8 -> '['). */
    {
        char d[1024], p[2048];
        snprintf(d, sizeof d, "%s/t3", work);
        K3St s;
        memset(&s, 0, sizeof s);
        int ok = make_case(fix, d, NULL, 0) == 0;
        snprintf(p, sizeof p, "%s/%s", d, first);
        FILE *f = ok ? fopen(p, "r+b") : NULL;
        if (f) {
            fseek(f, 8, SEEK_SET);
            ok = fputc('[', f) == '[';
            fclose(f);
        } else {
            ok = 0;
        }
        const int rc = ok ? k3_st_open(&s, d) : 0;
        ck(ok && rc != 0, "corrupt header JSON refused", "");
        if (rc == 0 && ok) k3_st_close(&s);
    }

    /* T4: valid header, zero data bytes. Open may succeed (lazy index), but a
     * read must come back SHORT -- never full counts over missing bytes. */
    {
        char d[1024], p[2048];
        snprintf(d, sizeof d, "%s/t4", work);
        K3St s;
        memset(&s, 0, sizeof s);
        int ok = make_case(fix, d, NULL, 0) == 0;
        snprintf(p, sizeof p, "%s/%s", d, first);
        size_t n = 0;
        unsigned char *b = ok ? slurp(p, &n) : NULL;
        if (b && n >= 8) {
            const uint64_t hlen = rdle64(b);
            if (8 + hlen <= n && hlen > 0 && hlen < (1u << 20)) {
                ok = writeall(p, b, (size_t)(8 + hlen)) == 0;
            } else {
                ok = 0;
            }
        } else {
            ok = 0;
        }
        free(b);
        const int rc = ok ? k3_st_open(&s, d) : -99;
        int shortread = 0;
        if (ok && rc == 0 && s.nt > 0) {
            /* First tensor of the truncated shard: find any tensor whose
             * advertised extent is non-empty and read it. */
            for (int i = 0; i < s.nt; i++) {
                if (s.t[i].nbytes <= 0) continue;
                unsigned char *buf = (unsigned char *)malloc((size_t)s.t[i].nbytes);
                if (!buf) break;
                const int64_t got = k3_st_read(&s, &s.t[i], buf);
                free(buf);
                if (got != s.t[i].nbytes) shortread = 1;
                break;
            }
            k3_st_close(&s);
        }
        /* Either refusal at open or a short read: both are loud. Full counts
         * over a truncated file would be silent corruption. */
        ck(ok && (rc != 0 || shortread), "tail-truncated data surfaces loudly", "");
    }

    printf("\n%s\n", g_fail ? "ST-FAULT TESTS FAILED" : "ST-FAULT TESTS PASSED");
    return g_fail ? 1 : 0;
}
