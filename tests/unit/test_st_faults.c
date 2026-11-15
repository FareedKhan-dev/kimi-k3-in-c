/* test_st_faults.c - partial and corrupt shards must fail loudly.
 *
 * WHY THIS FILE EXISTS
 *   The download script warns that a partial checkpoint "does not fail loudly,
 *   it produces wrong tokens". The reader is the last line of defense, and the
 *   happy-path test (test_st.c) never feeds it a broken file. Each case below
 *   builds a mutant of the real fixture shards and asserts the failure is
 *   LOUD: open refuses, or a read comes back short. What is NOT asserted is
 *   content corruption with intact structure, flipped data bytes, or lying
 *   but in-range offsets: those read back full and silent, so this test
 *   documents the boundary of what the reader itself can catch rather than
 *   claiming to catch everything. One content fault IS caught, by the expert
 *   loader rather than the reader: an E8M0 scale byte of 255, the NaN code no
 *   quantiser writes. The last case below builds a one-expert shard with
 *   exactly one such byte and asserts the cache refuses it by name.
 *
 * usage: test_st_faults <fixture_st_dir> <work_dir>
 */
#define _POSIX_C_SOURCE 200809L

#include "k3_portable_io.h"   /* mkdir() shim for MinGW; see the header for why */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>

#include "k3.h"
#include "k3_cache.h"
#include "k3_load.h"
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
        /* Checked, not just silenced: GCC cannot bound dst/src at compile time and
         * warns on that alone, but a genuinely truncated path here would make
         * writeall() below write the mutated shard under the wrong name, which
         * this file exists to catch elsewhere, not commit itself. */
        if (snprintf(sp, sizeof sp, "%s/%s", src, e->d_name) >= (int)sizeof sp ||
            snprintf(dp, sizeof dp, "%s/%s", dst, e->d_name) >= (int)sizeof dp) {
            closedir(d); return -1;
        }
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

/* Copy every .safetensors from src to dst EXCEPT `skip` (which must exist): the
 * incomplete-download shape, one shard never arrived. */
static int copy_shards(const char *src, const char *dst, const char *skip)
{
    mkdir(dst, 0755);
    DIR *d = opendir(src);
    if (!d) return -1;
    struct dirent *e;
    int skipped = 0;
    while ((e = readdir(d)) != NULL) {
        if (!endswith(e->d_name, ".safetensors")) continue;
        if (skip && strcmp(e->d_name, skip) == 0) { skipped = 1; continue; }
        char sp[1024], dp[1024];
        if (snprintf(sp, sizeof sp, "%s/%s", src, e->d_name) >= (int)sizeof sp ||
            snprintf(dp, sizeof dp, "%s/%s", dst, e->d_name) >= (int)sizeof dp) {
            closedir(d); return -1;
        }
        size_t n = 0;
        unsigned char *b = slurp(sp, &n);
        if (!b) { closedir(d); return -1; }
        const int rc = writeall(dp, b, n);
        free(b);
        if (rc != 0) { closedir(d); return -1; }
    }
    closedir(d);
    return (skip && !skipped) ? -1 : 0;
}

static int write_text(const char *dir, const char *file, const char *text)
{
    char p[2048];
    if (snprintf(p, sizeof p, "%s/%s", dir, file) >= (int)sizeof p) return -1;
    return writeall(p, (const unsigned char *)text, strlen(text));
}

static uint64_t rdle64(const unsigned char *b)
{
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | b[i];
    return v;
}

/* A shard built from scratch: [8-byte LE header length][json][data]. The cases
 * below need headers whose data_offsets the fixture generator would never
 * write (it lays tensors out back to back), so they are spelled out by hand. */
static int write_synthetic_shard(const char *path, const char *header_json,
                                 const unsigned char *data, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    const uint64_t hlen = (uint64_t)strlen(header_json);
    unsigned char le[8];
    for (int i = 0; i < 8; i++) le[i] = (unsigned char)(hlen >> (8 * i));
    int ok = fwrite(le, 1, 8, f) == 8
          && fwrite(header_json, 1, (size_t)hlen, f) == (size_t)hlen
          && (n == 0 || fwrite(data, 1, n, f) == n);
    fclose(f);
    return ok ? 0 : -1;
}

/* One synthetic shard alone in its own directory; returns k3_st_open's rc and
 * the tensor count through *nt (only meaningful when rc == 0). */
static int open_synthetic(const char *work, const char *sub, const char *header_json,
                          size_t data_bytes, int *nt)
{
    char d[1024], p[2048];
    unsigned char *data = (unsigned char *)calloc(data_bytes ? data_bytes : 1, 1);
    if (!data) return -99;
    for (size_t i = 0; i < data_bytes; i++) data[i] = (unsigned char)i;
    snprintf(d, sizeof d, "%s/%s", work, sub);
    mkdir(d, 0755);
    snprintf(p, sizeof p, "%s/model-00001-of-00001.safetensors", d);
    const int wrc = write_synthetic_shard(p, header_json, data, data_bytes);
    free(data);
    if (wrc != 0) return -99;
    K3St s;
    memset(&s, 0, sizeof s);
    const int rc = k3_st_open(&s, d);
    if (rc == 0) { *nt = s.nt; k3_st_close(&s); }
    return rc;
}

/* A one-expert shard in the real six-tensor layout, tiny: [2][64] packed and [2][4]
 * scales per matrix. Data bytes cycle mod 251, so 255 never occurs by accident; when
 * nan_at >= 0 that one byte is overwritten with 255. */
#define EXP_ROWS 2
#define EXP_PCOLS 64
#define EXP_SCOLS (EXP_PCOLS * 2 / 32)
#define EXP_P_BYTES (EXP_ROWS * EXP_PCOLS)
#define EXP_S_BYTES (EXP_ROWS * EXP_SCOLS)
#define EXP_BYTES (3 * (EXP_P_BYTES + EXP_S_BYTES))
#define EXP_FMT "language_model.model.layers.0.block_sparse_moe.experts.0.%s.%s"

static int write_expert_shard(const char *dir, int nan_at)
{
    static const char *W[3] = { "w1", "w2", "w3" };
    char hdr[2048], p[2048];
    size_t h = 0;
    int64_t off = 0;
    h += (size_t)snprintf(hdr + h, sizeof hdr - h, "{");
    for (int i = 0; i < 3; i++) {
        h += (size_t)snprintf(hdr + h, sizeof hdr - h,
                 "\"" EXP_FMT "\":{\"dtype\":\"U8\",\"shape\":[%d,%d],"
                 "\"data_offsets\":[%lld,%lld]},",
                 W[i], "weight_packed", EXP_ROWS, EXP_PCOLS,
                 (long long)off, (long long)(off + EXP_P_BYTES));
        off += EXP_P_BYTES;
        h += (size_t)snprintf(hdr + h, sizeof hdr - h,
                 "\"" EXP_FMT "\":{\"dtype\":\"U8\",\"shape\":[%d,%d],"
                 "\"data_offsets\":[%lld,%lld]}%s",
                 W[i], "weight_scale", EXP_ROWS, EXP_SCOLS,
                 (long long)off, (long long)(off + EXP_S_BYTES), i < 2 ? "," : "}");
        off += EXP_S_BYTES;
    }
    if (h >= sizeof hdr) return -1;
    while (h % 8) hdr[h++] = ' ';             /* 8-byte aligned body, as the writers do */
    hdr[h] = 0;

    unsigned char data[EXP_BYTES];
    for (int i = 0; i < EXP_BYTES; i++) data[i] = (unsigned char)(i % 251);
    if (nan_at >= 0 && nan_at < EXP_BYTES) data[nan_at] = 255;

    mkdir(dir, 0755);
    if (snprintf(p, sizeof p, "%s/model-00001-of-00001.safetensors", dir) >= (int)sizeof p)
        return -1;
    return write_synthetic_shard(p, hdr, data, sizeof data);
}

/* Open the shard, read the expert both straight and through the cache (batch
 * prefetch then get, and a fresh cache's serial get), and report what the scan said.
 * n255 receives the count the scan returned, why[] its message, serial and batch the
 * get() results (0 = served). Returns 0 when the harness itself worked. */
static int probe_expert_shard(const char *dir, long long *n255, char *why, size_t cap,
                              int *serial, int *batch)
{
    K3St s; memset(&s, 0, sizeof s);
    if (k3_st_open(&s, dir) != 0) return -1;
    K3ExpertRef r; memset(&r, 0, sizeof r);
    if (k3_expert_ref(&s, 0, 0, &r) != 0) { k3_st_close(&s); return -1; }
    unsigned char *buf = (unsigned char *)malloc((size_t)r.nbytes);
    if (!buf || k3_expert_load(&s, &r, buf) != r.nbytes) { free(buf); k3_st_close(&s); return -1; }
    why[0] = 0;
    *n255 = (long long)k3_expert_nan_scales(&s, &r, buf, why, cap);
    free(buf);

    K3Cfg c; memset(&c, 0, sizeof c);
    c.n_layers = 1; c.n_experts = 1; c.topk = 1;
    *serial = *batch = -99;
    for (int pass = 0; pass < 2; pass++) {
        K3Cache cache;
        int inited = 0;
        /* Slots are rounded up to the O_DIRECT window, so ask rather than compute. */
        for (int64_t budget = 16 * 1024; budget <= (int64_t)64 << 20; budget *= 2)
            if (k3_cache_init(&cache, &s, &c, budget) == 0) { inited = 1; break; }
        if (!inited) { k3_st_close(&s); return -1; }
        K3ExpertQ q;
        if (pass == 0) {
            const int one = 0;
            /* K3_NOPREFETCH leaves getmany NULL; then this pass is the serial one too. */
            if (cache.src.getmany) cache.src.getmany(&cache.src, 0, &one, 1);
            *batch = cache.src.get(&cache.src, 0, 0, &q);
        } else {
            *serial = cache.src.get(&cache.src, 0, 0, &q);
        }
        k3_cache_free(&cache);
    }
    k3_st_close(&s);
    return 0;
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

    /* T5-T8: the data region must be tiled exactly. The reference implementation
     * sorts entries by data_offsets and requires each start to equal the previous
     * end, from 0. Two tensors claiming one span (T5) means one of them reads the
     * other's bytes and the shape/size check on each passes on its own; a span no
     * tensor claims (T6, T7) means the header and the file disagree about where the
     * data is. Both read back full and silent without this sweep. T8 is the
     * control: a contiguous layout with a zero-length tensor in the middle, which
     * the fixture set (empty.f32) says is legal. */
    {
        int nt = 0, rc;
        rc = open_synthetic(work, "t5",
            "{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]},"
             "\"b\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[8,24]}}", 24, &nt);
        ck(rc != 0 && rc != -99, "overlapping spans refused", rc == -99 ? "harness" : "");

        rc = open_synthetic(work, "t6",
            "{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]},"
             "\"b\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[24,40]}}", 40, &nt);
        ck(rc != 0 && rc != -99, "gap between spans refused", rc == -99 ? "harness" : "");

        rc = open_synthetic(work, "t7",
            "{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[8,24]}}", 24, &nt);
        ck(rc != 0 && rc != -99, "first span not at offset 0 refused", rc == -99 ? "harness" : "");

        rc = open_synthetic(work, "t8",
            "{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]},"
             "\"z\":{\"dtype\":\"F32\",\"shape\":[0,8],\"data_offsets\":[16,16]},"
             "\"b\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[16,32]}}", 32, &nt);
        ck(rc == 0 && nt == 3, "contiguous + zero-length tensor opens",
           rc == -99 ? "harness" : (rc != 0 ? "refused" : (nt != 3 ? "wrong tensor count" : "")));
    }

    /* ---- missing-tensor triage: a miss must say WHICH FILE, not just which name ----
     * The fixture is model-0000{1,2}-of-00002, so the filenames declare two shards.
     * Removing the second is exactly the incomplete download the download script warns
     * "does not fail loudly". The reader must still open the subset, count 1 of 2, and
     * explain the first miss by naming the absent file; with an index it must name the
     * file the index maps the tensor to, and tell a never-existed name apart from an
     * absent shard. */
    static const char SHARD2[] = "model-00002-of-00002.safetensors";
    static const char SHARD1[] = "model-00001-of-00002.safetensors";

    /* M5: one shard absent, no index: the filenames name the file. */
    {
        char d[1024], why[1024] = "";
        snprintf(d, sizeof d, "%s/m5", work);
        K3St s; memset(&s, 0, sizeof s);
        const int ok = copy_shards(fix, d, SHARD2) == 0 && k3_st_open(&s, d) == 0;
        const int absent = ok && k3_st_find(&s, "second.shard.f32") == NULL;
        const int kind = ok ? (int)k3_st_explain_missing(&s, "second.shard.f32", why, sizeof why) : -1;
        ck(ok && s.declared == 2 && s.numbered == 1,
           "1 of 2 declared shards opens, counted", "");
        ck(absent && kind == K3_ST_MISS_SHARD_ABSENT && strstr(why, SHARD2) != NULL
           && strstr(why, "declare 2 shards and 1 are here") != NULL,
           "miss names the absent shard from the filenames", why);
        ck(ok && k3_st_find(&s, "plain.f32.2d") != NULL,
           "tensors of the present shard still resolve", "");
        if (ok) k3_st_close(&s);
    }

    /* M6: one shard absent, and an index that maps the name to it. */
    {
        char d[1024], why[1024] = "";
        snprintf(d, sizeof d, "%s/m6", work);
        K3St s; memset(&s, 0, sizeof s);
        const int ok = copy_shards(fix, d, SHARD2) == 0
            && write_text(d, "model.safetensors.index.json",
                          "{\"metadata\":{\"total_size\":1},\"weight_map\":{"
                          "\"second.shard.f32\":\"model-00002-of-00002.safetensors\","
                          "\"plain.f32.2d\":\"model-00001-of-00002.safetensors\"}}\n") == 0
            && k3_st_open(&s, d) == 0;
        const int kind = ok ? (int)k3_st_explain_missing(&s, "second.shard.f32", why, sizeof why) : -1;
        ck(ok && kind == K3_ST_MISS_SHARD_ABSENT
           && strstr(why, "index.json puts it in model-00002-of-00002.safetensors") != NULL,
           "index names the absent shard for the miss", why);
        if (ok) k3_st_close(&s);
    }

    /* M7: every shard present; the index maps a name to a shard that lacks it. */
    {
        char d[1024], why[1024] = "";
        snprintf(d, sizeof d, "%s/m7", work);
        K3St s; memset(&s, 0, sizeof s);
        const int ok = copy_shards(fix, d, NULL) == 0
            && write_text(d, "model.safetensors.index.json",
                          "{\"weight_map\":{\"ghost.tensor\":\"model-00001-of-00002.safetensors\"}}") == 0
            && k3_st_open(&s, d) == 0;
        const int kind = ok ? (int)k3_st_explain_missing(&s, "ghost.tensor", why, sizeof why) : -1;
        ck(ok && kind == K3_ST_MISS_SHARD_MISMATCH && strstr(why, SHARD1) != NULL
           && strstr(why, "does not declare it") != NULL,
           "index names a present shard that lacks the tensor", why);
        if (ok) k3_st_close(&s);
    }

    /* M8: every shard present; the index does not list the name at all. */
    {
        char d[1024], why[1024] = "";
        snprintf(d, sizeof d, "%s/m8", work);
        K3St s; memset(&s, 0, sizeof s);
        const int ok = copy_shards(fix, d, NULL) == 0
            && write_text(d, "model.safetensors.index.json",
                          "{\"weight_map\":{\"plain.f32.2d\":\"model-00001-of-00002.safetensors\"}}") == 0
            && k3_st_open(&s, d) == 0;
        const int kind = ok ? (int)k3_st_explain_missing(&s, "ghost.tensor", why, sizeof why) : -1;
        ck(ok && kind == K3_ST_MISS_NOT_IN_CHECKPOINT && strstr(why, "does not list it") != NULL,
           "index says the checkpoint never had the name", why);
        if (ok) k3_st_close(&s);
    }

    /* M9: every shard present, no index: complete by the filenames, so never existed. */
    {
        char d[1024], why[1024] = "";
        snprintf(d, sizeof d, "%s/m9", work);
        K3St s; memset(&s, 0, sizeof s);
        const int ok = copy_shards(fix, d, NULL) == 0 && k3_st_open(&s, d) == 0;
        const int kind = ok ? (int)k3_st_explain_missing(&s, "ghost.tensor", why, sizeof why) : -1;
        ck(ok && s.declared == 2 && s.numbered == 2 && kind == K3_ST_MISS_NOT_IN_CHECKPOINT
           && strstr(why, "2 of 2 declared shards are here") != NULL,
           "complete shard set: the name never existed", why);
        if (ok) k3_st_close(&s);
    }

    /* M10: the released checkpoint names its files model-00001-of-000096.safetensors,
     * five digits then six. Padding the missing name with the index's width alone gave
     * model-00002-of-00003, a file that does not exist anywhere, so a user told to fetch
     * it would get a 404. One fixture shard is published here under that naming scheme,
     * declaring three, and the missing files must be named exactly. */
    {
        char d[1024], src[1024], dst[1100], why[1024] = "";
        int ok = snprintf(d, sizeof d, "%s/m10", work) < (int)sizeof d
              && snprintf(src, sizeof src, "%s/%s", fix, SHARD1) < (int)sizeof src
              && snprintf(dst, sizeof dst, "%s/model-00001-of-000003.safetensors", d)
                     < (int)sizeof dst;
        mkdir(d, 0755);
        size_t n = 0;
        unsigned char *b = ok ? slurp(src, &n) : NULL;
        ok = ok && b && writeall(dst, b, n) == 0;
        free(b);
        K3St s; memset(&s, 0, sizeof s);
        ok = ok && k3_st_open(&s, d) == 0;
        const int kind = ok ? (int)k3_st_explain_missing(&s, "ghost.tensor", why, sizeof why) : -1;
        ck(ok && s.declared == 3 && s.numbered == 1 && kind == K3_ST_MISS_SHARD_ABSENT
           && strstr(why, "model-00002-of-000003.safetensors") != NULL
           && strstr(why, "model-00003-of-000003.safetensors") != NULL,
           "5+6 digit names: the missing files are named as they exist", why);
        if (ok) k3_st_close(&s);
    }

    /* E8M0 scale byte 255 is the NaN code; no quantiser writes it. A shard with one
     * such byte has intact structure and reads back full, so it is the loader, not the
     * reader, that must refuse it, naming the file, the tensor and the count. The clean
     * twin must load through both cache paths, or this case proves nothing. */
    {
        char d[1024], why[1024] = "";
        long long n = -1; int serial = -99, batch = -99;
        snprintf(d, sizeof d, "%s/e8m0_clean", work);
        int ok = write_expert_shard(d, -1) == 0
              && probe_expert_shard(d, &n, why, sizeof why, &serial, &batch) == 0;
        ck(ok && n == 0 && why[0] == 0 && serial == 0 && batch == 0,
           "clean E8M0 scales load through the cache",
           ok ? "" : "harness");

        /* w1.weight_scale is bytes [128,136) of the data region; poison byte 3 of it. */
        snprintf(d, sizeof d, "%s/e8m0_nan", work);
        ok = write_expert_shard(d, EXP_P_BYTES + 3) == 0
          && probe_expert_shard(d, &n, why, sizeof why, &serial, &batch) == 0;
        ck(ok && n == 1
           && strstr(why, "model-00001-of-00001.safetensors") != NULL
           && strstr(why, "experts.0.w1.weight_scale") != NULL
           && strstr(why, " 1 of 8 ") != NULL
           && serial != 0 && batch != 0,
           "one 255 scale byte refused by file/tensor/count", why);
    }

    printf("\n%s\n", g_fail ? "ST-FAULT TESTS FAILED" : "ST-FAULT TESTS PASSED");
    return g_fail ? 1 : 0;
}
