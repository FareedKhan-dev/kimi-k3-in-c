/* test_state.c - the --save-state / --load-state file must round-trip exactly and
 * refuse anything else loudly.
 *
 * WHY THIS FILE EXISTS
 *   A state file is restored straight into the KDA recurrent matrices and the MLA KV
 *   cache, and decode continues from whatever it holds. Nothing downstream can tell a
 *   good restore from a bad one: the tokens come out fluent either way. So the reader
 *   is the only place a damaged file can be caught, and until now it had no test that
 *   needed no checkpoint. The arrays here are synthetic; the header logic, the
 *   position-major KV slices, and every refusal are the real code (src/cli/k3_state.c).
 *
 * usage: test_state <work_dir>
 */
#define _POSIX_C_SOURCE 200809L

#include "k3_portable_io.h"   /* mkdir() shim for MinGW; see the header for why */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>
#include <time.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#include "k3_state.h"

static int g_fail = 0;

static void ck(int ok, const char *what, const char *detail)
{
    printf("  %s  %-40s %s\n", ok ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!ok) g_fail++;
}

static unsigned char *slurp(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    unsigned char *b = (unsigned char *)malloc((size_t)sz + 1);
    if (!b || fread(b, 1, (size_t)sz, f) != (size_t)sz) { free(b); fclose(f); return NULL; }
    fclose(f);
    *n = (size_t)sz;
    return b;
}

static int spit(const char *path, const unsigned char *b, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    const int ok = fwrite(b, 1, n, f) == n;
    return (fclose(f) == 0 && ok) ? 0 : -1;
}

/* Small but not degenerate: two MLA layers, four bound layers, a KV cache with more
 * capacity than positions in use, so the position-major slicing is exercised. KPER is
 * what cfg_fill's KDA geometry implies (16*8 + 3*16*3); the loader checks it. */
enum { N_BOUND = 4, N_MLA = 2, KV_CAP = 16, CACHED = 5, NSEQ = 6, KPER = 272 };

static void cfg_fill(K3Cfg *c)
{
    memset(c, 0, sizeof *c);
    c->hidden = 64; c->n_layers = 4; c->vocab = 256;
    c->kda_heads = 2; c->kda_head_dim = 8; c->conv_k = 4;
    c->n_heads = 2; c->qk_nope = 8; c->qk_rope = 4; c->v_head = 8;
    c->n_experts = 4; c->topk = 2;
}

typedef struct {
    int    seq[NSEQ];
    float  ks[KPER * N_BOUND];
    float *kvc, *ropec;       /* N_MLA slices of kv_cap positions */
    int    kv_cap;
    int64_t kvpp, ropepp;
} Arrays;

static void arrays_alloc(Arrays *a, int kv_cap, const K3Cfg *c)
{
    memset(a, 0, sizeof *a);
    a->kv_cap = kv_cap;
    a->kvpp   = (int64_t)c->n_heads * (c->qk_nope + c->v_head);
    a->ropepp = c->qk_rope;
    a->kvc    = (float *)calloc((size_t)N_MLA * kv_cap * a->kvpp, sizeof(float));
    a->ropec  = (float *)calloc((size_t)N_MLA * kv_cap * a->ropepp, sizeof(float));
}

static void arrays_free(Arrays *a) { free(a->kvc); free(a->ropec); }

/* Every value distinct, so a slice landing one row off is caught. */
static void arrays_pattern(Arrays *a)
{
    for (int i = 0; i < NSEQ; i++) a->seq[i] = 7 * i + 3;
    for (int i = 0; i < KPER * N_BOUND; i++) a->ks[i] = 0.25f * (float)i - 3.0f;
    for (int mi = 0; mi < N_MLA; mi++)
        for (int p = 0; p < a->kv_cap; p++) {
            for (int j = 0; j < a->kvpp; j++)
                a->kvc[((size_t)mi * a->kv_cap + p) * a->kvpp + j] =
                    1000.0f * mi + 10.0f * p + 0.01f * j;
            for (int j = 0; j < a->ropepp; j++)
                a->ropec[((size_t)mi * a->kv_cap + p) * a->ropepp + j] =
                    -1000.0f * mi - 10.0f * p - 0.01f * j;
        }
}

/* The OCCUPIED positions of every slice must match; capacity beyond them is scratch. */
static int arrays_same_occupied(const Arrays *x, const Arrays *y)
{
    if (memcmp(x->seq, y->seq, sizeof x->seq) != 0) return 0;
    if (memcmp(x->ks, y->ks, sizeof x->ks) != 0) return 0;
    for (int mi = 0; mi < N_MLA; mi++) {
        const float *xk = x->kvc + (size_t)mi * x->kv_cap * x->kvpp;
        const float *yk = y->kvc + (size_t)mi * y->kv_cap * y->kvpp;
        if (memcmp(xk, yk, (size_t)CACHED * x->kvpp * sizeof(float)) != 0) return 0;
        const float *xr = x->ropec + (size_t)mi * x->kv_cap * x->ropepp;
        const float *yr = y->ropec + (size_t)mi * y->kv_cap * y->ropepp;
        if (memcmp(xr, yr, (size_t)CACHED * x->ropepp * sizeof(float)) != 0) return 0;
    }
    return 1;
}

static int save(const char *path, const K3Cfg *c, const Arrays *a)
{
    return k3_state_save(path, c, a->seq, NSEQ, a->ks, a->kvc, a->ropec,
                         N_BOUND, N_MLA, a->kv_cap, CACHED, KPER, a->kvpp, a->ropepp);
}

static int load(const char *path, const K3Cfg *c, Arrays *a)
{
    K3StateHdr hd;
    if (k3_state_peek(path, &hd) != 0) return -1;
    return k3_state_load(path, c, &hd, a->seq, a->ks, a->kvc, a->ropec,
                         N_BOUND, N_MLA, a->kv_cap);
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: test_state <work_dir>\n"); return 2; }
    const char *work = argv[1];
    mkdir(work, 0755);
    char good[4096], mut[4096];
    snprintf(good, sizeof good, "%s/good.k3st", work);
    snprintf(mut,  sizeof mut,  "%s/mut.k3st",  work);

    K3Cfg c; cfg_fill(&c);
    Arrays src; arrays_alloc(&src, KV_CAP, &c); arrays_pattern(&src);

    printf("state file: round trip\n");
    ck(save(good, &c, &src) == 0, "save writes", good);
    {
        K3StateHdr hd; memset(&hd, 0, sizeof hd);
        ck(k3_state_peek(good, &hd) == 0, "peek accepts what save wrote", NULL);
        ck(hd.nseq == NSEQ && hd.cached == CACHED && hd.n_bound == N_BOUND &&
           hd.n_mla == N_MLA && hd.kper == KPER && hd.kvpp == src.kvpp &&
           hd.ropepp == src.ropepp, "header carries the geometry", NULL);
    }
    {
        /* A resumed run may size its KV cache differently; the slices must still land. */
        Arrays dst; arrays_alloc(&dst, KV_CAP + 9, &c);
        ck(load(good, &c, &dst) == 0, "load into a larger cache", NULL);
        ck(arrays_same_occupied(&src, &dst), "every occupied value identical", NULL);
        arrays_free(&dst);
    }
    {
        Arrays dst; arrays_alloc(&dst, CACHED, &c);   /* exactly full */
        ck(load(good, &c, &dst) == 0, "load into an exactly-full cache", NULL);
        ck(arrays_same_occupied(&src, &dst), "every occupied value identical", NULL);
        arrays_free(&dst);
    }

    printf("state file: refusals\n");
    size_t n = 0;
    unsigned char *img = slurp(good, &n);
    if (!img) { fprintf(stderr, "cannot read back %s\n", good); return 1; }
    ck(n > sizeof(K3StateHdr), "file is header plus payload", NULL);
    {
        Arrays dst; arrays_alloc(&dst, KV_CAP, &c);
        spit(mut, img, n - 7);
        ck(load(mut, &c, &dst) != 0, "truncated payload refused", NULL);
        spit(mut, img, sizeof(K3StateHdr) - 1);
        ck(load(mut, &c, &dst) != 0, "truncated header refused", NULL);
        arrays_free(&dst);
    }
    {
        unsigned char *m = (unsigned char *)malloc(n); memcpy(m, img, n);
        K3StateHdr *h = (K3StateHdr *)m;
        h->version = K3_STATE_VER + 1;
        spit(mut, m, n);
        K3StateHdr hd;
        ck(k3_state_peek(mut, &hd) != 0, "other version refused at peek", NULL);
        memcpy(m, img, n); m[1] ^= 0xFF;
        spit(mut, m, n);
        ck(k3_state_peek(mut, &hd) != 0, "wrong magic refused at peek", NULL);
        free(m);
    }
    {
        /* Same file, different architecture: the fingerprint is the guard. */
        K3Cfg other = c; other.topk = 3;
        Arrays dst; arrays_alloc(&dst, KV_CAP, &c);
        ck(load(good, &other, &dst) != 0, "other architecture refused", NULL);
        K3StateHdr hd; k3_state_peek(good, &hd);
        ck(k3_state_load(good, &c, &hd, dst.seq, dst.ks, dst.kvc, dst.ropec,
                         N_BOUND + 1, N_MLA, KV_CAP) != 0, "other layer count refused", NULL);
        ck(k3_state_load(good, &c, &hd, dst.seq, dst.ks, dst.kvc, dst.ropec,
                         N_BOUND, N_MLA, CACHED - 1) != 0, "cache too small refused", NULL);
        arrays_free(&dst);
    }
    printf("state file: header geometry\n");
    {
        /* The reads are sized and strided by these fields into buffers the caller sized
         * from its config, so each is refused before any read. Before that check, a kper
         * one too large overran ks, and the shifted pair below loaded cleanly: the bytes
         * read are the same, so the payload hash matched, and KV rows landed in the rope
         * cache. */
        int64_t kper, kvpp, ropepp;
        k3_state_geom(&c, &kper, &kvpp, &ropepp);
        ck(kper == KPER && kvpp == src.kvpp && ropepp == src.ropepp,
           "geometry helper matches the arrays", NULL);
        unsigned char *m = (unsigned char *)malloc(n);
        Arrays dst; arrays_alloc(&dst, KV_CAP, &c);
        K3StateHdr *h = (K3StateHdr *)m;
        memcpy(m, img, n); h->kper += 1; spit(mut, m, n);
        ck(load(mut, &c, &dst) != 0, "kper larger than the config refused", NULL);
        memcpy(m, img, n); h->kvpp += 4; h->ropepp -= 4; spit(mut, m, n);
        ck(load(mut, &c, &dst) != 0, "KV/rope stride shifted refused", NULL);
        memcpy(m, img, n); h->nseq = -1; spit(mut, m, n);
        ck(load(mut, &c, &dst) != 0, "negative token count refused", NULL);
        memcpy(m, img, n); h->cached = NSEQ + 1; spit(mut, m, n);
        ck(load(mut, &c, &dst) != 0, "more cached positions than ids refused", NULL);
        /* Same byte count, every check above satisfied: one position moved from the KV
         * cache into the token ids. Only the header being inside the hash sees this. */
        memcpy(m, img, n);
        h->cached -= 1; h->nseq += (int32_t)((src.kvpp + src.ropepp) * N_MLA);
        spit(mut, m, n);
        Arrays big; memset(&big, 0, sizeof big);
        int *seq = (int *)calloc((size_t)h->nseq, sizeof(int));
        arrays_alloc(&big, KV_CAP, &c);
        ck(seq && k3_state_load(mut, &c, h, seq, big.ks, big.kvc, big.ropec,
                                N_BOUND, N_MLA, KV_CAP) != 0,
           "counts moving the same bytes refused", NULL);
        free(seq); arrays_free(&big);
        unsigned char *t = (unsigned char *)malloc(n + 1);
        memcpy(t, img, n); t[n] = 0; spit(mut, t, n + 1);
        ck(load(mut, &c, &dst) != 0, "trailing bytes refused", NULL);
        free(t); free(m);
        arrays_free(&dst);
    }
    printf("state file: payload integrity\n");
    {
        /* One flipped byte in each region. Structure intact, every fread full: this is
         * exactly the damage a length check cannot see, and a restore from it decodes
         * fluent, wrong tokens. */
        const size_t H = sizeof(K3StateHdr);
        const size_t off_seq   = H;
        const size_t off_ks    = off_seq + (size_t)NSEQ * sizeof(int);
        const size_t off_kvc   = off_ks + (size_t)KPER * N_BOUND * sizeof(float);
        const size_t off_ropec = off_kvc + (size_t)N_MLA * CACHED * src.kvpp * sizeof(float);
        const struct { const char *name; size_t at; } flips[] = {
            { "flipped byte in the token ids refused", off_seq + 5 },
            { "flipped byte in the recurrent state refused", off_ks + 333 },
            { "flipped byte in the KV cache refused", off_kvc + 77 },
            { "flipped byte in the rope rows refused", off_ropec + 1 },
            { "flipped byte in the last payload byte refused", n - 1 },
        };
        unsigned char *m = (unsigned char *)malloc(n);
        Arrays dst; arrays_alloc(&dst, KV_CAP, &c);
        for (size_t i = 0; i < sizeof flips / sizeof flips[0]; i++) {
            memcpy(m, img, n);
            m[flips[i].at] ^= 0x01;
            spit(mut, m, n);
            char d[64]; snprintf(d, sizeof d, "byte %zu of %zu", flips[i].at, n);
            ck(load(mut, &c, &dst) != 0, flips[i].name, d);
        }
        /* The other direction: a good payload under a header whose hash was damaged. */
        memcpy(m, img, n);
        m[H - 1] ^= 0x01;
        spit(mut, m, n);
        ck(load(mut, &c, &dst) != 0, "damaged header hash refused", NULL);
        /* And the untouched file still loads, so the refusals above are the flips. */
        ck(load(good, &c, &dst) == 0, "untouched file still loads", NULL);
        free(m);
        arrays_free(&dst);
    }
    {
        /* Not a gate, a number: the hash runs once per save and once per load over a
         * payload that is 0.63 GB of recurrent state plus the KV cache at full scale. */
        const size_t nb = (size_t)64 << 20;
        unsigned char *buf = (unsigned char *)malloc(nb);
        for (size_t i = 0; i < nb; i++) buf[i] = (unsigned char)(i * 2654435761u >> 24);
        const clock_t t0 = clock();
        const uint64_t h = k3_state_hash(K3_STATE_HASH_INIT, buf, nb);
        const double s = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("  INFO  hash of %zu MB: %016llx in %.3f s (%.0f MB/s)\n",
               nb >> 20, (unsigned long long)h, s, s > 0 ? (double)(nb >> 20) / s : 0.0);
        ck(k3_state_hash(K3_STATE_HASH_INIT, buf, nb) == h, "hash is deterministic", NULL);
        buf[nb / 2] ^= 0x80;
        ck(k3_state_hash(K3_STATE_HASH_INIT, buf, nb) != h, "hash sees one flipped bit", NULL);
        free(buf);
    }

    printf("state file: publication\n");
    {
        /* Publish by rename, so the destination never holds a half-written state: a
         * crash mid-save leaves the previous file, not a truncated one. The observable
         * on POSIX is the inode: a rename installs a new file, an in-place rewrite keeps
         * the old one. */
#ifndef _WIN32
        struct stat before, after;
        ck(stat(good, &before) == 0, "stat before re-save", NULL);
        ck(save(good, &c, &src) == 0, "re-save over an existing file", NULL);
        ck(stat(good, &after) == 0, "stat after re-save", NULL);
        ck(before.st_ino != after.st_ino, "destination is a new inode (rename)", NULL);
        /* mkstemp creates 0600; the writer must hand the file back at the mode a plain
         * fopen would have given it, or a state saved by one user stops being readable
         * by the group that could read it before. */
        {
            const mode_t um = umask(0); umask(um);
            const mode_t want = 0666 & ~um, got = after.st_mode & 0777;
            char d[48]; snprintf(d, sizeof d, "mode %04o, want %04o (umask %04o)", (unsigned)got, (unsigned)want, (unsigned)um);
            ck(got == want, "published mode is fopen's, not mkstemp's", d);
        }
#else
        ck(save(good, &c, &src) == 0, "re-save over an existing file", NULL);
        printf("  SKIP  %-40s %s\n", "destination is a new inode (rename)", "no inode on Windows");
#endif
        /* Nothing staged is left behind. */
        int residue = 0;
        DIR *d = opendir(work);
        if (d) {
            struct dirent *e;
            while ((e = readdir(d)) != NULL)
                if (strstr(e->d_name, ".tmp")) { residue++; printf("        leftover: %s\n", e->d_name); }
            closedir(d);
        }
        ck(d != NULL && residue == 0, "no temp file left after a save", NULL);
    }
#ifndef _WIN32
    {
        /* A save that cannot stage its temp file must fail without touching the old
         * file. Directory write permission is what mkstemp needs and what an in-place
         * fopen(path, "wb") does not, so this is the difference between the two. */
        char rodir[4096], ro[4096];
        snprintf(rodir, sizeof rodir, "%s/ro", work);
        snprintf(ro, sizeof ro, "%s/ro/state.k3st", work);
        mkdir(rodir, 0755);
        chmod(rodir, 0755);
        ck(save(ro, &c, &src) == 0, "save into a writable directory", NULL);
        size_t n0 = 0; unsigned char *img0 = slurp(ro, &n0);
        Arrays other; arrays_alloc(&other, KV_CAP, &c); arrays_pattern(&other);
        other.seq[0] ^= 0x7F;   /* a different payload, so an in-place rewrite would show */
        if (chmod(rodir, 0555) == 0 && access(rodir, W_OK) != 0) {
            const int rc = save(ro, &c, &other);
            size_t n1 = 0; unsigned char *img1 = slurp(ro, &n1);
            ck(rc != 0, "save with no room to stage fails", NULL);
            ck(img0 && img1 && n0 == n1 && memcmp(img0, img1, n0) == 0,
               "and the old file is byte-identical", NULL);
            free(img1);
        } else {
            printf("  SKIP  %-40s %s\n", "save with no room to stage fails",
                   "directory stayed writable (running as root?)");
        }
        chmod(rodir, 0755);
        free(img0);
        arrays_free(&other);
    }
#endif
    free(img);
    arrays_free(&src);

    printf("\n%s\n", g_fail ? "STATE FILE TESTS FAILED" : "STATE FILE TESTS PASSED");
    return g_fail ? 1 : 0;
}
