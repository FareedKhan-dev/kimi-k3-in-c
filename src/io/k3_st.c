/* k3_st.c - safetensors reader for the real Kimi K3 checkpoint.
 *
 * WHY A HAND-WRITTEN SCANNER INSTEAD OF json.h
 *   A safetensors header is machine generated with a rigid shape: one flat object whose
 *   values are all small objects with exactly three known keys. Building a general DOM
 *   for it costs an allocation per node across 78 MB of JSON and 497,220 tensors, for
 *   no benefit. The scanner below walks the text once and writes straight into the
 *   index.
 *
 *   The obvious objection to a hand-written parser is that it can be subtly wrong. So
 *   it is not trusted: tools/verify_st.py reparses the same shard with Python's json
 *   and compares dtype, shape and both offsets for every tensor. The parser is checked
 *   against an external implementation on real data, the same way every other part of
 *   this engine has been.
 */
#define _GNU_SOURCE            /* O_DIRECT */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "k3_portable_io.h"   /* first: sets _DARWIN_C_SOURCE before any libc header */

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "k3_st.h"

/* ------------------------------------------------------------------ helpers */

int k3_st_elemsize(K3Dtype d)
{
    switch (d) {
    case K3_DT_U8:   return 1;
    case K3_DT_BF16:
    case K3_DT_F16:  return 2;
    case K3_DT_F32:  return 4;
    default:         return 0;
    }
}

int64_t k3_st_numel(const K3Tensor *t)
{
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return t->ndim ? n : 1;
}

static K3Dtype dtype_of(const char *s, size_t n)
{
    if (n == 2 && !memcmp(s, "U8", 2))   return K3_DT_U8;
    if (n == 4 && !memcmp(s, "BF16", 4)) return K3_DT_BF16;
    if (n == 3 && !memcmp(s, "F16", 3))  return K3_DT_F16;
    if (n == 3 && !memcmp(s, "F32", 3))  return K3_DT_F32;
    return K3_DT_UNKNOWN;
}

/* FNV-1a. Names are long and share deep prefixes
 * ("language_model.model.layers.N.block_sparse_moe.experts.M...."), so the hash must
 * mix every byte; a prefix-only or length-only hash would pile every expert of a layer
 * into one bucket. */
static uint64_t fnv1a(const char *s)
{
    uint64_t h = 1469598103934665603ull;
    while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211ull; }
    return h;
}

/* ------------------------------------------------------------------ scanner */

typedef struct { const char *p, *end; } Scan;

static void ws(Scan *s) { while (s->p < s->end && (unsigned char)*s->p <= ' ') s->p++; }

static int lit(Scan *s, char c) { ws(s); if (s->p < s->end && *s->p == c) { s->p++; return 1; } return 0; }

/* Read a JSON string into out (NUL terminated). Tensor names are dotted identifiers so
 * escapes never appear in practice, but a parser that silently mangles one would
 * corrupt a name and turn a lookup into a spurious "missing weight". Handle them. */
static int str_(Scan *s, char *out, size_t cap, size_t *len)
{
    ws(s);
    if (s->p >= s->end || *s->p != '"') return 0;
    s->p++;
    size_t n = 0;
    while (s->p < s->end && *s->p != '"') {
        char c = *s->p++;
        if (c == '\\') {
            if (s->p >= s->end) return 0;
            char e = *s->p++;
            switch (e) {
            case 'n': c = '\n'; break;  case 't': c = '\t'; break;
            case 'r': c = '\r'; break;  case 'b': c = '\b'; break;
            case 'f': c = '\f'; break;
            case 'u': {                 /* \uXXXX: keep ASCII, drop the rest */
                if (s->end - s->p < 4) return 0;
                unsigned v = 0;
                for (int i = 0; i < 4; i++) {
                    char h = *s->p++;
                    v = v * 16u + (unsigned)(h <= '9' ? h - '0' : (h | 32) - 'a' + 10);
                }
                c = (char)(v < 128 ? v : '?');
                break;
            }
            default: c = e;             /* covers \" \\ \/ */
            }
        }
        if (n + 1 < cap) out[n] = c;
        n++;
    }
    if (s->p >= s->end) return 0;
    s->p++;                              /* closing quote */
    if (n + 1 > cap) return 0;
    out[n] = '\0';
    if (len) *len = n;
    return 1;
}

static int i64_(Scan *s, int64_t *v)
{
    ws(s);
    int neg = 0;
    if (s->p < s->end && (*s->p == '-' || *s->p == '+')) neg = (*s->p++ == '-');
    if (s->p >= s->end || *s->p < '0' || *s->p > '9') return 0;
    int64_t a = 0;
    while (s->p < s->end && *s->p >= '0' && *s->p <= '9') a = a * 10 + (*s->p++ - '0');
    *v = neg ? -a : a;
    return 1;
}

/* Skip any value, tracking nesting and staying out of strings. Used for __metadata__,
 * whose shape is arbitrary. */
static int skip_value(Scan *s)
{
    ws(s);
    if (s->p >= s->end) return 0;
    if (*s->p == '"') {
        /* Walk the string rather than calling str_: this only needs to advance past the
         * value, and str_ requires a destination buffer and its capacity. */
        s->p++;
        while (s->p < s->end && *s->p != '"') { if (*s->p == '\\') s->p++; s->p++; }
        return s->p < s->end ? (s->p++, 1) : 0;
    }
    if (*s->p == '{' || *s->p == '[') {
        int depth = 0;
        do {
            if (s->p >= s->end) return 0;
            char c = *s->p++;
            if (c == '"') {
                while (s->p < s->end && *s->p != '"') { if (*s->p == '\\') s->p++; s->p++; }
                if (s->p >= s->end) return 0;
                s->p++;
            } else if (c == '{' || c == '[') depth++;
            else if (c == '}' || c == ']') depth--;
        } while (depth > 0);
        return 1;
    }
    while (s->p < s->end && *s->p != ',' && *s->p != '}' && *s->p != ']') s->p++;
    return 1;
}

/* ------------------------------------------------------------------ growth */

typedef struct {
    K3Tensor *t;  size_t n, cap;
    size_t   *noff;                       /* name offsets, resolved to pointers later */
} Build;

static int push(Build *b, K3St *s, const char *name, size_t nlen, const K3Tensor *src)
{
    if (b->n == b->cap) {
        size_t nc = b->cap ? b->cap * 2 : 4096;
        K3Tensor *nt = (K3Tensor *)realloc(b->t, nc * sizeof *nt);
        size_t   *no = (size_t   *)realloc(b->noff, nc * sizeof *no);
        if (!nt || !no) { b->t = nt ? nt : b->t; b->noff = no ? no : b->noff; return -1; }
        b->t = nt; b->noff = no; b->cap = nc;
    }
    if (s->strlen_ + nlen + 1 > s->strcap) {
        size_t nc = s->strcap ? s->strcap : 1 << 20;
        while (s->strlen_ + nlen + 1 > nc) nc *= 2;
        char *np = (char *)realloc(s->strpool, nc);
        if (!np) return -1;
        s->strpool = np; s->strcap = nc;
    }
    b->noff[b->n] = s->strlen_;
    memcpy(s->strpool + s->strlen_, name, nlen + 1);
    s->strlen_ += nlen + 1;
    b->t[b->n] = *src;
    b->n++;
    return 0;
}

/* ------------------------------------------------------------------ one shard */

static int scan_shard(K3St *s, Build *b, int shard, const char *path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "k3_st: cannot open %s\n", path); return -1; }

    unsigned char lenbuf[8];
    if (pread(fd, lenbuf, 8, 0) != 8) {
        fprintf(stderr, "k3_st: %s is too short for a header length\n", path);
        close(fd); return -1;
    }
    uint64_t hlen = 0;
    for (int i = 7; i >= 0; i--) hlen = (hlen << 8) | lenbuf[i];   /* little endian */

    off_t fsize = lseek(fd, 0, SEEK_END);
    if (hlen == 0 || (uint64_t)fsize < 8 + hlen) {
        fprintf(stderr, "k3_st: %s header length %llu is impossible (file %lld bytes)\n",
                path, (unsigned long long)hlen, (long long)fsize);
        close(fd); return -1;
    }

    char *json = (char *)malloc(hlen + 1);
    if (!json) { close(fd); return -1; }
    ssize_t got = 0;
    while ((uint64_t)got < hlen) {
        ssize_t r = pread(fd, json + got, hlen - got, 8 + got);
        if (r <= 0) break;
        got += r;
    }
    if ((uint64_t)got != hlen) {
        fprintf(stderr, "k3_st: short read of %s header\n", path);
        free(json); close(fd); return -1;
    }
    json[hlen] = '\0';

    const int64_t base = (int64_t)(8 + hlen);   /* data_offsets are relative to this */

    Scan sc = { json, json + hlen };
    if (!lit(&sc, '{')) {
        fprintf(stderr, "k3_st: %s header is not a JSON object\n", path); goto bad;
    }

    static char name[512];
    int first = 1, ntensor = 0;
    int64_t maxend = 0;
    for (;;) {
        ws(&sc);
        if (sc.p < sc.end && *sc.p == '}') { sc.p++; break; }
        if (!first && !lit(&sc, ',')) { fprintf(stderr, "k3_st: %s expected ','\n", path); goto bad; }
        first = 0;
        ws(&sc);
        if (sc.p < sc.end && *sc.p == '}') { sc.p++; break; }   /* trailing comma */

        size_t nlen = 0;
        if (!str_(&sc, name, sizeof name, &nlen)) {
            fprintf(stderr, "k3_st: %s bad tensor name near byte %ld\n",
                    path, (long)(sc.p - json)); goto bad;
        }
        if (!lit(&sc, ':')) { fprintf(stderr, "k3_st: %s expected ':'\n", path); goto bad; }

        if (!strcmp(name, "__metadata__")) { if (!skip_value(&sc)) goto bad; continue; }

        if (!lit(&sc, '{')) { fprintf(stderr, "k3_st: %s entry is not an object\n", path); goto bad; }

        K3Tensor t; memset(&t, 0, sizeof t);
        t.shard = shard;
        int have_dt = 0, have_off = 0;
        int64_t o0 = 0, o1 = 0;

        int f1 = 1;
        for (;;) {
            ws(&sc);
            if (sc.p < sc.end && *sc.p == '}') { sc.p++; break; }
            if (!f1 && !lit(&sc, ',')) goto bad;
            f1 = 0;
            char key[64]; size_t klen;
            if (!str_(&sc, key, sizeof key, &klen)) goto bad;
            if (!lit(&sc, ':')) goto bad;

            if (!strcmp(key, "dtype")) {
                char dv[32]; size_t dl;
                if (!str_(&sc, dv, sizeof dv, &dl)) goto bad;
                t.dtype = dtype_of(dv, dl);
                if (t.dtype == K3_DT_UNKNOWN) {
                    fprintf(stderr, "k3_st: %s: unsupported dtype '%s' on %s\n", path, dv, name);
                    goto bad;
                }
                have_dt = 1;
            } else if (!strcmp(key, "shape")) {
                if (!lit(&sc, '[')) goto bad;
                ws(&sc);
                if (sc.p < sc.end && *sc.p == ']') sc.p++;       /* scalar: shape [] */
                else for (;;) {
                    int64_t d;
                    if (!i64_(&sc, &d)) goto bad;
                    if (t.ndim < 4) t.shape[t.ndim] = d;
                    else { fprintf(stderr, "k3_st: %s has rank > 4\n", name); goto bad; }
                    t.ndim++;
                    ws(&sc);
                    if (lit(&sc, ',')) continue;
                    if (lit(&sc, ']')) break;
                    goto bad;
                }
            } else if (!strcmp(key, "data_offsets")) {
                if (!lit(&sc, '[')) goto bad;
                if (!i64_(&sc, &o0)) goto bad;
                if (!lit(&sc, ',')) goto bad;
                if (!i64_(&sc, &o1)) goto bad;
                if (!lit(&sc, ']')) goto bad;
                have_off = 1;
            } else {
                if (!skip_value(&sc)) goto bad;
            }
        }

        if (!have_dt || !have_off) {
            fprintf(stderr, "k3_st: %s: %s is missing dtype or data_offsets\n", path, name);
            goto bad;
        }

        /* Consistency: the byte span must equal elements times element size. A mismatch
         * means the shape and the data disagree, and every later read of this tensor
         * would be silently misaligned. Refuse rather than load it. */
        t.off    = base + o0;
        t.nbytes = o1 - o0;
        const int64_t want = k3_st_numel(&t) * k3_st_elemsize(t.dtype);
        if (t.nbytes != want) {
            fprintf(stderr, "k3_st: %s: %s spans %lld bytes but shape implies %lld\n",
                    path, name, (long long)t.nbytes, (long long)want);
            goto bad;
        }
        if (base + o1 > fsize) {
            fprintf(stderr, "k3_st: %s: %s ends past EOF\n", path, name);
            goto bad;
        }
        if (o1 > maxend) maxend = o1;

        if (push(b, s, name, nlen, &t) != 0) { fprintf(stderr, "k3_st: out of memory\n"); goto bad; }
        ntensor++;
    }

    if (base + maxend != fsize)
        fprintf(stderr, "k3_st: note: %s has %lld trailing bytes after the last tensor\n",
                path, (long long)(fsize - base - maxend));

    free(json);
    s->fd[shard] = fd;
    /* A second descriptor on the same file, for streamed expert reads that must not go
     * through the page cache. Optional: if the filesystem refuses O_DIRECT the reader
     * falls back to fd[]. */
    if (s->dfd) {
        s->dfd[shard] = open(path, O_RDONLY | O_DIRECT);
        k3_set_direct(s->dfd[shard]);   /* no-op off Darwin; advisory, failure is fine */
    }
    return ntensor;
bad:
    free(json);
    close(fd);
    return -1;
}

/* ------------------------------------------------------------------ open/close */

static int cmp_str(const void *a, const void *b)
{
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

int k3_st_open(K3St *s, const char *dir)
{
    memset(s, 0, sizeof *s);

    DIR *d = opendir(dir);
    if (!d) { fprintf(stderr, "k3_st: cannot open directory %s\n", dir); return -1; }

    char **files = NULL; int nf = 0, cf = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        size_t n = strlen(e->d_name);
        if (n < 12 || strcmp(e->d_name + n - 12, ".safetensors")) continue;
        if (nf == cf) { cf = cf ? cf * 2 : 32; files = (char **)realloc(files, cf * sizeof *files); }
        size_t len = strlen(dir) + 1 + n + 1;
        files[nf] = (char *)malloc(len);
        snprintf(files[nf], len, "%s/%s", dir, e->d_name);
        nf++;
    }
    closedir(d);

    if (nf == 0) { fprintf(stderr, "k3_st: no .safetensors files in %s\n", dir); free(files); return -1; }
    /* Sort so shard indices are stable across runs and machines; readdir order is not. */
    qsort(files, nf, sizeof *files, cmp_str);

    s->path = files; s->nshard = nf;
    s->fd  = (int *)malloc(nf * sizeof(int));
    s->dfd = (int *)malloc(nf * sizeof(int));
    /* k3_st_close, not free(files): s->path was aliased to `files` two lines above, so
     * freeing it here leaves s->path dangling and k3_st_close would free it a second
     * time. Let the one function that owns the teardown do all of it. */
    if (!s->fd || !s->dfd) { k3_st_close(s); return -1; }
    for (int i = 0; i < nf; i++) { s->fd[i] = -1; s->dfd[i] = -1; }

    Build b; memset(&b, 0, sizeof b);
    for (int i = 0; i < nf; i++) {
        if (scan_shard(s, &b, i, files[i]) < 0) {
            free(b.t); free(b.noff); k3_st_close(s); return -1;
        }
    }

    /* Resolve names now that the pool has stopped moving. Storing char* during the scan
     * would leave every earlier pointer dangling after a realloc. */
    s->t  = b.t;
    s->nt = (int)b.n;
    for (size_t i = 0; i < b.n; i++) s->t[i].name = s->strpool + b.noff[i];
    free(b.noff);

    int nb = 1024;
    while (nb < s->nt * 2) nb <<= 1;              /* load factor below 0.5 */
    s->nbucket = nb;
    s->bucket = (int32_t *)malloc((size_t)nb * sizeof(int32_t));
    if (!s->bucket) { k3_st_close(s); return -1; }
    memset(s->bucket, 0xFF, (size_t)nb * sizeof(int32_t));   /* -1 */

    for (int i = 0; i < s->nt; i++) {
        uint64_t h = fnv1a(s->t[i].name);
        int j = (int)(h & (uint64_t)(nb - 1));
        while (s->bucket[j] >= 0) {
            if (!strcmp(s->t[s->bucket[j]].name, s->t[i].name)) {
                fprintf(stderr, "k3_st: duplicate tensor name %s\n", s->t[i].name);
                k3_st_close(s); return -1;
            }
            j = (j + 1) & (nb - 1);
        }
        s->bucket[j] = i;
    }
    return 0;
}

void k3_st_close(K3St *s)
{
    if (s->fd)  { for (int i = 0; i < s->nshard; i++) if (s->fd[i]  >= 0) close(s->fd[i]);  free(s->fd); }
    if (s->dfd) { for (int i = 0; i < s->nshard; i++) if (s->dfd[i] >= 0) close(s->dfd[i]); free(s->dfd); }
    if (s->path) { for (int i = 0; i < s->nshard; i++) free(s->path[i]); free(s->path); }
    free(s->t); free(s->bucket); free(s->strpool);
    memset(s, 0, sizeof *s);
}

const K3Tensor *k3_st_find(const K3St *s, const char *name)
{
    if (!s->bucket) return NULL;
    uint64_t h = fnv1a(name);
    int j = (int)(h & (uint64_t)(s->nbucket - 1));
    while (s->bucket[j] >= 0) {
        const K3Tensor *t = &s->t[s->bucket[j]];
        if (!strcmp(t->name, name)) return t;
        j = (j + 1) & (s->nbucket - 1);
    }
    return NULL;
}

/* ------------------------------------------------------------------ reading */

int64_t k3_st_read_aligned(const K3St *s, int shard, int64_t off, int64_t nbytes,
                           void *buf, int64_t bufcap, int64_t *payload_off)
{
    if (shard < 0 || shard >= s->nshard) return 0;
    const int dfd = s->dfd ? s->dfd[shard] : -1;

    if (dfd < 0) {                      /* no O_DIRECT: plain buffered read */
        if (payload_off) *payload_off = 0;
        if (bufcap < nbytes) return 0;
        int64_t got = 0;
        while (got < nbytes) {
            ssize_t r = pread(s->fd[shard], (char *)buf + got,
                              (size_t)(nbytes - got), (off_t)(off + got));
            if (r <= 0) return got;
            got += r;
        }
        return got;
    }

    /* Widen outward to the enclosing aligned window. */
    const int64_t lo = off & ~(int64_t)(K3_ST_ALIGN - 1);
    const int64_t hi = (off + nbytes + K3_ST_ALIGN - 1) & ~(int64_t)(K3_ST_ALIGN - 1);
    const int64_t len = hi - lo;
    const int64_t pad = off - lo;
    if (len > bufcap) return 0;
    if (payload_off) *payload_off = pad;

    int64_t got = 0;
    while (got < len) {
        const int64_t req = (len - got < K3_PREAD_MAX) ? (len - got) : K3_PREAD_MAX;
        ssize_t r = pread(dfd, (char *)buf + got, (size_t)req, (off_t)(lo + got));
        if (r <= 0) {
            /* The final window of a shard can extend past EOF, which is a short read
             * rather than an error. Accept it once the payload itself is covered. */
            break;
        }
        got += r;
    }
    return got >= pad + nbytes ? nbytes : (got > pad ? got - pad : 0);
}

int64_t k3_st_read(const K3St *s, const K3Tensor *t, void *buf)
{
    /* One coalesced pread, looped only because the kernel may return short. This is the
     * call the streaming tier will make per expert: a 17.55 MB contiguous span. */
    int64_t got = 0;
    while (got < t->nbytes) {
        const int64_t remain = t->nbytes - got;
        const int64_t req = remain < K3_PREAD_MAX ? remain : K3_PREAD_MAX;
        ssize_t r = pread(s->fd[t->shard], (char *)buf + got,
                          (size_t)req, (off_t)(t->off + got));
        if (r <= 0) {
            fprintf(stderr, "k3_st: short read on %s at +%lld\n", t->name, (long long)got);
            return got;
        }
        got += r;
    }
    return got;
}

/* Widen in bounded chunks rather than reading the whole tensor into a temporary first.
 *
 * embed_tokens and lm_head are 2.35 GB each as bf16. A full-size staging buffer means a
 * 2.35 GB transient on top of the 4.70 GB destination, per tensor, which is enough to
 * get the process OOM-killed on a box that the memory plan says has room. Reading a few
 * megabytes at a time caps the transient at CHUNK regardless of tensor size, and costs
 * nothing: the reads are still sequential and still large enough to saturate the device.
 */
#define K3_WIDEN_CHUNK (4 << 20)

int64_t k3_st_read_f32(const K3St *s, const K3Tensor *t, float *out)
{
    const int64_t n = k3_st_numel(t);
    if (t->dtype == K3_DT_F32) return k3_st_read(s, t, out) / 4;

    const int esz = k3_st_elemsize(t->dtype);
    if (esz <= 0) return 0;
    /* A whole number of elements per chunk, so no element straddles a boundary. */
    const int64_t chunk_elems = K3_WIDEN_CHUNK / esz;

    void *raw = malloc((size_t)chunk_elems * esz);
    if (!raw) return 0;

    int64_t done = 0;
    while (done < n) {
        const int64_t take = (n - done < chunk_elems) ? (n - done) : chunk_elems;
        const int64_t want = take * esz;
        int64_t got = 0;
        while (got < want) {
            ssize_t r = pread(s->fd[t->shard], (char *)raw + got, (size_t)(want - got),
                              (off_t)(t->off + done * esz + got));
            if (r <= 0) {
                fprintf(stderr, "k3_st: short read widening %s at element %lld\n",
                        t->name, (long long)done);
                free(raw);
                return done;
            }
            got += r;
        }

        float *o = out + done;
        if (t->dtype == K3_DT_BF16) {
            const uint16_t *p = (const uint16_t *)raw;
            for (int64_t i = 0; i < take; i++) o[i] = k3_bf16_to_f32(p[i]);
        } else if (t->dtype == K3_DT_U8) {
            const unsigned char *p = (const unsigned char *)raw;
            for (int64_t i = 0; i < take; i++) o[i] = (float)p[i];
        } else if (t->dtype == K3_DT_F16) {
            const uint16_t *p = (const uint16_t *)raw;
            for (int64_t i = 0; i < take; i++) {
                uint16_t h = p[i];
                uint32_t sign = (uint32_t)(h & 0x8000) << 16;
                uint32_t exp  = (h >> 10) & 0x1F, man = h & 0x3FF;
                union { uint32_t u; float f; } v;
                if (exp == 0) {
                    if (man == 0) v.u = sign;
                    else {                              /* subnormal: renormalise */
                        int sh = 0;
                        while (!(man & 0x400)) { man <<= 1; sh++; }
                        man &= 0x3FF;
                        v.u = sign | ((uint32_t)(127 - 15 - sh + 1) << 23) | (man << 13);
                    }
                } else if (exp == 31) v.u = sign | 0x7F800000u | (man << 13);
                else v.u = sign | ((exp - 15 + 127) << 23) | (man << 13);
                o[i] = v.f;
            }
        }
        done += take;
    }
    free(raw);
    return n;
}
