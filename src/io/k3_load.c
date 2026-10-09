/* k3_load.c - see k3_load.h for the layout measurement this is built on. */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3.h"
#include "k3_load.h"

#define EXPERT_FMT "language_model.model.layers.%d.block_sparse_moe.experts.%d.%s.%s"

int64_t k3_qmat_numel(const K3QMat *m) { return (int64_t)m->rows * m->pcols * 2; }

int k3_expert_ref(const K3St *s, int layer, int expert, K3ExpertRef *r)
{
    static const char *W[3] = { "w1", "w2", "w3" };
    char name[256];
    const K3Tensor *pk[3], *sc[3];

    memset(r, 0, sizeof *r);
    r->layer = layer; r->expert = expert;

    for (int i = 0; i < 3; i++) {
        snprintf(name, sizeof name, EXPERT_FMT, layer, expert, W[i], "weight_packed");
        pk[i] = k3_st_find(s, name);
        if (!pk[i]) { fprintf(stderr, "k3_load: missing %s\n", name); k3_st_note_missing(s, name); return -1; }
        snprintf(name, sizeof name, EXPERT_FMT, layer, expert, W[i], "weight_scale");
        sc[i] = k3_st_find(s, name);
        if (!sc[i]) { fprintf(stderr, "k3_load: missing %s\n", name); k3_st_note_missing(s, name); return -1; }

        if (pk[i]->dtype != K3_DT_U8 || sc[i]->dtype != K3_DT_U8) {
            fprintf(stderr, "k3_load: L%d expert %d %s is not U8\n", layer, expert, W[i]);
            return -1;
        }
        if (pk[i]->ndim != 2 || sc[i]->ndim != 2) {
            fprintf(stderr, "k3_load: L%d expert %d %s is not 2D\n", layer, expert, W[i]);
            return -1;
        }
        if (pk[i]->shape[0] != sc[i]->shape[0]) {
            fprintf(stderr, "k3_load: L%d expert %d %s row mismatch %lld vs %lld\n",
                    layer, expert, W[i],
                    (long long)pk[i]->shape[0], (long long)sc[i]->shape[0]);
            return -1;
        }
        /* The scale count must equal the logical width divided by the group size. If it
         * does not, the group size is not 32 for this tensor and every scale after the
         * first would be applied to the wrong 32 weights. Silent, and catastrophic. */
        const int64_t logical = pk[i]->shape[1] * 2;
        if (sc[i]->shape[1] * K3_MXFP4_GROUP != logical) {
            fprintf(stderr, "k3_load: L%d expert %d %s: %lld scales for %lld elements "
                            "implies group size %.2f, not %d\n",
                    layer, expert, W[i], (long long)sc[i]->shape[1], (long long)logical,
                    (double)logical / (double)sc[i]->shape[1], K3_MXFP4_GROUP);
            return -1;
        }
        if (pk[i]->shard != sc[i]->shard) {
            fprintf(stderr, "k3_load: L%d expert %d %s is split across shards\n",
                    layer, expert, W[i]);
            return -1;
        }
        if (i && pk[i]->shard != pk[0]->shard) {
            fprintf(stderr, "k3_load: L%d expert %d spans shards\n", layer, expert);
            return -1;
        }

        r->m[i].rows  = (int)pk[i]->shape[0];
        r->m[i].pcols = (int)pk[i]->shape[1];
        r->m[i].scols = (int)sc[i]->shape[1];
        r->m[i].p_bytes = pk[i]->nbytes;
        r->m[i].s_bytes = sc[i]->nbytes;
    }

    r->shard = pk[0]->shard;

    /* Is the whole expert one run? Take the lowest offset as the base and check that
     * the six spans tile it exactly. */
    int64_t lo = pk[0]->off, hi = 0, own = 0;
    const K3Tensor *all[6] = { pk[0], sc[0], pk[1], sc[1], pk[2], sc[2] };
    for (int i = 0; i < 6; i++) {
        if (all[i]->off < lo) lo = all[i]->off;
        if (all[i]->off + all[i]->nbytes > hi) hi = all[i]->off + all[i]->nbytes;
        own += all[i]->nbytes;
    }
    r->contiguous = (hi - lo == own);

    if (r->contiguous) {
        r->off = lo;
        r->nbytes = own;
        for (int i = 0; i < 3; i++) {
            r->m[i].p_off = pk[i]->off - lo;
            r->m[i].s_off = sc[i]->off - lo;
        }
    } else {
        /* Pack sequentially in the canonical order instead; six preads will fill it. */
        int64_t o = 0;
        for (int i = 0; i < 3; i++) {
            r->m[i].p_off = o; o += r->m[i].p_bytes;
            r->m[i].s_off = o; o += r->m[i].s_bytes;
        }
        r->off = lo;
        r->nbytes = o;
    }
    return 0;
}

int64_t k3_expert_load(const K3St *s, const K3ExpertRef *r, unsigned char *buf)
{
    if (r->contiguous) {
        /* The whole point: one coalesced read of 17.55 MB. */
        K3Tensor t;
        memset(&t, 0, sizeof t);
        t.name = (char *)"expert";
        t.shard = r->shard;
        t.off = r->off;
        t.nbytes = r->nbytes;
        t.dtype = K3_DT_U8;
        t.ndim = 1;
        t.shape[0] = r->nbytes;
        return k3_st_read(s, &t, buf);
    }

    static const char *W[3] = { "w1", "w2", "w3" };
    char name[256];
    int64_t got = 0;
    for (int i = 0; i < 3; i++) {
        snprintf(name, sizeof name, EXPERT_FMT, r->layer, r->expert, W[i], "weight_packed");
        const K3Tensor *p = k3_st_find(s, name);
        snprintf(name, sizeof name, EXPERT_FMT, r->layer, r->expert, W[i], "weight_scale");
        const K3Tensor *c = k3_st_find(s, name);
        if (!p || !c) return got;
        got += k3_st_read(s, p, buf + r->m[i].p_off);
        got += k3_st_read(s, c, buf + r->m[i].s_off);
    }
    return got;
}

int64_t k3_expert_load_direct(const K3St *s, const K3ExpertRef *r,
                              unsigned char *buf, int64_t bufcap, int64_t *payload_off)
{
    if (!r->contiguous) {                 /* six scattered reads: use the plain path */
        if (payload_off) *payload_off = 0;
        return k3_expert_load(s, r, buf);
    }
    return k3_st_read_aligned(s, r->shard, r->off, r->nbytes, buf, bufcap, payload_off);
}

/* memchr first: a clean tensor, the only kind a quantiser writes, is one vectorised scan
 * and no counting loop. Only a tensor that already failed pays for the exact count. */
static int64_t count_255(const unsigned char *p, size_t n)
{
    const unsigned char *q = (const unsigned char *)memchr(p, 0xFF, n);
    if (!q) return 0;
    int64_t c = 0;
    for (const unsigned char *end = p + n; q < end; q++) c += (*q == 0xFF);
    return c;
}

int64_t k3_expert_nan_scales(const K3St *s, const K3ExpertRef *r, const unsigned char *buf,
                             char *msg, size_t cap)
{
    static const char *W[3] = { "w1", "w2", "w3" };
    int64_t total = 0, first = 0;
    int fi = -1, nbad = 0;
    for (int i = 0; i < 3; i++) {
        const int64_t n = count_255(buf + r->m[i].s_off, (size_t)r->m[i].s_bytes);
        if (n == 0) continue;
        if (fi < 0) { fi = i; first = n; }
        nbad++;
        total += n;
    }
    if (total && msg && cap) {
        const char *file = (s && s->path && r->shard >= 0 && r->shard < s->nshard)
                         ? s->path[r->shard] : "?";
        snprintf(msg, cap,
                 "k3_load: %s: " EXPERT_FMT ": %lld of %lld E8M0 scale bytes are 255, "
                 "the NaN code no quantiser writes%s; refusing L%d expert %d",
                 file, r->layer, r->expert, W[fi], "weight_scale",
                 (long long)first, (long long)r->m[fi].s_bytes,
                 nbad > 1 ? " (other matrices of this expert too)" : "",
                 r->layer, r->expert);
    }
    return total;
}

void k3_expert_dequant(const K3ExpertRef *r, const unsigned char *buf,
                       float *w1, float *w2, float *w3)
{
    float *out[3] = { w1, w2, w3 };
    for (int i = 0; i < 3; i++) {
        if (!out[i]) continue;
        k3_mxfp4_dequant(out[i], buf + r->m[i].p_off, buf + r->m[i].s_off,
                         r->m[i].rows, r->m[i].pcols, K3_MXFP4_GROUP);
    }
}
