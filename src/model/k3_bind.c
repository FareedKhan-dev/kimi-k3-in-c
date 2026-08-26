/* k3_bind.c - see k3_bind.h. */
#define _POSIX_C_SOURCE 200809L

#include "k3_portable_io.h"   /* first: sets _DARWIN_C_SOURCE before any libc header;
                                * on Windows, supplies posix_memalign */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "k3_bind.h"

#define PRE "language_model.model."
#define MAXB 64

/* One requested tensor: where it goes, how big it must be, and in which format.
 *
 * WIDE means widen to float32 on load: the small vectors that kernels dereference
 * elementwise (norms, biases, A_log, dt_bias, conv kernels). NARROW means keep the
 * checkpoint's own bf16 bytes: the large matrices, which are only ever read through
 * k3_mmw and are 99% of the bytes. Holding those at fp32 is what makes the trunk
 * ~227 GB instead of 113.49 GB. */
typedef struct {
    char            name[224];
    const K3Tensor *t;
    int64_t         want;      /* elements the engine expects, -1 to accept whatever */
    int64_t         take;      /* elements actually copied (A_log takes a prefix)    */
    int             narrow;    /* 1 = keep bf16 bytes, 0 = widen to fp32             */
    const void    **dest;
    size_t          off;       /* byte offset into the blob, filled during sizing    */
} Req;

typedef struct {
    Req  r[MAXB];
    int  n;
    int  bad;
    int  narrow_ok;            /* 0 forces everything to fp32 (see k3_bind_layer)    */
    int  demoted;              /* a reqn() tensor was not BF16 after all             */
} Plan;

static void req_(Plan *p, const void **dest, int narrow, int64_t want, int64_t take,
                 const char *fmt, va_list ap)
{
    if (p->n >= MAXB) { fprintf(stderr, "k3_bind: too many tensors\n"); p->bad++; return; }
    Req *q = &p->r[p->n];
    vsnprintf(q->name, sizeof q->name, fmt, ap);
    q->dest = dest; q->want = want; q->take = take < 0 ? want : take;
    q->narrow = narrow && p->narrow_ok;
    q->t = NULL; q->off = 0;
    p->n++;
}

/* WIDE: widened to fp32. */
static void reqw(Plan *p, const float **dest, int64_t want, int64_t take,
                 const char *fmt, ...)
{
    va_list ap; va_start(ap, fmt);
    req_(p, (const void **)dest, 0, want, take, fmt, ap);
    va_end(ap);
}

/* NARROW: kept as the checkpoint's bf16. */
static void reqn(Plan *p, const void **dest, int64_t want, const char *fmt, ...)
{
    va_list ap; va_start(ap, fmt);
    req_(p, dest, 1, want, -1, fmt, ap);
    va_end(ap);
}

static size_t align8(size_t x) { return (x + 7u) & ~(size_t)7u; }

/* Resolve and validate everything, and lay out the blob, before reading a byte. */
static int64_t plan_resolve(Plan *p, const K3St *s)
{
    size_t off = 0;
    for (int i = 0; i < p->n; i++) {
        Req *q = &p->r[i];
        q->t = k3_st_find(s, q->name);
        if (!q->t) {
            fprintf(stderr, "k3_bind: missing tensor %s\n", q->name);
            p->bad++;
            continue;
        }
        const int64_t have = k3_st_numel(q->t);
        /* The check that earns its keep: a shape the engine did not expect means the
         * config and the checkpoint disagree, and every kernel downstream would read
         * the wrong strides while producing plausible numbers. */
        if (q->want >= 0 && have != q->want) {
            fprintf(stderr, "k3_bind: %s has %lld elements, engine expects %lld\n",
                    q->name, (long long)have, (long long)q->want);
            p->bad++;
            continue;
        }
        if (q->take > have) {
            fprintf(stderr, "k3_bind: %s: asked for %lld of %lld elements\n",
                    q->name, (long long)q->take, (long long)have);
            p->bad++;
            continue;
        }
        /* Narrow storage is only legal when the checkpoint really holds bf16. If a
         * tensor ships F32, keeping "its own bytes" would mean handing 4-byte floats
         * to a kernel that reads 2-byte elements.
         *
         * Demoting just this tensor is NOT enough, because the dtype tag lives on the
         * STRUCT, not the field: one demoted tensor inside a struct tagged K3_WBF16
         * would be read as bf16 anyway. So record it, and let the caller fall back
         * wholesale. Right byte count, finite plausible numbers, wrong model is exactly
         * the failure this file's element-count check exists to prevent, and it would
         * be reintroduced one level down. */
        if (q->narrow && q->t->dtype != K3_DT_BF16) { q->narrow = 0; p->demoted++; }

        /* reqn() always takes the whole tensor. If that ever changes, plan_load's raw
         * branch would write nbytes into a region sized take*2 and run off the end. */
        if (q->narrow && q->take != have) {
            fprintf(stderr, "k3_bind: %s: a partial take of a narrow tensor is not "
                            "implemented (%lld of %lld)\n",
                    q->name, (long long)q->take, (long long)have);
            p->bad++;
            continue;
        }

        /* Align every tensor so a bf16 array never leaves the next fp32 array
         * misaligned. */
        off = align8(off);
        q->off = off;
        off += (size_t)q->take * (q->narrow ? 2u : 4u);
    }
    return p->bad ? -1 : (int64_t)off;
}

static int plan_load(Plan *p, const K3St *s, unsigned char *blob)
{
    for (int i = 0; i < p->n; i++) {
        Req *q = &p->r[i];
        const int64_t have = k3_st_numel(q->t);
        void *dst = blob + q->off;

        if (q->narrow) {
            /* Straight bytes, no conversion: this is the whole point. */
            if (k3_st_read(s, q->t, dst) != q->t->nbytes) {
                fprintf(stderr, "k3_bind: short read of %s\n", q->name);
                return -1;
            }
        } else if (q->take == have) {
            if (k3_st_read_f32(s, q->t, (float *)dst) != have) {
                fprintf(stderr, "k3_bind: short read of %s\n", q->name);
                return -1;
            }
        } else {
            /* A prefix: read the whole tensor into scratch, keep the front. Only A_log
             * needs this, and it is 128 floats. */
            float *tmp = (float *)malloc((size_t)have * sizeof(float));
            if (!tmp) return -1;
            if (k3_st_read_f32(s, q->t, tmp) != have) { free(tmp); return -1; }
            memcpy(dst, tmp, (size_t)q->take * sizeof(float));
            free(tmp);
        }
        *q->dest = dst;
    }
    return 0;
}

/* ------------------------------------------------------------------ one layer */

static void plan_layer(Plan *p, const K3Cfg *c, int L, K3LayerBind *b, int is_mla, int is_dense)
{
    const int64_t H = c->hidden;
    const int64_t P = (int64_t)c->kda_heads * c->kda_head_dim;   /* 12288 */

    /* Norms and the attn-res projections are folded ELEMENTWISE, never through a
     * matmul, so they must stay fp32. */
    reqw(p, &b->lay.in_norm,       H, -1, PRE "layers.%d.input_layernorm.weight", L);
    reqw(p, &b->lay.post_norm,     H, -1, PRE "layers.%d.post_attention_layernorm.weight", L);
    reqw(p, &b->lay.attn_res_norm, H, -1, PRE "layers.%d.self_attention_res_norm.weight", L);
    reqw(p, &b->lay.attn_res_proj, H, -1, PRE "layers.%d.self_attention_res_proj.weight", L);
    reqw(p, &b->lay.mlp_res_norm,  H, -1, PRE "layers.%d.mlp_res_norm.weight", L);
    reqw(p, &b->lay.mlp_res_proj,  H, -1, PRE "layers.%d.mlp_res_proj.weight", L);

    if (is_mla) {
        const int64_t qh = (int64_t)c->qk_nope + c->qk_rope;      /* 192 */
        reqn(p, &b->mla.q_a,  (int64_t)c->q_lora * H, PRE "layers.%d.self_attn.q_a_proj.weight", L);
        reqw(p, &b->mla.q_a_norm, c->q_lora, -1, PRE "layers.%d.self_attn.q_a_layernorm.weight", L);
        reqn(p, &b->mla.q_b,  (int64_t)c->n_heads * qh * c->q_lora,
             PRE "layers.%d.self_attn.q_b_proj.weight", L);
        reqn(p, &b->mla.kv_a, (int64_t)(c->kv_lora + c->qk_rope) * H,
             PRE "layers.%d.self_attn.kv_a_proj_with_mqa.weight", L);
        reqw(p, &b->mla.kv_a_norm, c->kv_lora, -1, PRE "layers.%d.self_attn.kv_a_layernorm.weight", L);
        reqn(p, &b->mla.kv_b, (int64_t)c->n_heads * (c->qk_nope + c->v_head) * c->kv_lora,
             PRE "layers.%d.self_attn.kv_b_proj.weight", L);
        reqn(p, &b->mla.o,    H * (int64_t)c->n_heads * c->v_head,
             PRE "layers.%d.self_attn.o_proj.weight", L);
        if (c->mla_out_gate)
            reqn(p, &b->mla.g, (int64_t)c->n_heads * c->v_head * H,
                 PRE "layers.%d.self_attn.g_proj.weight", L);
    } else {
        reqn(p, &b->kda.q, P * H, PRE "layers.%d.self_attn.q_proj.weight", L);
        reqn(p, &b->kda.k, P * H, PRE "layers.%d.self_attn.k_proj.weight", L);
        reqn(p, &b->kda.v, P * H, PRE "layers.%d.self_attn.v_proj.weight", L);
        reqn(p, &b->kda.g, P * H, PRE "layers.%d.self_attn.g_proj.weight", L);
        reqn(p, &b->kda.o, H * P, PRE "layers.%d.self_attn.o_proj.weight", L);
        /* Rank 3 on disk, [H*D][1][conv_k]; the element count is what matters. Read
         * elementwise by k3_shortconv, so fp32. */
        reqw(p, &b->kda.q_conv, P * c->conv_k, -1, PRE "layers.%d.self_attn.q_conv1d.weight", L);
        reqw(p, &b->kda.k_conv, P * c->conv_k, -1, PRE "layers.%d.self_attn.k_conv1d.weight", L);
        reqw(p, &b->kda.v_conv, P * c->conv_k, -1, PRE "layers.%d.self_attn.v_conv1d.weight", L);
        reqn(p, &b->kda.f_a, (int64_t)c->kda_head_dim * H, PRE "layers.%d.self_attn.f_a_proj.weight", L);
        reqn(p, &b->kda.f_b, P * c->kda_head_dim,          PRE "layers.%d.self_attn.f_b_proj.weight", L);
        reqn(p, &b->kda.b,   (int64_t)c->kda_heads * H,    PRE "layers.%d.self_attn.b_proj.weight", L);
        /* PER HEAD. The checkpoint ships kda_head_dim values and zeroes the tail; take
         * the first kda_heads. Accepting all 128 as per-channel is the silent bug this
         * project has documented since the beginning. Read elementwise by
         * k3_kda_decay, so fp32. */
        reqw(p, &b->kda.A_log,   c->kda_head_dim, c->kda_heads, PRE "layers.%d.self_attn.A_log", L);
        reqw(p, &b->kda.dt_bias, P,               -1, PRE "layers.%d.self_attn.dt_bias", L);
        reqw(p, &b->kda.o_norm,  c->kda_head_dim, -1, PRE "layers.%d.self_attn.o_norm.weight", L);
    }

    if (is_dense) {
        reqn(p, &b->lay.dense_gate, (int64_t)c->dense_inter * H, PRE "layers.%d.mlp.gate_proj.weight", L);
        reqn(p, &b->lay.dense_up,   (int64_t)c->dense_inter * H, PRE "layers.%d.mlp.up_proj.weight", L);
        reqn(p, &b->lay.dense_down, H * (int64_t)c->dense_inter, PRE "layers.%d.mlp.down_proj.weight", L);
    } else {
        const int64_t SI = (int64_t)c->moe_inter * c->n_shared;   /* fused: 6144 */
        /* gate stays fp32: k3_router has its own inline matmul. See k3.h. */
        reqw(p, &b->moe.gate, (int64_t)c->n_experts * H, -1,
             PRE "layers.%d.block_sparse_moe.gate.weight", L);
        reqw(p, &b->moe.bias, c->n_experts, -1,
             PRE "layers.%d.block_sparse_moe.gate.e_score_correction_bias", L);
        reqn(p, &b->moe.down, (int64_t)c->latent * H,
             PRE "layers.%d.block_sparse_moe.routed_expert_down_proj.weight", L);
        reqn(p, &b->moe.up,   H * (int64_t)c->latent,
             PRE "layers.%d.block_sparse_moe.routed_expert_up_proj.weight", L);
        reqw(p, &b->moe.latent_norm, c->latent, -1,
             PRE "layers.%d.block_sparse_moe.routed_expert_norm.weight", L);
        reqn(p, &b->moe.sh1, SI * H, PRE "layers.%d.block_sparse_moe.shared_experts.gate_proj.weight", L);
        reqn(p, &b->moe.sh3, SI * H, PRE "layers.%d.block_sparse_moe.shared_experts.up_proj.weight", L);
        reqn(p, &b->moe.sh2, H * SI, PRE "layers.%d.block_sparse_moe.shared_experts.down_proj.weight", L);
    }
}

int64_t k3_bind_layer_bytes(const K3St *s, const K3Cfg *c, int L)
{
    K3LayerBind tmp; memset(&tmp, 0, sizeof tmp);
    Plan p; memset(&p, 0, sizeof p);
    p.narrow_ok = 1;
    plan_layer(&p, c, L, &tmp, k3_is_mla(c, L), k3_is_dense(c, L));
    const int64_t bytes = plan_resolve(&p, s);
    return bytes < 0 ? -1 : bytes;      /* BYTES, not floats */
}

int k3_bind_layer(const K3St *s, const K3Cfg *c, int L, K3LayerBind *b)
{
    memset(b, 0, sizeof *b);
    b->layer = L;
    const int is_mla = k3_is_mla(c, L), is_dense = k3_is_dense(c, L);

    Plan p; memset(&p, 0, sizeof p);
    p.narrow_ok = 1;
    plan_layer(&p, c, L, b, is_mla, is_dense);

    int64_t need = plan_resolve(&p, s);
    if (need < 0) return -1;

    /* If any large matrix is not BF16, redo the whole layer at fp32.
     * The tag is per struct, so a mixed layer cannot be described. */
    if (p.demoted) {
        fprintf(stderr, "k3_bind: layer %d has %d large tensor(s) that are not BF16; "
                        "binding the whole layer at fp32 instead\n", L, p.demoted);
        memset(b, 0, sizeof *b);
        b->layer = L;
        memset(&p, 0, sizeof p);
        p.narrow_ok = 0;
        plan_layer(&p, c, L, b, is_mla, is_dense);
        need = plan_resolve(&p, s);
        if (need < 0) return -1;
    }
    const int wdt = p.narrow_ok ? K3_WBF16 : K3_WF32;

    b->blob = malloc((size_t)need);
    if (!b->blob) {
        fprintf(stderr, "k3_bind: cannot allocate %.2f GB for layer %d\n",
                (double)need / 1e9, L);
        return -1;
    }
    b->nbytes = (size_t)need;
    if (plan_load(&p, s, (unsigned char *)b->blob) != 0) {
        free(b->blob); b->blob = NULL; return -1;
    }

    /* Tag the structs to match how their matrices were ACTUALLY stored, which is what
     * plan_resolve just decided, not what this function hoped for. */
    b->kda.wdt = b->mla.wdt = b->moe.wdt = b->lay.wdt = wdt;

    /* Exactly one of kda/mla is non-NULL; the decoder branches on that, not on a flag. */
    b->lay.kda = is_mla ? NULL : &b->kda;
    b->lay.mla = is_mla ? &b->mla : NULL;
    b->lay.moe = is_dense ? NULL : &b->moe;
    return 0;
}

void k3_bind_free(K3LayerBind *b)
{
    free(b->blob);
    memset(b, 0, sizeof *b);
}

/* ------------------------------------------------- binding from a memory buffer */

size_t k3_bind_widen_bytes(const K3Cfg *c)
{
    /* Only the BF16 vectors that kernels read elementwise are copied. Everything else
     * is pointed at in place. The router gate dominates: it is BF16 on disk but stays
     * fp32 in the engine because k3_router walks it with its own inline matmul. */
    const size_t H = (size_t)c->hidden;
    size_t n = 6 * H                       /* in/post norm, attn-res and mlp-res pair  */
             + (size_t)c->q_lora + c->kv_lora   /* MLA q_a/kv_a layernorms             */
             + (size_t)c->latent                /* routed_expert_norm                  */
             + (size_t)c->n_experts * H;        /* router gate                          */
    return n * sizeof(float) + 4096;       /* slack for per-tensor 8-byte alignment    */
}

int k3_bind_layer_mem(const K3Cfg *c, int L, K3LayerBind *b,
                      const unsigned char *run, const K3MemSrc *src,
                      unsigned char *widen, size_t widen_cap, size_t *widen_used)
{
    memset(b, 0, sizeof *b);
    b->layer = L;
    const int is_mla = k3_is_mla(c, L), is_dense = k3_is_dense(c, L);

    Plan p; memset(&p, 0, sizeof p);
    p.narrow_ok = 1;
    plan_layer(&p, c, L, b, is_mla, is_dense);

    size_t w = 0;
    int narrowed_all = 1;
    int i8_seen = 0;
    for (int i = 0; i < p.n; i++) {
        Req *q = &p.r[i];
        int64_t off = 0, nb = 0; int dt = 0;
        if (src->find(src->ctx, q->name, &off, &nb, &dt) != 0) {
            fprintf(stderr, "k3_bind_mem: %s not present in the packed run\n", q->name);
            return -1;
        }
        /* Per-row int8 draft weight: [f32 scale][int8 * cols] per row. A matmul weight is
         * pointed at directly and the layer is tagged K3_WI8; a tensor the engine reads
         * elementwise as fp32 (the AttnRes projection) is DEQUANTISED into the widen
         * buffer here, row scale times int8, exactly parallel to the bf16 widen path.
         * The element-count check does not apply to the scale-interleaved layout; the
         * packer owns the shape. */
        if (dt == K3_DT_I8R) {
            if (q->narrow) {
                *q->dest = run + off;
                i8_seen = 1;
                continue;
            }
            /* want fp32: dequantise. take is the logical element count (rows*cols); the
             * row width is derivable because each row is [4 bytes scale][cols int8] and
             * nb = rows*(4+cols) with rows*cols == take. Solve rows from nb and take. */
            const int64_t take = q->take;
            /* nb = rows*4 + take  ->  rows = (nb - take)/4 */
            if ((nb - take) % 4 != 0) {
                fprintf(stderr, "k3_bind_mem: %s bad int8 layout\n", q->name);
                return -1;
            }
            const int64_t rows = (nb - take) / 4;
            if (rows <= 0 || take % rows != 0) {
                fprintf(stderr, "k3_bind_mem: %s bad int8 shape\n", q->name);
                return -1;
            }
            const int64_t cols = take / rows;
            w = (w + 7u) & ~(size_t)7u;
            if (w + (size_t)take * 4 > widen_cap) {
                fprintf(stderr, "k3_bind_mem: widen area too small at %s\n", q->name);
                return -1;
            }
            float *dst = (float *)(widen + w);
            const unsigned char *rp = run + off;
            const size_t rowb = 4u + (size_t)cols;
            for (int64_t r = 0; r < rows; r++) {
                float scale;
                memcpy(&scale, rp + (size_t)r * rowb, 4);
                const signed char *q8 = (const signed char *)(rp + (size_t)r * rowb + 4);
                for (int64_t k = 0; k < cols; k++)
                    dst[r * cols + k] = (float)q8[k] * scale;
            }
            *q->dest = dst;
            w += (size_t)take * 4;
            continue;
        }
        const int esz = (dt == K3_DT_F32) ? 4 : (dt == K3_DT_U8 ? 1 : 2);
        const int64_t have = nb / esz;
        if (q->want >= 0 && have != q->want) {
            fprintf(stderr, "k3_bind_mem: %s has %lld elements, engine expects %lld\n",
                    q->name, (long long)have, (long long)q->want);
            return -1;
        }
        if (q->take > have) {
            fprintf(stderr, "k3_bind_mem: %s: asked for %lld of %lld\n",
                    q->name, (long long)q->take, (long long)have);
            return -1;
        }

        if (q->narrow) {
            if (dt != K3_DT_BF16) { narrowed_all = 0; }   /* handled below */
            else { *q->dest = run + off; continue; }
        }

        /* Wanted as fp32. If it is already F32 on disk, point at it; a prefix take
         * (A_log) is just the front of the same array, so that is free too. */
        if (dt == K3_DT_F32) { *q->dest = run + off; continue; }

        if (dt != K3_DT_BF16) {
            fprintf(stderr, "k3_bind_mem: %s has dtype %d, cannot widen\n", q->name, dt);
            return -1;
        }
        w = (w + 7u) & ~(size_t)7u;
        if (w + (size_t)q->take * 4 > widen_cap) {
            fprintf(stderr, "k3_bind_mem: widen area too small at %s (%zu of %zu)\n",
                    q->name, w + (size_t)q->take * 4, widen_cap);
            return -1;
        }
        float *dst = (float *)(widen + w);
        const uint16_t *sp = (const uint16_t *)(run + off);
        for (int64_t k = 0; k < q->take; k++) dst[k] = k3_bf16f(sp[k]);
        *q->dest = dst;
        w += (size_t)q->take * 4;
    }

    if (!narrowed_all && !i8_seen) {
        /* A large matrix was not BF16 in the packed run. The tag is per struct, so this
         * cannot be described; refuse rather than read fp32 bytes as bf16. */
        fprintf(stderr, "k3_bind_mem: layer %d has a non-BF16 large tensor\n", L);
        return -1;
    }

    /* An int8 draft trunk has every matmul weight as I8R (norms stay f32), so one tag
     * describes the layer. The two formats are never mixed within a packed trunk. */
    const int lw = i8_seen ? K3_WI8 : K3_WBF16;
    b->kda.wdt = b->mla.wdt = b->moe.wdt = b->lay.wdt = lw;
    b->lay.kda = is_mla ? NULL : &b->kda;
    b->lay.mla = is_mla ? &b->mla : NULL;
    b->lay.moe = is_dense ? NULL : &b->moe;
    if (widen_used) *widen_used = w;
    return 0;
}

/* ------------------------------------------------------------------ model level */

/* embed is gathered a row at a time rather than multiplied, so k3_run widens the row it
 * needs. lm_head goes through k3_mmw. Both are 2.35 GB as bf16 and 4.70 GB widened,
 * which is why neither is widened here. */
static void plan_model(Plan *p, const K3Cfg *c, int want_embed, int want_lm_head,
                       K3ModelBind *m)
{
    const int64_t H = c->hidden;
    if (want_embed)
        reqn(p, &m->embed, (int64_t)c->vocab * H, PRE "embed_tokens.weight");
    reqw(p, &m->norm,  H, -1, PRE "norm.weight");
    reqw(p, &m->out_res_norm, H, -1, PRE "output_attn_res_norm.weight");
    reqw(p, &m->out_res_proj, H, -1, PRE "output_attn_res_proj.weight");
    if (want_lm_head)
        reqn(p, &m->lm_head, (int64_t)c->vocab * H, "language_model.lm_head.weight");
}

int k3_bind_model_parts(const K3St *s, const K3Cfg *c,
                        int want_embed, int want_lm_head, K3ModelBind *m)
{
    memset(m, 0, sizeof *m);
    Plan p; memset(&p, 0, sizeof p);
    p.narrow_ok = 1;
    plan_model(&p, c, want_embed, want_lm_head, m);

    int64_t need = plan_resolve(&p, s);
    if (need < 0) return -1;
    if (p.demoted) {                       /* same wholesale fallback as k3_bind_layer */
        fprintf(stderr, "k3_bind: %d model-level tensor(s) are not BF16; binding the "
                        "model-level weights at fp32 instead\n", p.demoted);
        memset(m, 0, sizeof *m);
        memset(&p, 0, sizeof p);
        p.narrow_ok = 0;
        plan_model(&p, c, want_embed, want_lm_head, m);
        need = plan_resolve(&p, s);
        if (need < 0) return -1;
    }
    m->blob = malloc((size_t)need);
    if (!m->blob) {
        fprintf(stderr, "k3_bind: cannot allocate %.2f GB for model-level weights\n",
                (double)need / 1e9);
        return -1;
    }
    m->nbytes = (size_t)need;
    if (plan_load(&p, s, (unsigned char *)m->blob) != 0) {
        free(m->blob); m->blob = NULL; return -1;
    }
    m->wdt = p.narrow_ok ? K3_WBF16 : K3_WF32;
    return 0;
}

int k3_bind_model(const K3St *s, const K3Cfg *c, int want_lm_head, K3ModelBind *m)
{
    return k3_bind_model_parts(s, c, 1, want_lm_head, m);
}

void k3_bind_model_free(K3ModelBind *m)
{
    free(m->blob);
    memset(m, 0, sizeof *m);
}

/* ------------------------------------------------------ streamed model matrices */

static int model_matrix(const K3St *s, const char *name, int rows, int cols,
                        const K3Tensor **out, int *wdt)
{
    const K3Tensor *t = k3_st_find(s, name);
    if (!t) {
        fprintf(stderr, "k3_model_stream: missing tensor %s\n", name);
        return -1;
    }
    if (t->ndim != 2 || t->shape[0] != rows || t->shape[1] != cols) {
        fprintf(stderr,
                "k3_model_stream: %s has shape [%lld,%lld] rank %d; expected [%d,%d]\n",
                name, (long long)(t->ndim > 0 ? t->shape[0] : -1),
                (long long)(t->ndim > 1 ? t->shape[1] : -1), t->ndim, rows, cols);
        return -1;
    }
    if (t->dtype == K3_DT_BF16) *wdt = K3_WBF16;
    else if (t->dtype == K3_DT_F32) *wdt = K3_WF32;
    else {
        fprintf(stderr, "k3_model_stream: %s must be BF16 or F32, got dtype %d\n",
                name, (int)t->dtype);
        return -1;
    }
    *out = t;
    return 0;
}

static double model_now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int k3_model_stream_init(K3ModelStream *m, const K3St *s, const K3Cfg *c)
{
    memset(m, 0, sizeof *m);
    m->st = s;
    m->hidden = c->hidden;
    m->vocab = c->vocab;
    if (model_matrix(s, PRE "embed_tokens.weight", c->vocab, c->hidden,
                     &m->embed, &m->embed_wdt) != 0 ||
        model_matrix(s, "language_model.lm_head.weight", c->vocab, c->hidden,
                     &m->lm_head, &m->lm_head_wdt) != 0)
        return -1;

    m->bufcap = K3_MODEL_STREAM_CHUNK + 2u * K3_ST_ALIGN;
    if (posix_memalign((void **)&m->buf, K3_ST_ALIGN, m->bufcap) != 0) {
        fprintf(stderr, "k3_model_stream: cannot allocate %zu-byte aligned I/O buffer\n",
                m->bufcap);
        memset(m, 0, sizeof *m);
        return -1;
    }
    return 0;
}

void k3_model_stream_free(K3ModelStream *m)
{
    k3_aligned_free(m->buf);
    memset(m, 0, sizeof *m);
}

static int model_read_rows(K3ModelStream *m, const K3Tensor *t, int first, int n,
                           const void **rows, uint64_t *counter)
{
    const int esz = k3_st_elemsize(t->dtype);
    const int64_t row_bytes = (int64_t)m->hidden * esz;
    const int64_t nbytes = (int64_t)n * row_bytes;
    int64_t payload = 0;
    if (first < 0 || n <= 0 || first > m->vocab - n ||
        nbytes > (int64_t)K3_MODEL_STREAM_CHUNK)
        return -1;
    const double t0 = model_now_s();
    const int64_t got = k3_st_read_aligned(m->st, t->shard,
                            t->off + (int64_t)first * row_bytes, nbytes,
                            m->buf, (int64_t)m->bufcap, &payload);
    m->read_seconds += model_now_s() - t0;
    if (got != nbytes) {
        fprintf(stderr,
                "k3_model_stream: short read of %s rows %d..%d (%lld of %lld bytes)\n",
                t->name, first, first + n, (long long)got, (long long)nbytes);
        return -1;
    }
    unsigned char *src = m->buf + payload;
    /* Safetensors guarantees byte ranges, not C type alignment. A BF16 tensor may
     * legally follow an odd-sized U8 tensor, making its payload address odd even though
     * the O_DIRECT destination itself is page aligned. Typed uint16_t/float loads from
     * that address are undefined on strict-alignment targets. Compact only in that rare
     * case; memmove is overlap-safe and the existing buffer has enough room. */
    if ((uintptr_t)src % (uintptr_t)esz != 0) {
        memmove(m->buf, src, (size_t)nbytes);
        src = m->buf;
    }
    *rows = src;
    *counter += (uint64_t)got;
    return 0;
}

int k3_model_stream_embed_row(K3ModelStream *m, float *dst, int64_t row)
{
    const void *src = NULL;
    if (row < 0 || row >= m->vocab ||
        model_read_rows(m, m->embed, (int)row, 1, &src, &m->embed_bytes_read) != 0)
        return -1;
    k3_embed_row(dst, src, m->embed_wdt, 0, m->hidden);
    return 0;
}

int k3_model_stream_project(K3ModelStream *m, float *logits, const float *x)
{
    const int esz = k3_st_elemsize(m->lm_head->dtype);
    const int64_t row_bytes = (int64_t)m->hidden * esz;
    const int rows_per_chunk = (int)(K3_MODEL_STREAM_CHUNK / row_bytes);
    if (rows_per_chunk < 1) return -1;

    for (int first = 0; first < m->vocab; first += rows_per_chunk) {
        const int n = (m->vocab - first < rows_per_chunk)
                    ? m->vocab - first : rows_per_chunk;
        const void *rows = NULL;
        if (model_read_rows(m, m->lm_head, first, n, &rows,
                            &m->lm_head_bytes_read) != 0)
            return -1;
        k3_mmw(logits + first, x, rows, m->lm_head_wdt, m->hidden, n);
    }
    return 0;
}
