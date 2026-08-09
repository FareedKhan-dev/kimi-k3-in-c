/* k3_bind.h - bind real checkpoint tensors into the engine's weight structs.
 *
 * WHAT THIS SOLVES
 *   k3_st finds any tensor by name; k3.h describes what the kernels want. Nothing
 *   connected the two. This does, for one layer at a time, and it validates as it goes:
 *   every tensor's element count is checked against what the config implies BEFORE the
 *   bytes are read. A wrong assumption about a shape then fails loudly at load instead
 *   of silently producing a transposed or truncated matrix.
 *
 * NAMES, taken from the released checkpoint rather than guessed
 *   layers.L.input_layernorm.weight                       [hidden]
 *   layers.L.post_attention_layernorm.weight              [hidden]
 *   layers.L.self_attention_res_{norm,proj}.weight        [hidden], [1][hidden]
 *   layers.L.mlp_res_{norm,proj}.weight                   [hidden], [1][hidden]
 *   KDA: self_attn.{q,k,v,g}_proj, {q,k,v}_conv1d, f_a_proj, f_b_proj,
 *        b_proj, A_log, dt_bias, o_norm, o_proj
 *   MLA: self_attn.q_a_proj, q_a_layernorm, q_b_proj,
 *        kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, o_proj, g_proj
 *   MoE: block_sparse_moe.gate.weight, gate.e_score_correction_bias,
 *        routed_expert_{down,up}_proj.weight, routed_expert_norm.weight,
 *        shared_experts.{gate,up,down}_proj.weight
 *   Dense (layer 0 only): mlp.{gate,up,down}_proj.weight
 *
 * TWO SHAPES THAT LOOK WRONG AND ARE NOT
 *   q_conv1d.weight ships as [H*D][1][conv_k], rank 3. The kernel wants [H*D][conv_k].
 *   Same bytes, same order; only the redundant middle axis differs, so the element
 *   count check passes and no repacking is needed.
 *
 *   A_log ships as [128] on a 96-head model. Elements 96..127 are exactly zero. It is
 *   PER HEAD, so the loader takes the first n_heads values. Reading all 128 as if they
 *   were per channel runs without complaint and gives a different model.
 *
 * MEMORY, AND THE TWO STORAGE CLASSES
 *   Large matrices keep the checkpoint's own bf16 bytes and are widened inside
 *   k3_mmw. Small vectors are widened here to fp32, because the kernels that consume
 *   them (k3_rmsnorm, k3_shortconv, k3_kda_decay, k3_router) index them ELEMENTWISE,
 *   where a silently wrong element type is not a crash but a different model.
 *
 *   The split is deliberate and cheap: the small vectors are 606,371,424 elements
 *   across the 93-layer trunk, so keeping them fp32 costs 2.43 GB, of which 1.18 GB is
 *   the router gate alone. In exchange the C type system enforces the distinction, and
 *   a mistake shows up as a compile error rather than as plausible output.
 *
 *   Measured totals: trunk 108.81 GB, embed and lm_head 4.70 GB, so 113.49 GB resident
 *   at bf16 against about 227 GB if everything were fp32. See k3.h for the breakdown.
 */
#ifndef K3_BIND_H
#define K3_BIND_H

#include <string.h>          /* k3_embed_row uses memcpy on the fp32 path */

#include "k3.h"
#include "k3_st.h"

typedef struct {
    void    *blob;         /* one allocation holding every weight of this layer,
                            * MIXED bf16 and fp32; each tensor is 8-byte aligned */
    size_t   nbytes;
    int      layer;
    K3KdaW   kda;
    K3MlaW   mla;
    K3MoeW   moe;
    K3LayerW lay;          /* points into kda/mla/moe and blob */
} K3LayerBind;

/* Bind one decoder layer. Returns 0 on success. Routed expert pointers (moe.w1/w2/w3)
 * are left NULL: those are streamed per token by k3_load, not resident. */
int  k3_bind_layer(const K3St *s, const K3Cfg *c, int layer, K3LayerBind *b);
void k3_bind_free(K3LayerBind *b);

/* Model-level weights: embedding, final norm, lm_head, and the one model-level
 * attn-res aggregator. embed and lm_head are 2.35 GB each at bf16 and 4.70 GB at fp32,
 * so pass want_lm_head = 0 when only the trunk is being exercised. */
typedef struct {
    void  *blob;
    size_t nbytes;
    const void  *embed, *lm_head;        /* bf16 when wdt is K3_WBF16 */
    const float *norm;
    const float *out_res_norm, *out_res_proj;
    int    wdt;
} K3ModelBind;

int  k3_bind_model(const K3St *s, const K3Cfg *c, int want_lm_head, K3ModelBind *m);
/* Selective form used by the ultra-low-memory path. The ordinary entry point above
 * remains the compatibility wrapper that always binds the embedding table. */
int  k3_bind_model_parts(const K3St *s, const K3Cfg *c,
                         int want_embed, int want_lm_head, K3ModelBind *m);
void k3_bind_model_free(K3ModelBind *m);

/* ---- streamed model-level matrices ------------------------------------------------
 *
 * embed_tokens and lm_head are 2.35 GB each in the released BF16 checkpoint. A short
 * prompt needs only a few embedding rows, and greedy projection can walk lm_head in
 * bounded row chunks. Each chunk still goes through k3_mmw, so the per-row arithmetic
 * and reduction tree are identical to the resident path; only ownership and I/O
 * lifetime change.
 *
 * One page-aligned buffer is shared by both operations. Returned values never point
 * into it: embedding rows are widened into caller memory, and logits are completed
 * before the next chunk overwrites the buffer. */
typedef struct {
    const K3St     *st;
    const K3Tensor *embed, *lm_head;
    unsigned char  *buf;
    size_t          bufcap;
    int             hidden, vocab;
    int             embed_wdt, lm_head_wdt;
    uint64_t        embed_bytes_read, lm_head_bytes_read;
    double          read_seconds;
} K3ModelStream;

#define K3_MODEL_STREAM_CHUNK (4u << 20)

int  k3_model_stream_init(K3ModelStream *m, const K3St *s, const K3Cfg *c);
void k3_model_stream_free(K3ModelStream *m);
int  k3_model_stream_embed_row(K3ModelStream *m, float *dst, int64_t row);
int  k3_model_stream_project(K3ModelStream *m, float *logits, const float *x);

/* BYTES one layer needs, without loading it. For sizing and for reporting. */
int64_t k3_bind_layer_bytes(const K3St *s, const K3Cfg *c, int layer);

/* ---- binding from a memory buffer, for the streaming trunk ------------------------
 * k3_trunk holds a layer's tensors as a verbatim copy of their contiguous run on disk.
 * Binding from that buffer must use EXACTLY the same tensor names, shapes and
 * wide/narrow decisions as binding from the shards, or the two paths silently diverge.
 * So it reuses the same plan, and only the source of the bytes differs.
 *
 * find() reports where a tensor sits inside the run. Most tensors then need no copy at
 * all: a BF16 matrix is pointed at directly, and so is an F32 vector. Only the handful
 * of BF16 vectors that kernels read elementwise get widened into `widen`.
 */
typedef struct {
    int (*find)(void *ctx, const char *name,
                int64_t *off, int64_t *nbytes, int *dtype);
    void *ctx;
} K3MemSrc;

int k3_bind_layer_mem(const K3Cfg *c, int layer, K3LayerBind *b,
                      const unsigned char *run, const K3MemSrc *src,
                      unsigned char *widen, size_t widen_cap, size_t *widen_used);

/* Upper bound on the widen area one layer needs, so a slot can be sized once. */
size_t k3_bind_widen_bytes(const K3Cfg *c);

/* Gather one embedding row into dst[hidden], widening if the table is bf16. The table
 * is indexed rather than multiplied, so it cannot go through k3_mmw: a plain memcpy
 * with a float stride would read half a row of the wrong values. */
static inline void k3_embed_row(float *dst, const void *table, int wdt,
                                int64_t row, int hidden)
{
    if (wdt == K3_WBF16) {
        const uint16_t *p = (const uint16_t *)table + row * hidden;
        for (int i = 0; i < hidden; i++) dst[i] = k3_bf16f(p[i]);
    } else {
        memcpy(dst, (const float *)table + row * hidden, (size_t)hidden * sizeof(float));
    }
}

#endif /* K3_BIND_H */
