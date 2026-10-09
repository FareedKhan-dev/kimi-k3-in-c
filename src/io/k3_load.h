/* k3_load.h - resolve Kimi K3 checkpoint tensors and stream routed experts.
 *
 * THE MEASUREMENT THIS FILE IS BUILT ON
 *   Reading all 96 shard headers showed that the six tensors making up one routed
 *   expert (w1, w2, w3, each with weight_packed and weight_scale) are laid out as ONE
 *   contiguous run of exactly 17,547,264 bytes, in the order
 *       w1.packed w1.scale w2.packed w2.scale w3.packed w3.scale
 *   with no gaps. This holds for all 896 experts in the shard checked, and the loader
 *   verifies it per expert rather than assuming it.
 *
 *   That is what makes streaming cheap: fetching an
 *   expert is a single coalesced pread, not six scattered ones. On a spinning-rust or
 *   network-backed store the difference is six seeks per expert per layer per token,
 *   which at top-16 over 92 layers is 8,832 seeks a token instead of 1,472.
 *
 *   17,547,264 bytes is 0.53125 bytes per parameter over 33,030,144 parameters: half a
 *   byte for the nibble plus one byte per 32-element group for the E8M0 scale.
 *
 * THE FALLBACK IS NOT DEAD CODE
 *   If a future checkpoint revision interleaves tensors differently, k3_expert_ref
 *   reports contiguous = 0 and k3_expert_load issues six preads into the same buffer
 *   layout. The result is identical; only the seek count changes.
 */
#ifndef K3_LOAD_H
#define K3_LOAD_H

#include "k3_st.h"

/* Geometry of one quantised matrix inside an expert's byte run. */
typedef struct {
    int64_t p_off, p_bytes;   /* packed nibbles, offset WITHIN the run */
    int64_t s_off, s_bytes;   /* E8M0 scales,    offset WITHIN the run */
    int     rows, pcols;      /* packed is [rows][pcols]; logical width is pcols*2 */
    int     scols;            /* scales are [rows][scols], scols == pcols*2/32     */
} K3QMat;

typedef struct {
    int     layer, expert;
    int     shard;
    int64_t off;              /* absolute file offset of the run start */
    int64_t nbytes;           /* total bytes, 17,547,264 for real K3   */
    int     contiguous;       /* 1 when one pread suffices             */
    K3QMat  m[3];             /* w1, w2, w3                            */
} K3ExpertRef;

/* Resolve the six tensors of one expert and validate their geometry. Returns 0 on
 * success, non-zero if any are missing or shaped inconsistently. */
int k3_expert_ref(const K3St *s, int layer, int expert, K3ExpertRef *r);

/* Fetch the raw bytes. buf must hold r->nbytes. One pread when contiguous. */
int64_t k3_expert_load(const K3St *s, const K3ExpertRef *r, unsigned char *buf);

/* As above but with O_DIRECT, bypassing the page cache. A streamed expert is read once
 * and evicted, so buffering only copies it twice and evicts something useful; measured
 * under a 32 GB cap the buffered path ran at 1,247 MB/s on a 6,400 MB/s disk.
 *
 * The read is widened to aligned bounds, so buf must hold r->nbytes + 2*K3_ST_ALIGN and
 * be page aligned. *payload_off receives where the expert actually starts inside buf;
 * every m[i].p_off/s_off is relative to THAT, not to buf. Only valid for a contiguous
 * expert; falls back to k3_expert_load otherwise. */
int64_t k3_expert_load_direct(const K3St *s, const K3ExpertRef *r,
                              unsigned char *buf, int64_t bufcap, int64_t *payload_off);

/* Count the E8M0 scale bytes equal to 255 in an expert's three matrices.
 *
 * WHY 255 IS REFUSED RATHER THAN READ
 *   E8M0 is a bare biased exponent and 255 is its NaN code in the OCP Microscaling (MX)
 *   spec. A quantiser never writes it: every scale is an exact power of two, so a 255
 *   byte can only mean a damaged download, a bad mirror or a repack gone wrong. The
 *   kernels map it to zero so one byte cannot poison a whole row (see k3.h, MXFP4), the
 *   right thing for a kernel and the wrong thing for a loader to rely on: an
 *   expert with 255 bytes in its scales runs with whole 32-weight groups missing and
 *   nothing says so. The scan here is where the bytes first land in memory, so the
 *   refusal happens once, before the first multiply, and names the file.
 *
 *   The scales are 1 byte per 32 weights: 1,032,192 of a real expert's 17,547,264
 *   bytes, one pass, a memchr for the clean case.
 *
 * buf is the expert exactly as k3_expert_load fills it, or the payload inside a
 * k3_expert_load_direct buffer (buf + *payload_off). Returns the total count over the
 * three matrices. When it is non-zero and msg is non-NULL, one line naming the shard
 * file, the first affected weight_scale tensor and its count is written to msg; the
 * caller prints it and refuses the expert. */
int64_t k3_expert_nan_scales(const K3St *s, const K3ExpertRef *r, const unsigned char *buf,
                             char *msg, size_t cap);

/* Dequantise from a buffer filled by k3_expert_load. Each output must hold
 * rows * pcols * 2 floats. Any of the three may be NULL to skip it. */
void k3_expert_dequant(const K3ExpertRef *r, const unsigned char *buf,
                       float *w1, float *w2, float *w3);

/* Elements in each dequantised matrix, for sizing the destination buffers. */
int64_t k3_qmat_numel(const K3QMat *m);

#endif /* K3_LOAD_H */
