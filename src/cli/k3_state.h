/* k3_state.h - the conversation state file behind --save-state / --load-state.
 *
 * Everything the engine carries between tokens, on disk. The point is turn two of a
 * conversation: without this, resuming re-reads the whole prompt through all 93 layers,
 * which on a streamed trunk costs minutes; with it, a resumed session pays only for the
 * tokens actually new.
 *
 * Three things are carried, and only three: the KDA recurrent matrices plus ShortConv
 * history (fixed size, independent of context), the MLA KV cache, and the shared rope
 * rows. The AttnRes block buffer is NOT carried because forward() clears it on entry and
 * rebuilds it from the layer outputs every pass; saving it would be saving scratch.
 *
 * The KV cache is stored position-major inside each MLA layer's slice, so only the
 * OCCUPIED positions are written and a resumed run may size its cache differently. The
 * header carries a config fingerprint: restoring state built by a different architecture
 * would produce fluent, wrong output with nothing to indicate it, which is the one
 * failure mode this engine refuses to have.
 *
 * This lives outside k3_run.c so the format has a test that needs no checkpoint
 * (tests/unit/test_state.c): the arrays are just floats and ints of the sizes the header
 * declares, and every refusal below is reachable with synthetic ones. */
#ifndef K3_STATE_H
#define K3_STATE_H

#include <stddef.h>
#include <stdint.h>

#include "k3.h"

#define K3_STATE_MAGIC "K3ST"
/* Version 2 adds payload_hash. Version 3 folds the header into it. An older file is
 * refused rather than read with a guessed layout; state files are session artifacts,
 * written and read by the same binary within one conversation, not a long-lived
 * format. */
#define K3_STATE_VER   3

typedef struct {
    char    magic[4];
    int32_t version;
    int32_t fp[12];        /* config fingerprint */
    int32_t n_bound, n_mla, cached, nseq;
    int64_t kper;          /* KDA+conv floats per layer */
    int64_t kvpp, ropepp;  /* KV / rope floats per position, per MLA layer */
    /* k3_state_hash over this header with payload_hash zeroed, then the payload bytes
     * in file order: seq, ks, then the KV and rope slices. Checked on load; a file
     * whose every fread comes back full can still hold a damaged matrix, or a header
     * whose counts move the same bytes to different places, and this is the only check
     * that sees either. */
    uint64_t payload_hash;
} K3StateHdr;

#define K3_STATE_HASH_INIT 0xcbf29ce484222325ULL   /* FNV-1a offset basis */
uint64_t k3_state_hash(uint64_t h, const void *p, size_t n);

void k3_state_fp(const K3Cfg *c, int32_t *fp);

/* The per-layer state geometry this config implies: KDA+conv floats per bound layer,
 * and KV / rope floats per position per MLA layer. The one definition the engine sizes
 * its buffers from and the loader checks a header against. */
void k3_state_geom(const K3Cfg *c, int64_t *kper, int64_t *kvpp, int64_t *ropepp);

/* Reads only the header, so the caller can size buffers before committing to a load.
 * Returns 0, or -1 after printing why the file was refused. */
int k3_state_peek(const char *path, K3StateHdr *hd);

/* Restores the payload described by a header k3_state_peek accepted. seq must hold
 * hd->nseq ints, ks hd->kper * n_bound floats, kvc and ropec n_mla slices of kv_cap
 * positions each. Returns 0, or -1 after printing why; on -1 the arrays are not to be
 * trusted. */
int k3_state_load(const char *path, const K3Cfg *c, const K3StateHdr *hd,
                  int *seq, float *ks, float *kvc, float *ropec,
                  int n_bound, int n_mla, int kv_cap);

int k3_state_save(const char *path, const K3Cfg *c, const int *seq, int nseq,
                  const float *ks, const float *kvc, const float *ropec,
                  int n_bound, int n_mla, int kv_cap, int cached,
                  int64_t kper, int64_t kvpp, int64_t ropepp);

#endif /* K3_STATE_H */
