/* SPDX-License-Identifier: Apache-2.0 */
/* k3_cfg.h - build a K3Cfg from a config JSON, in either shape, without ever guessing.
 *
 * TWO SHAPES, ONE READER
 *   Two different JSON layouts describe this model, and the engine must read both:
 *
 *     released config.json    nested. The language model lives under "text_config",
 *                             and the KDA/MLA layer map under
 *                             text_config.linear_attn_config. The SiTU betas are named
 *                             activation_situ_beta / activation_situ_linear_beta.
 *
 *     ref_k3.json             flat. make_k3_oracle.py hoists everything to one level
 *                             and shortens the betas to situ_beta / situ_linear_beta.
 *
 *   Every lookup here tries both spellings, so callers never branch on which shape they
 *   were handed and the test fixtures exercise the same code path as the real model.
 *
 * THE ONE RULE: AN ABSENT FIELD IS AN ERROR, NEVER A DEFAULT.
 *   This is worth stating plainly because the alternative fails in the worst possible
 *   way. Point a flat-only reader at the released nested config and every lookup misses.
 *   If each miss took a default:
 *     - situ_beta/situ_linear_beta would default to 4.0/25.0, which are CORRECT, so the
 *       activation looks right and inspires confidence;
 *     - full_attn_layers would come back empty, so k3_is_mla() answers 0 for every layer
 *       and all 93 layers run as KDA.
 *   The result is a model that loads, streams, decodes, and prints fluent-looking tokens
 *   from the wrong architecture. Nothing crashes and no number looks out of place.
 *
 *   So a missing key fails the load. Every missing key is collected and reported
 *   together rather than one per run, because rediscovering them one at a time across a
 *   93-layer load wastes an enormous amount of operator time.
 *
 * Both shapes are covered by the test suite: tests/unit/test_cfg.c reads the flat
 * fixture, and the nested path is exercised by any run against a released checkpoint.
 */
#ifndef K3_CFG_H
#define K3_CFG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "json.h"
#include "k3.h"

/* ---------------------------------------------------------------- lookup ---- */
/* Candidate objects, searched in order: the nested language-model config, the nested
 * linear-attention block, then the root (which IS the config in the flat fixture).
 * Candidate names cover the released spelling and the fixture spelling. */
typedef struct {
    jval *txt;    /* text_config, or NULL in the flat shape                     */
    jval *lin;    /* text_config.linear_attn_config, or NULL                    */
    jval *root;   /* the document root; also the config object when flat        */
    int   nested; /* 1 = released shape, 0 = flat fixture shape                 */
    /* Missing required keys accumulate here so one run reports all of them. */
    const char *missing[32];
    int   nmissing;
} K3CfgSrc;

static inline jval *k3cfg_find(K3CfgSrc *s, const char *primary, const char *alias)
{
    jval *objs[3] = { s->txt, s->lin, s->root };
    const char *names[2] = { primary, alias };
    for (int i = 0; i < 3; i++) {
        if (!objs[i]) continue;
        for (int j = 0; j < 2; j++) {
            if (!names[j]) continue;
            jval *v = json_get(objs[i], names[j]);
            if (v) return v;
        }
    }
    return NULL;
}

static inline void k3cfg_miss(K3CfgSrc *s, const char *name)
{
    if (s->nmissing < (int)(sizeof s->missing / sizeof s->missing[0]))
        s->missing[s->nmissing] = name;
    s->nmissing++;
}

static inline int k3cfg_i(K3CfgSrc *s, const char *primary, const char *alias)
{
    jval *v = k3cfg_find(s, primary, alias);
    if (!v || v->t != J_NUM) { k3cfg_miss(s, primary); return 0; }
    return (int)v->num;
}

static inline float k3cfg_f(K3CfgSrc *s, const char *primary, const char *alias)
{
    jval *v = k3cfg_find(s, primary, alias);
    if (!v || v->t != J_NUM) { k3cfg_miss(s, primary); return 0.0f; }
    return (float)v->num;
}

/* Booleans are written as JSON true/false in the released config and as Python-ish
 * true/false in the fixture; both parse to J_BOOL. A numeric 0/1 is also accepted
 * because hand-edited configs use it. */
static inline int k3cfg_b(K3CfgSrc *s, const char *primary, const char *alias, int dflt)
{
    jval *v = k3cfg_find(s, primary, alias);
    if (!v) return dflt;                       /* genuinely optional: has a real default */
    if (v->t == J_BOOL) return v->boolean ? 1 : 0;
    if (v->t == J_NUM)  return v->num != 0.0;
    return dflt;
}

/* ------------------------------------------------------------------ load ---- */
/* fa receives the ONE-BASED full-attention (MLA) layer list, as both config shapes
 * store it; k3_is_mla() compares against layer+1. Returns 1 on success.
 *
 * On failure it prints every missing key and returns 0. Callers must not proceed:
 * a half-filled K3Cfg is exactly the silent-wrong-model case this file exists to stop.
 */
static inline int k3_cfg_load(K3Cfg *c, int *fa, int fa_max, jval *root, const char *whence)
{
    memset(c, 0, sizeof *c);

    K3CfgSrc s; memset(&s, 0, sizeof s);
    s.root = root;
    s.txt  = json_get(root, "text_config");
    s.nested = (s.txt != NULL);
    jval *base = s.txt ? s.txt : root;
    s.lin  = json_get(base, "linear_attn_config");

    c->hidden   = k3cfg_i(&s, "hidden_size", NULL);
    c->n_layers = k3cfg_i(&s, "num_hidden_layers", NULL);
    c->vocab    = k3cfg_i(&s, "vocab_size", NULL);
    c->rms_eps  = k3cfg_f(&s, "rms_norm_eps", NULL);

    /* KDA. Released: linear_attn_config.num_heads / .head_dim. Fixture: kda_num_heads /
     * kda_head_dim at top level. Searching lin before root is what disambiguates
     * num_heads, which would otherwise collide with nothing but reads ambiguously. */
    c->kda_heads    = k3cfg_i(&s, "num_heads", "kda_num_heads");
    c->kda_head_dim = k3cfg_i(&s, "head_dim",  "kda_head_dim");
    c->conv_k       = k3cfg_i(&s, "short_conv_kernel_size", NULL);
    c->gate_lb      = k3cfg_f(&s, "gate_lower_bound", NULL);

    /* Gated MLA */
    c->n_heads      = k3cfg_i(&s, "num_attention_heads", NULL);
    c->q_lora       = k3cfg_i(&s, "q_lora_rank", NULL);
    c->kv_lora      = k3cfg_i(&s, "kv_lora_rank", NULL);
    c->qk_nope      = k3cfg_i(&s, "qk_nope_head_dim", NULL);
    c->qk_rope      = k3cfg_i(&s, "qk_rope_head_dim", NULL);
    c->v_head       = k3cfg_i(&s, "v_head_dim", NULL);
    c->mla_out_gate = k3cfg_b(&s, "mla_use_output_gate", NULL, 1);

    /* Stable LatentMoE */
    c->n_experts    = k3cfg_i(&s, "num_experts", NULL);
    c->topk         = k3cfg_i(&s, "num_experts_per_token", NULL);
    c->n_shared     = k3cfg_i(&s, "num_shared_experts", NULL);
    c->latent       = k3cfg_i(&s, "routed_expert_hidden_size", NULL);
    c->moe_inter    = k3cfg_i(&s, "moe_intermediate_size", NULL);
    c->routed_scale = k3cfg_f(&s, "routed_scaling_factor", NULL);
    c->moe_renorm   = k3cfg_b(&s, "moe_renormalize", NULL, 1);
    c->latent_norm  = k3cfg_b(&s, "latent_moe_use_norm", NULL, 1);

    c->first_dense  = k3cfg_i(&s, "first_k_dense_replace", NULL);
    c->dense_inter  = k3cfg_i(&s, "intermediate_size", NULL);
    c->attn_res_block = k3cfg_i(&s, "attn_res_block_size", NULL);

    /* SiTU-GLU. The released and fixture spellings differ; both are accepted, and
     * neither is defaulted. These two are the most dangerous fields in the file to
     * default, because 4.0/25.0 are the correct values: a defaulting reader produces a
     * plausible activation and hides the fact that it read nothing else either. */
    c->situ_b1 = k3cfg_f(&s, "activation_situ_beta",        "situ_beta");
    c->situ_b2 = k3cfg_f(&s, "activation_situ_linear_beta", "situ_linear_beta");

    /* Layer map. Released: text_config.linear_attn_config.full_attn_layers.
     * Fixture: full_attn_layers at top level. */
    jval *fal = k3cfg_find(&s, "full_attn_layers", NULL);
    if (!fal || fal->t != J_ARR || fal->len == 0) {
        k3cfg_miss(&s, "full_attn_layers");
    } else if (fal->len > fa_max) {
        fprintf(stderr, "k3_cfg: %s lists %d full-attention layers, buffer holds %d\n",
                whence, fal->len, fa_max);
        return 0;
    } else {
        for (int i = 0; i < fal->len; i++) {
            if (fal->kids[i]->t != J_NUM) {
                fprintf(stderr, "k3_cfg: %s full_attn_layers[%d] is not a number\n",
                        whence, i);
                return 0;
            }
            fa[i] = (int)fal->kids[i]->num;
        }
        c->n_full_attn = fal->len;
        c->full_attn = fa;
    }

    if (s.nmissing) {
        fprintf(stderr, "k3_cfg: %s is missing %d required field(s):\n", whence, s.nmissing);
        int shown = s.nmissing;
        int cap = (int)(sizeof s.missing / sizeof s.missing[0]);
        if (shown > cap) shown = cap;
        for (int i = 0; i < shown; i++) fprintf(stderr, "    %s\n", s.missing[i]);
        if (s.nmissing > shown) fprintf(stderr, "    ... and %d more\n", s.nmissing - shown);
        fprintf(stderr, "  refusing to substitute defaults: a config this reader cannot\n"
                        "  fully understand would silently produce a DIFFERENT model.\n");
        return 0;
    }

    /* ---- structural checks ----
     * These catch a config that parses cleanly but cannot describe this architecture.
     * Each has been seen or is one typo away. */
    if (c->n_layers <= 0 || c->hidden <= 0 || c->vocab <= 0) {
        fprintf(stderr, "k3_cfg: %s has non-positive layers/hidden/vocab\n", whence);
        return 0;
    }
    if (c->n_full_attn >= c->n_layers) {
        fprintf(stderr, "k3_cfg: %s marks %d of %d layers as full attention, leaving no "
                        "KDA layers\n", whence, c->n_full_attn, c->n_layers);
        return 0;
    }
    for (int i = 0; i < c->n_full_attn; i++) {
        if (fa[i] < 1 || fa[i] > c->n_layers) {
            fprintf(stderr, "k3_cfg: %s full_attn_layers[%d] = %d is outside 1..%d "
                            "(the list is ONE-based)\n", whence, i, fa[i], c->n_layers);
            return 0;
        }
    }
    if (c->topk > K3_MAX_TOPK) {
        fprintf(stderr, "k3_cfg: %s selects top-%d, but this build supports at most %d\n"
                        "  (K3_MAX_TOPK in k3.h bounds the fixed-size routing arrays)\n",
                whence, c->topk, K3_MAX_TOPK);
        return 0;
    }
    if (c->topk > c->n_experts) {
        fprintf(stderr, "k3_cfg: %s selects %d of %d experts\n",
                whence, c->topk, c->n_experts);
        return 0;
    }
    if (c->attn_res_block <= 0) {
        fprintf(stderr, "k3_cfg: %s has attn_res_block_size %d; layer_idx %% 0 would "
                        "divide by zero\n", whence, c->attn_res_block);
        return 0;
    }
    if (c->conv_k < 1) {
        fprintf(stderr, "k3_cfg: %s has short_conv_kernel_size %d\n", whence, c->conv_k);
        return 0;
    }

    printf("config: %s (%s shape) | hidden=%d layers=%d vocab=%d | %d MLA + %d KDA | "
           "experts %d top%d shared%d | latent=%d\n",
           whence, s.nested ? "nested" : "flat", c->hidden, c->n_layers, c->vocab,
           c->n_full_attn, c->n_layers - c->n_full_attn,
           c->n_experts, c->topk, c->n_shared, c->latent);
    return 1;
}

/* Convenience: read and parse a config file, then load it. k3_cfg_load copies
 * every value it keeps (numbers into K3Cfg, layer indices into fa), so the
 * parse tree is dead on return and freed here on ALL paths, success or
 * refusal. (Older revisions leaked the tree and called it process-lifetime;
 * a config parse is kilobytes, and leaks are how 700 KB disappear.) */
static inline int k3_cfg_load_file(K3Cfg *c, int *fa, int fa_max, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 0; }
    if (fseek(f, 0, SEEK_END) != 0) { perror(path); fclose(f); return 0; }
    long n = ftell(f);
    if (n < 0 || n > (1L << 28)) {
        fprintf(stderr, "%s: implausible config size %ld\n", path, n);
        fclose(f); return 0;
    }
    if (fseek(f, 0, SEEK_SET) != 0) { perror(path); fclose(f); return 0; }
    char *txt = (char *)malloc((size_t)n + 1);
    if (!txt) { fprintf(stderr, "OOM reading %s\n", path); fclose(f); return 0; }
    size_t got = fread(txt, 1, (size_t)n, f);
    fclose(f);
    txt[got] = 0;

    char *arena = NULL;
    jval *root = json_parse(txt, &arena);
    if (!root) { fprintf(stderr, "%s: not valid JSON\n", path); free(txt); return 0; }
    const int rc = k3_cfg_load(c, fa, fa_max, root, path);
    json_free_tree(root);
    free(txt);
    return rc;
}

#endif /* K3_CFG_H */
