# C API

For embedding the engine rather than using the `k3` binary. The public surface is
`include/k3/k3.h` and `include/k3/k3_cfg.h`.

```c
#include <k3/k3.h>
#include <k3/k3_cfg.h>
```

The API is deliberately small: a configuration struct, weight-binding structs, and the
kernels. There is no context object and no hidden global state, everything a call needs
is passed to it.

## Configuration

```c
K3Cfg cfg;
int   full_attn[128];

if (!k3_cfg_load_file(&cfg, full_attn, 128, "model/config.json")) {
    /* Do not proceed. A partially-populated K3Cfg describes a different model. */
    return 1;
}
```

`k3_cfg_load_file` reads the checkpoint's own configuration and populates every field.

**It never substitutes a default for a missing field.** If a key is absent it collects
every such key, reports them together, and returns 0. This matters more than it appears:
a configuration reader that defaults silently produces an engine that loads, streams,
decodes, and emits fluent text from the wrong architecture, with nothing to indicate it.

Both layouts are accepted, the released nested form (`text_config.*`) and the flat form
used by the test fixtures.

Layer roles:

```c
k3_is_mla(&cfg, layer)     /* Gated MLA layer          */
k3_is_kda(&cfg, layer)     /* Kimi Delta Attention     */
k3_is_dense(&cfg, layer)   /* the single dense layer   */
```

`full_attn` holds ONE-BASED layer indices, matching the checkpoint's own convention.

## Scratch memory

Every kernel takes caller-provided scratch. Sizes come from the engine, not from your
own arithmetic, recomputing them by hand is the easiest way to overrun a buffer
silently.

```c
size_t n = k3_layer_scratch(&cfg, n_tokens);
float *scratch = malloc(n * sizeof(float));
```

| function | covers |
|---|---|
| `k3_layer_scratch(cfg, T)` | a whole decoder layer |
| `k3_kda_scratch(cfg, T)` | a KDA layer |
| `k3_mla_scratch(cfg, T)` | an MLA layer, no KV cache |
| `k3_mla_scratch_cached(cfg, T, cap, mode)` | an MLA layer with a KV cache |
| `k3_moe_scratch(cfg)` | the MoE block |

## Weight structures

`K3KdaW`, `K3MlaW`, `K3MoeW` and `K3LayerW` describe where a layer's tensors live.

> **Zero every weight struct before filling it.** These hold function pointers
> (`K3MoeW::src`) and pointers whose NULL-ness selects a code path, dense versus MoE,
> gated versus ungated MLA. An uninitialised stack struct does not merely read the wrong
> weights; it can jump to an arbitrary address.

```c
K3MoeW moe;
memset(&moe, 0, sizeof moe);   /* required, not defensive */
```

Weight matrices are tagged pointers: `K3_WF32` (0) or `K3_WBF16` (1). Because `K3_WF32`
is zero, a `memset` struct defaults to fp32. Dispatch through `k3_mmw()` rather than
calling the typed matmuls directly.

## Running a layer

```c
k3_decoder_layer(hidden, block_residual, &n_blocks, &weights, &cfg,
                 layer_idx, n_tokens, kda_state, scratch);
```

`k3_decoder_layer_inc()` is the incremental form: MLA attends over a KV cache of earlier
positions and appends its own. KDA needs nothing carried, it updates its recurrent state
in place, and the Attention-Residual block stack is per token.

Both must produce identical tokens. The test suite asserts this rather than assuming it.

## Streaming experts

Provide a `K3ExpertSrc` and the MoE block will fetch experts on demand instead of reading
a resident bank:

```c
typedef struct K3ExpertSrc {
    int (*get)(struct K3ExpertSrc *, int layer, int expert, K3ExpertQ *out);
    int (*getmany)(struct K3ExpertSrc *, int layer, const int *experts, int n);
    void *ctx;
} K3ExpertSrc;
```

- `get` must keep the returned pointers valid until the caller finishes the token.
- `getmany` is an optional batch hint. It may be NULL, and callers must cope, falling
  back to `get` alone is always correct, only slower. It exists because issuing the whole
  top-k at once lets the reads overlap; serial `get` calls give the device a queue depth
  of one, which most NVMe hardware needs depth to saturate.

Experts stay in packed MXFP4 throughout. `k3_matmul_mxfp4` consumes nibbles directly and
never materialises a dequantised matrix, one expert is 17.5 MB packed against 132 MB
expanded, and a token touches 1,472 of them.

## Error handling

Two failure modes need explicit attention from callers.

**`k3_expert_drops`** is a global counter incremented whenever a streamed expert could
not be loaded. Non-zero means some token was computed with part of its routed
contribution missing, silent numerical corruption. The run completes and prints a
plausible token.

```c
if (k3_expert_drops) {
    fprintf(stderr, "%ld experts failed to load; output is corrupt\n", k3_expert_drops);
    return 1;   /* fail the run; do not report success */
}
```

**Configuration load failure** must abort, as above. There is no safe partial state.

## Thread safety

The kernels are reentrant and parallelise internally with OpenMP. They hold no global
state except `k3_expert_drops`.

The cache, the trunk reader, and the safetensors index are **not** thread-safe. One
inference at a time per instance.

## Minimal example

```c
K3Cfg cfg; int fa[128];
if (!k3_cfg_load_file(&cfg, fa, 128, "model/config.json")) return 1;

float *scratch = malloc(k3_layer_scratch(&cfg, T) * sizeof(float));

for (int L = 0; L < cfg.n_layers; L++)
    k3_decoder_layer(h, block_res, &nblocks, &layer_w[L], &cfg, L, T, kstate, scratch);

if (k3_expert_drops) return 1;   /* check before trusting the output */
```

See `src/cli/k3_run.c` for the complete path, including trunk streaming and the KV cache.

### Access-trace memory

The library retains its default request tracing. Call
`k3_cache_set_trace(&cache, 0)` to discard and disable the growing request buffer;
the bounded histogram and cache serving remain active. Call with `1` to record
subsequent requests again. The CLI enables request tracing only when
`--dump-cache-trace` is supplied, so ordinary inference does not retain this
unbounded diagnostic history.

Change trace collection only between cache requests; the toggle is not a
synchronization primitive for concurrent cache use. Rebuild consumers when the
public cache structure changes.
