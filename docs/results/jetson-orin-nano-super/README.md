# Jetson Orin Nano Super proof of life

This is a four-run reproduction of one-token inference with the complete official
Kimi K3 text checkpoint on an 8 GB-class Jetson. It demonstrates completion under a
tight RAM budget; it is not an interactive-performance claim.

## Result

Input: `The capital of France is`

| run | token ID | decoded token | layers | expert drops | peak RSS | inference time |
|---:|---:|---|---:|---:|---:|---:|
| 1 | `17374` | ` Paris` | 93/93 | 0 | 2,760,306,688 bytes | 948.5179 s |
| 2 | `17374` | ` Paris` | 93/93 | 0 | 2,759,565,312 bytes | 949.8126 s |
| 3 | `17374` | ` Paris` | 93/93 | 0 | 2,762,563,584 bytes | 950.6945 s |
| 4 | `17374` | ` Paris` | 93/93 | 0 | 2,761,093,120 bytes | 948.7180 s |

All four runs passed every structural gate and matched the recorded greedy reference.
Their complete float32 logit dumps also had the same SHA-256, so the agreement was not
limited to argmax. Mean inference time was 949.4358 seconds (15m 49.4s); maximum peak RSS
was 2.573 GiB.

The proof runner treats token ID `17374`, decoded token ` Paris`, 93/93 completed layers
and zero expert drops as hard acceptance gates. Its case-insensitive `Paris` check is
reported only as an additional semantic diagnostic and cannot turn a mismatch into PASS.

Each run read exactly:

- 108,811,952,128 bytes of packed dense trunk from the internal NVMe;
- 99,703,554,048 bytes of routed experts from the external rotating HDD;
- 71,680 bytes of selected embedding rows;
- 2,348,810,240 bytes of chunked lm_head data.

The checkpoint kept its official BF16 and MXFP4 representations. The path uses the
original Top-16 routing and all official weights; it does not drop experts, layers, or
precision. Embedding rows and lm_head chunks use the same resident kernels after an
aligned read, and the full-recompute recurrent state reuses one slot only after the
previous layer has finished consuming it.

## Machine and storage

- NVIDIA Jetson Orin Nano Super, six visible CPU cores;
- Ubuntu 22.04.5 LTS, Jetson Linux R36.4.4, Linux 5.15.148-tegra, `aarch64`;
- scalar/ARM C path with OpenMP; no CUDA inference implementation;
- official checkpoint and routed experts on `/dev/sda2`, a rotating external HDD;
- packed 93-layer dense trunk on `/dev/nvme0n1p1`, the internal NVMe.

The expert loader used serial reads (`K3_NOPREFETCH=1`) to avoid multi-stream seeks on
the HDD. That changes I/O scheduling only; routing, weights, accumulation and output are
unchanged.

## Checkpoint identity

- repository: `moonshotai/Kimi-K3`;
- immutable revision: `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`;
- official safetensors shards: 96;
- exact safetensors bytes: 1,560,936,091,448;
- checksum verification completed: `2026-08-08T21:02:46+08:00`.

The verification marker is created only after `hf cache verify --fail-on-missing-files`
succeeds. The proof script also rejects a missing marker, a changed/newer shard, a wrong
repository or revision, the wrong shard count/byte total, or a trunk other than the exact
93-layer 108,811,952,128-byte manifest.

## Source provenance

This worktree starts from upstream v1.0.0 commit
`ff11dce858a2eb8a781224facdffd33a1fa48d25`. The Jetson executed a source bundle without
Git metadata. A sanitizer pass after the first three runs found a strict-alignment edge
case in synthetic BF16 data; the safeguard was added and a fourth complete run was
performed. The official embedding and lm_head offsets were already 4-byte aligned, and
all four logit hashes remained identical. The final runtime source manifest has SHA-256
`fb416123155a9d76a7559050a4b42ed2174f7e910b27b6911d17a76a6c978ea6`.
Before publication cleanup, all 293 captured files matched it byte-for-byte. That
manifest incidentally included one generated `tools/__pycache__` file. In the proposed
tree, 288 entries still match exactly; the four intentional differences are CI coverage,
README text, the proof runner's stricter exact-output gate, and line wrapping only in
`src/cli/k3_run.c`. The generated cache is omitted and the seven result files below are
new. All other runtime, build, fixture, download and packer files remain byte-identical
to the fourth run.

## Artifact hashes

The small result and acceptance JSON files are included here. Large logs and float32
logit dumps are retained on the experiment host rather than committed to Git.

```text
4197fa5554f2c97c82f24bb76481941d1d8252a1001388d8e458aded9c3145f8  result-run-1.json
ea5d5dd9bb5130a3b864aba8b6360359636b551ebab9ec5eedba33f13cadc22c  result-run-2.json
7a832b6ff39e43e88be2274231c3cbd1ce008883fa8c3cd2fc2e303e762dca62  result-run-3.json
f8106d0c6b1e1f586737f42727d586d8f5755056ea3a0c75e68d5390c164d6f6  result-run-4.json
d41c716a64a85ac6ecbad2ffb31f45e5fb0f998c4d003852345af49510acec16  acceptance.json
d5f6e2ef3d533cf1b80ba2cff4ee74cf4fc6d3011f80d7b64cf9703f38d5fe20  checkpoint-verification.txt
4518032e5cac0763caae4c6f23a026f4644bb3c1209779e53d21136354880d28  logits.f32 (all four runs)
8ef4776ab9e412821d00867b1de76c754b06d45c3ee70471cf73cadd2ae0c8dd  trunk.json
```

The official checkpoint, packed trunk, routed weights and logit dumps are not
redistributed by this repository.
