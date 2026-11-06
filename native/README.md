# Handset layer

Four reductions between the Linux engine in this repo and an arm64 Android
phone. They do not change a weight, a route, or a logit. Step by step for a
Codespace and for Termux: [../HANDSET.md](../HANDSET.md).

| # | Reduction | What changes | What does not |
|---|-----------|--------------|---------------|
| 1 | Page-cache drop | `posix_fadvise(DONTNEED)` after every buffered read when `O_DIRECT` returns `EINVAL` (exFAT, FUSE, most USB mounts) | the bytes that were multiplied |
| 2 | Handset budget | RSS stays near the 2.57 GiB floor. Laptop preset (8.24 GB) is refused. Embedding and `lm_head` stay streamed | layer count, expert count, precision |
| 3 | Big-core pin | `sched_setaffinity` to CPUs at ≥ 70% of max `cpu_capacity` | summation order |
| 4 | Weight home | 1.67 TB lives on a USB-C SSD, or is read with HTTP Range | the checkpoint |

```sh
make test     # planner. No checkpoint.
make stream   # three Range GETs match the file.
```

Upstream already proved the official checkpoint on a Jetson Orin Nano Super:
2.57 GiB RSS, token ` Paris`, 949 seconds on a hard disk. This directory does
not repeat that proof. `k3_stream_pread` is the stand-in for `pread` in
`src/io/k3_st.c` and `src/io/k3_trunk.c`. It is tested here. It is not yet
called by `bin/k3`.

Weights are Moonshot AI's. This layer is Apache-2.0, like the engine.
