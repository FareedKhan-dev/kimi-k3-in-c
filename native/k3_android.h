/* k3_android.h — four reductions that take kimi-k3-in-c from a Linux laptop
 * onto an arm64 Android handset.
 *
 * Upstream (FareedKhan-dev/kimi-k3-in-c, Apache-2.0) already runs the official
 * 2.78T checkpoint in about 8 GB on x86-64 and has been proven on an 8 GB-class
 * Jetson Orin Nano Super at 2.57 GiB RSS. Android is not that machine:
 *
 *   1. Page cache. O_DIRECT fails on exFAT and on the USB stack most phones
 *      actually mount. Without an explicit drop, buffered trunk reads sit in
 *      the cache and the low-memory killer treats the process as the cost.
 *   2. The ceiling is not MemTotal. SurfaceFlinger, system_server and the
 *      killer need a reserve, and an app cgroup may cap the rest.
 *   3. big.LITTLE. OpenMP static schedules wait on the slowest core. Little
 *      cores must not take a lane.
 *   4. The checkpoint does not fit in internal flash. Shards are
 *      1,560,936,091,448 bytes and the packed trunk is another
 *      108,811,952,128. The bytes stay elsewhere and are read by range:
 *      a USB-C SSD, or an object store (S3, R2, B2, or a gateway that
 *      already turned Telegram file parts into Range GETs). The phone
 *      multiplies a slice and drops it. Raw Telegram parts are the same
 *      idea with a 1 MiB ceiling, which means many more round trips.
 *
 * None of this changes a weight, a route, or a logit. It decides where bytes
 * are allowed to live and which cores may touch them.
 *
 * Weights remain under Moonshot AI's license. This file is Apache-2.0,
 * matching the engine it plugs into.
 */
#ifndef K3_ANDROID_H
#define K3_ANDROID_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Official sizes. Do not "round" these; the fit check is exact. */
#define K3_ANDROID_CHECKPOINT_BYTES ((int64_t)1560936091448LL)
#define K3_ANDROID_TRUNK_BYTES      ((int64_t)108811952128LL)
#define K3_ANDROID_NEED_BYTES       (K3_ANDROID_CHECKPOINT_BYTES + K3_ANDROID_TRUNK_BYTES)

/* Jetson Orin Nano Super, four runs, streaming tables, peak RSS. */
#define K3_ANDROID_FLOOR_GIB 2.573

/* Upstream ultra preset caps. Expert cache below ~36 GB does not pay. */
#define K3_ANDROID_TRUNK_CAP_GB 2.5
#define K3_ANDROID_CACHE_CAP_GB 0.31

/* One routed expert, and how much of the checkpoint a token re-reads when
 * nothing is cached. Expert cache below ~36 GB does not change the second
 * number; the router was trained flat. */
#define K3_ANDROID_EXPERT_BYTES       ((int64_t)17547264LL)
#define K3_ANDROID_EXPERT_TOUCH_BYTES ((int64_t)99703554048LL)
#define K3_ANDROID_LM_HEAD_BYTES      ((int64_t)2348810240LL)

/* Range size. S3 will accept a multi-megabyte GET. Telegram's own
 * upload.getFile will not; a gateway that speaks S3 is what removes that cap. */
#define K3_ANDROID_PART_S3       (8LL << 20)
#define K3_ANDROID_PART_TELEGRAM (1LL << 20)

/* Laptop preset peak, from the upstream README. The handset plan exists
 * because this number is how Android kills the process. */
#define K3_ANDROID_LAPTOP_RSS_GIB 8.24

typedef struct {
    double   total_gib;       /* physical RAM */
    double   avail_gib;       /* MemAvailable, or a planning estimate */
    double   cgroup_high_gib; /* 0 = no cgroup cap */
    double   reserve_gib;     /* left for the rest of Android */
    int      big_cores;
    int      little_cores;
    int      direct_io;       /* 1 if the weight filesystem accepted O_DIRECT */
    int      drop_cache;      /* reduction 1 enabled */
    int      handset_budget;  /* reduction 2 enabled; 0 = laptop preset */
    int      pin_big;         /* reduction 3 enabled */
    int64_t  internal_free;   /* bytes free on UFS */
    int64_t  external_free;   /* bytes free on the USB volume; 0 = no disk */
    int      external_direct; /* 1 if that volume accepted O_DIRECT */
    int      object_stream;  /* 1 = weights live in an object store, Range GET */
} K3AndroidFacts;

typedef enum {
    K3_HOME_NONE = 0,
    K3_HOME_INTERNAL,
    K3_HOME_EXTERNAL,
    K3_HOME_SPLIT,         /* trunk on internal UFS, shards on USB */
    K3_HOME_OBJECT         /* HTTP Range / S3 / Telegram-S3 gateway */
} K3WeightHome;

typedef struct {
    int      ok;             /* 1 = process and weights both have a home */
    int      process_ok;     /* RSS plan fits the ceiling */
    int      weights_ok;
    double   ceiling_gib;
    double   trunk_gb;
    double   cache_gb;
    double   peak_gib;
    int      ultra;          /* always 1 when ok and handset_budget */
    int      threads;
    int      direct_io;      /* effective: use O_DIRECT, else buffered+drop */
    int      drop_cache;     /* effective */
    K3WeightHome home;
    char     reason[320];
} K3HandsetPlan;

/* How many Range GETs one token issues at a given part size.
 * trunk_bytes is the unpinned remainder. head_bytes is 0 unless ultra
 * streaming is on (the laptop preset keeps lm_head resident). */
typedef struct {
    int64_t part_bytes;
    int64_t trunk_bytes;
    int64_t expert_bytes;
    int64_t head_bytes;
    int64_t trunk_ranges;
    int64_t expert_ranges;
    int64_t head_ranges;
    int64_t ranges;
} K3RangePlan;

/* Pure. No syscalls. The on-device probes only fill K3AndroidFacts. */
void k3_android_plan_from(const K3AndroidFacts *in, K3HandsetPlan *out);
void k3_android_ranges(const K3HandsetPlan *plan, int64_t part, K3RangePlan *out);

/* "max", empty, or a byte count. Returns 1 and writes bytes when the cgroup
 * states a finite cap. Returns 0 for "max" / empty / garbage (no cap). */
int k3_android_parse_mem_high(const char *text, double *out_bytes);

/* cap[i] is cpu_capacity or cpuinfo_max_freq. Writes big-core indexes.
 * A core is big when its value is >= 70% of the maximum. 80% drops the
 * mid cluster on 1+5+2 SoCs (those mids sit near 75% of the prime).
 * Little cores are far below either cut. Returns the count. */
int k3_android_pick_big(const int *cap, int n, int *big_idx, int max_out);

/* Reduction 1, for real file descriptors. Both return 0 on success.
 * direct_ok is 1 only when open(O_DIRECT) actually succeeds — EINVAL on
 * exFAT/FUSE is a normal answer, not an error. */
int k3_android_direct_ok(const char *path);
int k3_android_drop_cache(int fd, int64_t off, int64_t len);

#ifdef __cplusplus
}
#endif

#endif /* K3_ANDROID_H */
