#include "k3_android.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#ifndef POSIX_FADV_DONTNEED
#define POSIX_FADV_DONTNEED 4
#endif

static double clampd(double v, double lo, double hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static void set_reason(K3HandsetPlan *o, const char *s)
{
    snprintf(o->reason, sizeof o->reason, "%s", s);
}

void k3_android_plan_from(const K3AndroidFacts *in, K3HandsetPlan *out)
{
    memset(out, 0, sizeof *out);
    out->ultra = 1;

    const double avail = in->avail_gib > 0 ? in->avail_gib : in->total_gib;
    double ceiling = avail;
    if (in->cgroup_high_gib > 0 && in->cgroup_high_gib < ceiling)
        ceiling = in->cgroup_high_gib;
    ceiling -= in->reserve_gib > 0 ? in->reserve_gib : 0;
    if (ceiling < 0) ceiling = 0;
    out->ceiling_gib = ceiling;

    /* Reduction 3. Unpinned, little cores still exist; the caller applies a
     * straggler penalty. The plan's thread count is what we would pass. */
    if (in->pin_big && in->big_cores > 0)
        out->threads = in->big_cores;
    else {
        const int all = in->big_cores + in->little_cores;
        out->threads = all > 0 ? all : 1;
    }

    /* Reduction 2. */
    if (!in->handset_budget) {
        out->trunk_gb = 3.0;
        out->cache_gb = 1.0;
        out->ultra = 0;
        out->peak_gib = K3_ANDROID_LAPTOP_RSS_GIB;
    } else {
        const double slack = ceiling * 0.92 - K3_ANDROID_FLOOR_GIB;
        if (slack < 0.15) {
            out->process_ok = 0;
            out->peak_gib = K3_ANDROID_FLOOR_GIB;
            set_reason(out,
                "ceiling is under the 2.57 GiB streaming floor measured on "
                "the Orin Nano Super, plus a little pin");
        } else {
            double cache = clampd(slack * 0.12, 0.08, K3_ANDROID_CACHE_CAP_GB);
            double trunk = slack - cache;
            if (trunk > K3_ANDROID_TRUNK_CAP_GB) trunk = K3_ANDROID_TRUNK_CAP_GB;
            if (trunk < 0.05) {
                out->process_ok = 0;
                set_reason(out, "no room to pin even a sliver of trunk");
            } else {
                out->trunk_gb = trunk;
                out->cache_gb = cache;
                out->peak_gib = K3_ANDROID_FLOOR_GIB + trunk + cache;
                out->process_ok = out->peak_gib <= ceiling + 1e-9;
                out->ultra = 1;
            }
        }
    }

    if (in->handset_budget && out->process_ok && out->peak_gib > ceiling) {
        out->process_ok = 0;
        set_reason(out, "handset peak exceeds the Android ceiling");
    }
    if (!in->handset_budget) {
        out->process_ok = out->peak_gib <= ceiling + 1e-9;
        if (!out->process_ok)
            set_reason(out,
                "laptop preset peaks at 8.24 GB; this ceiling cannot hold it");
    }

    /* Reduction 4. An object store wins when asked: the phone stores none of
     * the checkpoint, and there is no page cache to drop. Otherwise prefer a
     * split when internal flash can hold the trunk and the USB volume holds
     * the shards — trunk traffic is sequential and UFS is usually faster. */
    const int64_t trunk = K3_ANDROID_TRUNK_BYTES;
    const int64_t shards = K3_ANDROID_CHECKPOINT_BYTES;
    const int64_t both = K3_ANDROID_NEED_BYTES;

    if (in->object_stream) {
        out->home = K3_HOME_OBJECT;
        out->weights_ok = 1;
        out->direct_io = 0;
    } else if (in->external_free >= both) {
        out->home = K3_HOME_EXTERNAL;
        out->weights_ok = 1;
        out->direct_io = in->external_direct ? 1 : 0;
    } else if (in->external_free >= shards && in->internal_free >= trunk) {
        out->home = K3_HOME_SPLIT;
        out->weights_ok = 1;
        /* O_DIRECT only if BOTH homes accept it; the fallback is global. */
        out->direct_io = (in->external_direct && in->direct_io) ? 1 : 0;
    } else if (in->internal_free >= both) {
        out->home = K3_HOME_INTERNAL;
        out->weights_ok = 1;
        out->direct_io = in->direct_io ? 1 : 0;
    } else {
        out->home = K3_HOME_NONE;
        out->weights_ok = 0;
        out->direct_io = 0;
    }

    /* Reduction 1. Object ranges never enter the page cache. A file read
     * that cannot use O_DIRECT has to drop what it just read. */
    if (out->home == K3_HOME_OBJECT) {
        out->drop_cache = 0;
    } else {
        out->drop_cache = (!out->direct_io && in->drop_cache) ? 1 : 0;
        if (out->direct_io) out->drop_cache = 0;
        if (!in->drop_cache && !out->direct_io && out->weights_ok && out->process_ok) {
            /* A 2.4 GB layer left resident in the cache is enough to cross the
             * killer even when the process RSS itself fits. */
            out->process_ok = 0;
            set_reason(out,
                "O_DIRECT unavailable and page-cache drop is off; "
                "a trunk layer can sit in cache and trip the low-memory killer");
        }
    }

    out->ok = out->process_ok && out->weights_ok;

    if (out->ok) {
        const char *home =
            out->home == K3_HOME_OBJECT ? "HTTP Range from an object store" :
            out->home == K3_HOME_SPLIT ? "trunk on UFS, shards on USB" :
            out->home == K3_HOME_EXTERNAL ? "shards and trunk on USB" :
            "internal flash (unusual)";
        const char *io = out->home == K3_HOME_OBJECT ? "streamed, then dropped" :
                         out->direct_io ? "O_DIRECT" :
                         out->drop_cache ? "buffered + DONTNEED" : "buffered";
        snprintf(out->reason, sizeof out->reason,
                 "peak %.2f GiB under a %.2f GiB ceiling, %d thread%s, %s, %s",
                 out->peak_gib, out->ceiling_gib, out->threads,
                 out->threads == 1 ? "" : "s", io, home);
    } else if (!out->weights_ok && out->reason[0] == '\0') {
        set_reason(out,
            "need 1.67 TB (1.561 TB shards + 109 GB trunk); "
            "no shipping phone holds that internally");
    } else if (!out->process_ok && out->reason[0] == '\0') {
        set_reason(out, "process budget does not fit this ceiling");
    }
}

static int64_t div_up(int64_t n, int64_t d)
{
    if (n <= 0) return 0;
    return (n + d - 1) / d;
}

void k3_android_ranges(const K3HandsetPlan *plan, int64_t part, K3RangePlan *out)
{
    memset(out, 0, sizeof *out);
    if (part <= 0) part = K3_ANDROID_PART_S3;
    out->part_bytes = part;

    int64_t pinned = (int64_t)(plan->trunk_gb * 1000000000.0);
    if (pinned < 0) pinned = 0;
    if (pinned > K3_ANDROID_TRUNK_BYTES) pinned = K3_ANDROID_TRUNK_BYTES;
    out->trunk_bytes = K3_ANDROID_TRUNK_BYTES - pinned;
    out->expert_bytes = K3_ANDROID_EXPERT_TOUCH_BYTES;
    out->head_bytes = plan->ultra ? K3_ANDROID_LM_HEAD_BYTES : 0;

    out->trunk_ranges = div_up(out->trunk_bytes, part);
    out->expert_ranges = div_up(out->expert_bytes, part);
    out->head_ranges = div_up(out->head_bytes, part);
    out->ranges = out->trunk_ranges + out->expert_ranges + out->head_ranges;
}

int k3_android_parse_mem_high(const char *text, double *out_bytes)
{
    if (!text || !out_bytes) return 0;
    while (*text == ' ' || *text == '\t' || *text == '\n') text++;
    if (*text == '\0' || strncmp(text, "max", 3) == 0) return 0;
    char *end = NULL;
    const double v = strtod(text, &end);
    if (end == text || v < 0) return 0;
    if (v > 1e30) return 0; /* "max" already handled; reject overflow junk */
    *out_bytes = v;
    return 1;
}

int k3_android_pick_big(const int *cap, int n, int *big_idx, int max_out)
{
    if (!cap || n <= 0 || !big_idx || max_out <= 0) return 0;
    int mx = 0;
    for (int i = 0; i < n; i++)
        if (cap[i] > mx) mx = cap[i];
    if (mx <= 0) return 0;
    const int cut = (mx * 70) / 100;
    int w = 0;
    for (int i = 0; i < n && w < max_out; i++)
        if (cap[i] >= cut) big_idx[w++] = i;
    return w;
}

int k3_android_direct_ok(const char *path)
{
#if defined(_WIN32)
    (void)path;
    return 0;
#else
    if (!path) return 0;
#ifdef O_DIRECT
    const int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd >= 0) {
        close(fd);
        return 1;
    }
    return 0;
#else
    (void)path;
    return 0;
#endif
#endif
}

int k3_android_drop_cache(int fd, int64_t off, int64_t len)
{
#if defined(_WIN32) || !defined(__linux__) && !defined(__ANDROID__)
    (void)fd; (void)off; (void)len;
    return 0; /* advisory; nothing to do off Linux */
#else
    if (fd < 0 || len <= 0) return 0;
    /* Failure is not fatal. The read was already correct. */
    (void)posix_fadvise(fd, (off_t)off, (off_t)len, POSIX_FADV_DONTNEED);
    return 0;
#endif
}
