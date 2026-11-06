#include "k3_android.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int g_fail = 0;

static void expect(int cond, const char *msg)
{
    if (!cond) {
        fprintf(stderr, "FAIL %s\n", msg);
        g_fail++;
    }
}

static void expect_near(double got, double want, double eps, const char *msg)
{
    if (fabs(got - want) > eps) {
        fprintf(stderr, "FAIL %s (got %.6f want %.6f)\n", msg, got, want);
        g_fail++;
    }
}

static K3AndroidFacts base(void)
{
    K3AndroidFacts f;
    memset(&f, 0, sizeof f);
    f.total_gib = 12;
    f.avail_gib = 7.2;
    f.reserve_gib = 1.15;
    f.big_cores = 8;
    f.little_cores = 0;
    f.direct_io = 0;
    f.drop_cache = 1;
    f.handset_budget = 1;
    f.pin_big = 1;
    f.internal_free = 80LL << 30;
    f.external_free = 2LL << 40; /* 2 TiB */
    f.external_direct = 0;
    return f;
}

int main(void)
{
    /* 12 GB class, USB disk, handset budget. */
    {
        K3AndroidFacts f = base();
        K3HandsetPlan p;
        k3_android_plan_from(&f, &p);
        const double ceiling = 7.2 - 1.15;
        const double slack = ceiling * 0.92 - 2.573;
        const double cache = slack * 0.12 > 0.31 ? 0.31 : slack * 0.12;
        const double trunk = 2.5; /* slack - cache is well above the cap */
        expect(p.ok, "12 GB handset should run");
        expect(p.ultra == 1, "ultra streaming stays on");
        expect(p.threads == 8, "pin to 8 big cores");
        expect(p.home == K3_HOME_EXTERNAL, "2 TiB external holds both");
        expect(p.drop_cache == 1, "exFAT path drops the cache");
        expect(p.direct_io == 0, "exFAT is not O_DIRECT");
        expect_near(p.ceiling_gib, ceiling, 1e-9, "ceiling");
        expect_near(p.trunk_gb, trunk, 1e-9, "trunk cap");
        expect_near(p.cache_gb, cache, 1e-6, "cache");
        expect_near(p.peak_gib, 2.573 + trunk + cache, 1e-6, "peak");
        expect(p.peak_gib < K3_ANDROID_LAPTOP_RSS_GIB, "under the laptop peak");
    }

    /* Same phone, laptop preset: the killer's favourite mistake. */
    {
        K3AndroidFacts f = base();
        f.handset_budget = 0;
        K3HandsetPlan p;
        k3_android_plan_from(&f, &p);
        expect(!p.ok, "laptop preset must refuse");
        expect(!p.process_ok, "8.24 GB does not fit a 6.05 GiB ceiling");
        expect(p.weights_ok, "weights were fine; the preset was not");
        expect_near(p.peak_gib, 8.24, 1e-9, "laptop peak");
    }

    /* 8 GB class. Tight, but the Jetson floor still fits. */
    {
        K3AndroidFacts f = base();
        f.total_gib = 8;
        f.avail_gib = 4.2;
        f.big_cores = 4;
        f.little_cores = 4;
        K3HandsetPlan p;
        k3_android_plan_from(&f, &p);
        expect(p.ok, "8 GB handset with USB should still run");
        expect(p.threads == 4, "little cores stay off the plan");
        expect(p.peak_gib <= p.ceiling_gib, "peak under ceiling");
        expect(p.peak_gib < 3.2, "stays near the measured floor");
    }

    /* No USB, 1 TB internal: still short. */
    {
        K3AndroidFacts f = base();
        f.internal_free = 900LL << 30;
        f.external_free = 0;
        K3HandsetPlan p;
        k3_android_plan_from(&f, &p);
        expect(!p.weights_ok, "900 GiB cannot hold 1.67 TB");
        expect(!p.ok, "no weight home");
        expect(p.home == K3_HOME_NONE, "no home");
    }

    /* Split: USB holds shards, UFS holds the trunk. */
    {
        K3AndroidFacts f = base();
        f.external_free = K3_ANDROID_CHECKPOINT_BYTES + (1LL << 30);
        f.internal_free = K3_ANDROID_TRUNK_BYTES + (1LL << 30);
        f.direct_io = 1;
        f.external_direct = 0;
        K3HandsetPlan p;
        k3_android_plan_from(&f, &p);
        expect(p.weights_ok, "split fits");
        expect(p.home == K3_HOME_SPLIT, "split home");
        expect(p.direct_io == 0, "split requires both sides direct");
        expect(p.drop_cache == 1, "so the drop stays on");
    }

    /* Drop disabled and no O_DIRECT: refuse even when RAM fits. */
    {
        K3AndroidFacts f = base();
        f.drop_cache = 0;
        K3HandsetPlan p;
        k3_android_plan_from(&f, &p);
        expect(p.weights_ok, "weights ok");
        expect(!p.process_ok, "page cache hazard refuses");
        expect(!p.ok, "not ok");
    }

    /* O_DIRECT means the drop is unnecessary. */
    {
        K3AndroidFacts f = base();
        f.external_direct = 1;
        f.drop_cache = 0;
        K3HandsetPlan p;
        k3_android_plan_from(&f, &p);
        expect(p.ok, "O_DIRECT does not need the drop");
        expect(p.direct_io == 1, "direct on");
        expect(p.drop_cache == 0, "drop off");
    }

    /* cgroup cap tighter than MemAvailable, still above the floor. */
    {
        K3AndroidFacts f = base();
        f.cgroup_high_gib = 5.0;
        K3HandsetPlan p;
        k3_android_plan_from(&f, &p);
        expect_near(p.ceiling_gib, 5.0 - 1.15, 1e-9, "cgroup ceiling");
        expect(p.ok, "still runs inside the cgroup");
        expect(p.peak_gib <= p.ceiling_gib, "under cgroup");
    }

    /* memory.high parser. */
    {
        double b = -1;
        expect(k3_android_parse_mem_high("max\n", &b) == 0, "max is no cap");
        expect(k3_android_parse_mem_high("  ", &b) == 0, "blank is no cap");
        expect(k3_android_parse_mem_high("3221225472\n", &b) == 1, "bytes");
        expect_near(b, 3221225472.0, 1.0, "3 GiB high");
        expect(k3_android_parse_mem_high("nope", &b) == 0, "garbage");
    }

    /* big.LITTLE picker. Capacities like a Tensor-class 1+5+2. */
    {
        const int cap[] = { 1024, 800, 800, 800, 780, 760, 250, 240 };
        int idx[8];
        const int n = k3_android_pick_big(cap, 8, idx, 8);
        expect(n == 6, "six big cores");
        expect(idx[0] == 0 && idx[5] == 5, "indexes are the fast cluster");
    }

    /* Object store, no local copy of the checkpoint, drop not required. */
    {
        K3AndroidFacts f = base();
        f.external_free = 0;
        f.internal_free = 40LL << 30;
        f.drop_cache = 0;
        f.object_stream = 1;
        K3HandsetPlan p;
        k3_android_plan_from(&f, &p);
        expect(p.ok, "S3 stream runs without a disk and without a page cache");
        expect(p.home == K3_HOME_OBJECT, "object home");
        expect(p.direct_io == 0 && p.drop_cache == 0, "ranges are not files");
        expect(p.weights_ok && p.process_ok, "both sides ok");

        K3RangePlan r;
        k3_android_ranges(&p, K3_ANDROID_PART_S3, &r);
        const int64_t pinned = (int64_t)(p.trunk_gb * 1000000000.0);
        const int64_t moved = K3_ANDROID_TRUNK_BYTES - pinned;
        const int64_t part = K3_ANDROID_PART_S3;
        expect(r.trunk_bytes == moved, "unpinned trunk");
        expect(r.trunk_ranges == (moved + part - 1) / part, "trunk ranges");
        expect(r.expert_ranges == (K3_ANDROID_EXPERT_TOUCH_BYTES + part - 1) / part,
               "expert ranges");
        expect(r.head_bytes == K3_ANDROID_LM_HEAD_BYTES, "ultra re-reads lm_head");
        expect(r.ranges == r.trunk_ranges + r.expert_ranges + r.head_ranges, "sum");

        k3_android_ranges(&p, K3_ANDROID_PART_TELEGRAM, &r);
        expect(r.ranges > 100000, "1 MiB Telegram parts explode the round trips");
    }
    {
        K3AndroidFacts f = base();
        f.big_cores = 6;
        f.little_cores = 2;
        f.pin_big = 0;
        K3HandsetPlan p;
        k3_android_plan_from(&f, &p);
        expect(p.threads == 8, "unpinned plan uses every core");
    }

    /* Direct-I/O probe and drop must not crash on a real file. Either answer
     * from the probe is legal; tmpfs often refuses O_DIRECT. */
    {
        const char *path = "/tmp/k3-android-probe.bin";
        FILE *fp = fopen(path, "wb");
        expect(fp != NULL, "probe file");
        if (fp) {
            char buf[4096];
            memset(buf, 0xAB, sizeof buf);
            fwrite(buf, 1, sizeof buf, fp);
            fclose(fp);
        }
        const int direct = k3_android_direct_ok(path);
        expect(direct == 0 || direct == 1, "probe returns a bool");
        const int fd = open(path, O_RDONLY);
        if (fd >= 0) {
            expect(k3_android_drop_cache(fd, 0, 4096) == 0, "drop on real fd");
            close(fd);
        }
        /* drop_cache on an invalid fd is a no-op success. */
        expect(k3_android_drop_cache(-1, 0, 4096) == 0, "drop on bad fd");
        expect(k3_android_direct_ok(NULL) == 0, "null path");
        expect(k3_android_direct_ok("/no/such/k3/path") == 0, "missing path");
    }

    if (g_fail) {
        fprintf(stderr, "%d failure(s)\n", g_fail);
        return 1;
    }
    printf("k3_android: all checks passed\n");
    return 0;
}
