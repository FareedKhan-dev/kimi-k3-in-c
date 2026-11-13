/* test_sysmem.c - the available-memory probe must return a real number on every
 * platform the engine builds on.
 *
 * WHY THIS FILE EXISTS
 *   Two refusals in k3_run.c compare the memory plan against what the machine can
 *   actually hand out: the KV cache check and the whole-plan check before
 *   allocation. Both are written as "if (available > 0 && need > available)", so a
 *   probe that returns 0 does not fail loudly, it silently disables both guards, and
 *   --preset auto refuses outright. Before src/cli/k3_sysmem.h the probe was
 *   /proc/meminfo behind #ifdef _WIN32, which is exactly 0 on macOS. This test fails
 *   on any platform where the number is 0 or impossible.
 *
 * usage: test_sysmem
 */
#define _POSIX_C_SOURCE 200809L
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3_sysmem.h"

static int g_fail = 0;

static void ck(int ok, const char *what, const char *detail)
{
    printf("  %s  %-40s %s\n", ok ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!ok) g_fail++;
}

int main(void)
{
    K3MemAvail m;
    const double avail = k3_mem_probe(&m);
    const double total = k3_mem_total_bytes();
    char d[96];

    printf("system memory probe\n");
    snprintf(d, sizeof d, "%.2f GB", total / 1e9);
    ck(total > 0.0, "total is known", d);
    snprintf(d, sizeof d, "%.2f GB", avail / 1e9);
    ck(avail > 0.0, "available is known (0 disables two refusals)", d);
    ck(avail <= total, "available does not exceed total", NULL);
    /* Any real machine that can run this test has more than this to spare; a probe
     * that returns a page count where bytes were wanted lands far below it. */
    ck(avail >= 64.0 * 1024 * 1024, "available is at least 64 MB", d);

#if !defined(_WIN32) && !defined(__APPLE__)
    {
        /* Linux: the probe's host term must agree with an independent read of the same
         * source. Memory moves between two reads, so the bound is loose, but a wrong
         * field or a wrong unit is off by far more than this. The figure itself is
         * compared with the host term, not with /proc/meminfo: a cgroup limit caps it,
         * and inside a container or a `systemd-run -p MemoryMax=` scope the two differ
         * by as much as the cap is below the host. */
        FILE *f = fopen("/proc/meminfo", "r");
        double kb_avail = 0.0, kb_total = 0.0;
        char line[256];
        while (f && fgets(line, sizeof line, f)) {
            unsigned long long v;
            if (sscanf(line, "MemAvailable: %llu kB", &v) == 1) kb_avail = (double)v;
            if (sscanf(line, "MemTotal: %llu kB", &v) == 1)     kb_total = (double)v;
        }
        if (f) fclose(f);
        const double ref_avail = kb_avail * 1024.0, ref_total = kb_total * 1024.0;
        snprintf(d, sizeof d, "probe %.3f GB, sscanf %.3f GB", m.host / 1e9, ref_avail / 1e9);
        ck(ref_avail > 0.0 && m.host > 0.8 * ref_avail && m.host < 1.25 * ref_avail,
           "host figure agrees with /proc/meminfo", d);
        ck(avail <= m.host, "available does not exceed the host figure", NULL);
        ck(ref_total > 0.0 && total == ref_total, "total equals MemTotal", NULL);
    }
#endif

    printf("\n%s\n", g_fail ? "SYSMEM TESTS FAILED" : "SYSMEM TESTS PASSED");
    return g_fail ? 1 : 0;
}
