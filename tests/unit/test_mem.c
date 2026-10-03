/* test_mem.c - the memory probe must see a cgroup cap, not just MemAvailable.
 *
 * Companion to test_sysmem.c: that one checks the live probe returns a real number
 * on every platform; this one checks the Linux cgroup walk in k3_sysmem.h against
 * trees it builds itself, so it runs the same on every platform too.
 *
 * WHY THIS FILE EXISTS
 *   The admission check compares the memory plan against what the kernel says is
 *   available. tests/fixtures/gates/gates.txt records a run under
 *   `MemoryMax=12G` whose plan printed `available 65.56 GB`: /proc/meminfo is
 *   machine-wide and knows nothing about the cgroup the process sits in, so a
 *   capped run was admitted against memory it could never touch and the plan
 *   line was wrong by 5x. Every case below builds a fake /proc + /sys/fs/cgroup
 *   tree and asserts the probe reports the TIGHTEST bound, that an unlimited
 *   controller falls back to the host figure, that `unknown` is never conflated
 *   with 0, and that a control file which is present but unreadable fails
 *   CLOSED rather than reading as unlimited.
 *
 * usage: test_mem <work_dir>
 */
#define _POSIX_C_SOURCE 200809L

#include "k3_portable_io.h"   /* mkdir() shim for MinGW; see the header for why */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "k3_sysmem.h"

static int g_fail = 0;

static void ck(int ok, const char *what, const char *detail)
{
    printf("  %s  %-40s %s\n", ok ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!ok) g_fail++;
}

static int endswith(const char *s, const char *suf)
{
    size_t a = strlen(s), b = strlen(suf);
    return a >= b && strcmp(s + a - b, suf) == 0;
}

/* mkdir -p for a path made of our own components. */
static void mkdirs(const char *path)
{
    char buf[1024];
    snprintf(buf, sizeof buf, "%s", path);
    for (char *p = buf + 1; *p; p++) {
        if (*p == '/') { *p = 0; mkdir(buf, 0755); *p = '/'; }
    }
    mkdir(buf, 0755);
}

static int put(const char *dir, const char *name, const char *text)
{
    char p[1200];
    snprintf(p, sizeof p, "%s/%s", dir, name);
    FILE *f = fopen(p, "w");
    if (!f) return -1;
    fputs(text, f);
    fclose(f);
    return 0;
}

/* One case = one fresh subtree: <work>/<tag>/{meminfo,cgroup,sys}. */
typedef struct { char root[600], meminfo[700], cgroup[700], sys[700]; } Tree;

static void tree(Tree *t, const char *work, const char *tag, int with_meminfo)
{
    snprintf(t->root,    sizeof t->root,    "%s/%s", work, tag);
    snprintf(t->meminfo, sizeof t->meminfo, "%s/meminfo", t->root);
    snprintf(t->cgroup,  sizeof t->cgroup,  "%s/cgroup", t->root);
    snprintf(t->sys,     sizeof t->sys,     "%s/sys", t->root);
    mkdirs(t->sys);
    if (with_meminfo)
        put(t->root, "meminfo",
            "MemTotal:       98304000 kB\n"
            "MemFree:        12345678 kB\n"
            "MemAvailable:   68746240 kB\n"
            "Buffers:          123456 kB\n");
}

#define HOST_BYTES (68746240.0 * 1024.0)   /* 70.40 GB, the fake MemAvailable */

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: test_mem <work_dir>\n"); return 2; }
    const char *work = argv[1];
    mkdir(work, 0755);

    /* (a) host only: no cgroup membership file at all. Known, and not capped. */
    {
        Tree t; tree(&t, work, "a", 1);
        K3MemAvail m;
        int rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
        char d[2048]; snprintf(d, sizeof d, "bytes %.0f known %d cgroup %.0f", m.bytes, m.known, m.cgroup);
        ck(rc == 0 && m.known && !m.malformed && m.bytes == HOST_BYTES && m.cgroup < 0,
           "no cgroup: host MemAvailable, known", d);
    }

    /* (b) v2 leaf capped at 12 GiB with 1 GiB in use -> 11 GiB, tighter than host. */
    {
        Tree t; tree(&t, work, "b", 1);
        put(t.root, "cgroup", "0::/user.slice/run-1.scope\n");
        char leaf[900]; snprintf(leaf, sizeof leaf, "%s/user.slice/run-1.scope", t.sys);
        mkdirs(leaf);
        put(leaf, "memory.max", "12884901888\n");
        put(leaf, "memory.current", "1073741824\n");
        K3MemAvail m;
        int rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
        char d[2048]; snprintf(d, sizeof d, "bytes %.0f v%d at %s", m.bytes, m.cgroup_v, m.limit_path);
        ck(rc == 0 && m.known && !m.malformed && m.bytes == 11811160064.0
           && m.cgroup_v == 2 && endswith(m.limit_path, "/user.slice/run-1.scope/memory.max"),
           "v2 leaf cap: limit - current wins", d);
    }

    /* (c) v2 present but `max` everywhere: unlimited, host figure stands. */
    {
        Tree t; tree(&t, work, "c", 1);
        put(t.root, "cgroup", "0::/user.slice/run-2.scope\n");
        char leaf[900]; snprintf(leaf, sizeof leaf, "%s/user.slice/run-2.scope", t.sys);
        mkdirs(leaf);
        char mid[900]; snprintf(mid, sizeof mid, "%s/user.slice", t.sys);
        put(leaf, "memory.max", "max\n");     put(leaf, "memory.current", "5\n");
        put(mid,  "memory.max", "max\n");     put(mid,  "memory.current", "6\n");
        K3MemAvail m;
        int rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
        char d[2048]; snprintf(d, sizeof d, "bytes %.0f cgroup %.0f", m.bytes, m.cgroup);
        ck(rc == 0 && m.known && !m.malformed && m.bytes == HOST_BYTES && m.cgroup < 0,
           "v2 `max`: unlimited, host figure stands", d);
    }

    /* (d) nested: leaf allows 12 GiB but its parent only 8 GiB -> the parent binds. */
    {
        Tree t; tree(&t, work, "d", 1);
        put(t.root, "cgroup", "0::/user.slice/run-3.scope\n");
        char leaf[900]; snprintf(leaf, sizeof leaf, "%s/user.slice/run-3.scope", t.sys);
        mkdirs(leaf);
        char mid[900]; snprintf(mid, sizeof mid, "%s/user.slice", t.sys);
        put(leaf, "memory.max", "12884901888\n"); put(leaf, "memory.current", "1073741824\n");
        put(mid,  "memory.max", "8589934592\n");  put(mid,  "memory.current", "2147483648\n");
        K3MemAvail m;
        int rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
        char d[2048]; snprintf(d, sizeof d, "bytes %.0f at %s", m.bytes, m.limit_path);
        ck(rc == 0 && m.known && m.bytes == 6442450944.0
           && endswith(m.limit_path, "/user.slice/memory.max"),
           "v2 nested: tightest ancestor binds", d);
    }

    /* (e) leaf has no memory files (controller enabled only above it). */
    {
        Tree t; tree(&t, work, "e", 1);
        put(t.root, "cgroup", "0::/user.slice/run-4.scope\n");
        char leaf[900]; snprintf(leaf, sizeof leaf, "%s/user.slice/run-4.scope", t.sys);
        mkdirs(leaf);
        char mid[900]; snprintf(mid, sizeof mid, "%s/user.slice", t.sys);
        put(mid, "memory.max", "4294967296\n"); put(mid, "memory.current", "1073741824\n");
        K3MemAvail m;
        int rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
        char d[2048]; snprintf(d, sizeof d, "bytes %.0f at %s", m.bytes, m.limit_path);
        ck(rc == 0 && m.known && m.bytes == 3221225472.0
           && endswith(m.limit_path, "/user.slice/memory.max"),
           "v2 leaf without files: nearest ancestor", d);
    }

    /* (f) v1: no 0:: line, memory controller on a named hierarchy. */
    {
        Tree t; tree(&t, work, "f", 1);
        put(t.root, "cgroup", "5:cpu,cpuacct:/\n4:memory:/foo\n3:pids:/\n");
        char leaf[900]; snprintf(leaf, sizeof leaf, "%s/memory/foo", t.sys);
        mkdirs(leaf);
        put(leaf, "memory.limit_in_bytes", "2147483648\n");
        put(leaf, "memory.usage_in_bytes", "536870912\n");
        K3MemAvail m;
        int rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
        char d[2048]; snprintf(d, sizeof d, "bytes %.0f v%d at %s", m.bytes, m.cgroup_v, m.limit_path);
        ck(rc == 0 && m.known && m.bytes == 1610612736.0 && m.cgroup_v == 1
           && endswith(m.limit_path, "/memory/foo/memory.limit_in_bytes"),
           "v1 limit_in_bytes - usage_in_bytes", d);

        /* v1 unlimited is PAGE_COUNTER_MAX, not the word `max`. */
        put(leaf, "memory.limit_in_bytes", "9223372036854771712\n");
        rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
        snprintf(d, sizeof d, "bytes %.0f cgroup %.0f", m.bytes, m.cgroup);
        ck(rc == 0 && m.known && m.bytes == HOST_BYTES && m.cgroup < 0,
           "v1 PAGE_COUNTER_MAX: unlimited", d);
    }

    /* (g) a visible but garbled memory.max must fail CLOSED: 0, known, flagged. */
    {
        Tree t; tree(&t, work, "g", 1);
        put(t.root, "cgroup", "0::/run-5.scope\n");
        char leaf[900]; snprintf(leaf, sizeof leaf, "%s/run-5.scope", t.sys);
        mkdirs(leaf);
        put(leaf, "memory.max", "garbage\n"); put(leaf, "memory.current", "1\n");
        K3MemAvail m;
        int rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
        char d[2048]; snprintf(d, sizeof d, "bytes %.0f known %d malformed %d: %s", m.bytes, m.known, m.malformed, m.detail);
        ck(rc == 0 && m.known && m.malformed && m.bytes == 0.0 && m.detail[0],
           "malformed memory.max fails closed", d);
    }

    /* (h) hybrid host: a v1 line precedes the v2 line; the v2 path must still be
     * the one that is mapped (this box's /proc/self/cgroup starts `1:net_cls:/`). */
    {
        Tree t; tree(&t, work, "h", 1);
        put(t.root, "cgroup", "1:net_cls:/\n0::/user.slice/run-6.scope\n");
        char leaf[900]; snprintf(leaf, sizeof leaf, "%s/user.slice/run-6.scope", t.sys);
        mkdirs(leaf);
        put(leaf, "memory.max", "12884901888\n"); put(leaf, "memory.current", "0\n");
        K3MemAvail m;
        int rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
        char d[2048]; snprintf(d, sizeof d, "bytes %.0f at %s", m.bytes, m.limit_path);
        ck(rc == 0 && m.known && m.bytes == 12884901888.0 && m.cgroup_v == 2
           && endswith(m.limit_path, "/run-6.scope/memory.max"),
           "hybrid /proc/self/cgroup: 0:: line wins", d);
    }

    /* (i) nothing readable at all: unknown, which is NOT zero. */
    {
        Tree t; tree(&t, work, "i", 0);
        K3MemAvail m;
        int rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
        char d[2048]; snprintf(d, sizeof d, "rc %d known %d bytes %.0f", rc, m.known, m.bytes);
        ck(rc != 0 && !m.known && !m.malformed, "no source at all: unknown, not 0", d);
    }

    /* (j) a cgroup with 60 GiB of headroom must not RAISE the figure above host. */
    {
        Tree t; tree(&t, work, "j", 1);
        put(t.root, "cgroup", "0::/run-7.scope\n");
        char leaf[900]; snprintf(leaf, sizeof leaf, "%s/run-7.scope", t.sys);
        mkdirs(leaf);
        put(leaf, "memory.max", "137438953472\n"); put(leaf, "memory.current", "0\n");
        K3MemAvail m;
        int rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
        char d[2048]; snprintf(d, sizeof d, "bytes %.0f cgroup %.0f", m.bytes, m.cgroup);
        ck(rc == 0 && m.known && m.bytes == HOST_BYTES && m.cgroup == 137438953472.0,
           "loose cgroup: min() keeps the host figure", d);
    }

    printf("\n%s\n", g_fail ? "MEMORY PROBE TESTS FAILED" : "MEMORY PROBE TESTS PASSED");
    return g_fail ? 1 : 0;
}
