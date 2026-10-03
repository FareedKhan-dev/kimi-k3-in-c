/* k3_sysmem.h - how much memory this machine can actually hand out, on every platform
 * the engine builds on.
 *
 * Two refusals in k3_run.c compare the plan against this number: the KV cache check
 * and the whole-plan check before allocation. Both are guarded by "available > 0", so
 * a probe that returns 0 does not fail loudly, it silently turns both guards off, and
 * --preset auto refuses outright. Until this header the probe was /proc/meminfo behind
 * #ifdef _WIN32, which is exactly 0 on macOS.
 *
 *   Linux     MemAvailable, the kernel's own estimate of what can be handed out without
 *             swapping, which counts reclaimable page cache; MemFree does not. Capped
 *             by the tightest memory cgroup the process sits under: /proc/meminfo is
 *             machine-wide and knows nothing about a container or a `systemd-run
 *             -p MemoryMax=` scope, so under a 12 GB cap the plan printed
 *             `available 65.56 GB` (tests/fixtures/gates/gates.txt:219, :229) and
 *             admitted a run the kernel then OOM-killed. The figure is
 *             min(MemAvailable, tightest cgroup ancestor's limit - usage). v2 is
 *             authoritative where its memory controller is visible; v1 only when
 *             it is not. A control file that is PRESENT but garbled or unreadable
 *             fails CLOSED (k3_mem_probe reports malformed, and the bytes figure
 *             is 0 with known set): that says the budget is unknowable, not that
 *             it is unlimited. See the K3MemAvail contract below.
 *   Darwin    free + inactive + purgeable pages from host_statistics64. Darwin has no
 *             MemAvailable; this is the same set of pages, the ones the kernel will give
 *             up without paging anything out. free_count already includes speculative.
 *   Windows   ullAvailPhys, the MemAvailable equivalent (it accounts for the standby
 *             list the way MemAvailable accounts for cache), capped by ullAvailPageFile,
 *             the remaining commit limit: a machine with a small pagefile can refuse to
 *             commit an allocation while plenty of physical memory sits free, and the
 *             engine's arenas are committed up front.
 *
 * Both functions return 0 when the platform cannot say, which every caller already
 * treats as "unknown", not as "none". k3_mem_probe() is the same probe with the
 * breakdown attached, for the callers that must tell "unknown" from "fail closed" and
 * for the plan's `available` line. */
#ifndef K3_SYSMEM_H
#define K3_SYSMEM_H

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

/* ---- the breakdown behind k3_mem_available_bytes() --------------------------
 *   bytes      what the caller may plan against, or 0
 *   known      1 when at least one source was read; 0 means UNKNOWN. Unknown is not
 *              zero: callers keep their skip-the-check behaviour on it, exactly as
 *              they did when /proc/meminfo could not be opened.
 *   malformed  1 when a control file is VISIBLE but cannot be trusted: garbled,
 *              negative, unreadable. That is an authoritative signal that the
 *              budget is unknowable, not a hint that it is unlimited, so bytes is
 *              0 with known set: the caller refuses to start and prints `detail`.
 *   host       the platform's own figure (MemAvailable, host_statistics64,
 *              ullAvailPhys capped by the commit limit), or -1 when unreadable
 *   cgroup     the tightest finite cgroup headroom, or -1 when no finite limit
 *              binds (always -1 off Linux: Windows job objects and Darwin have no
 *              cgroup and are not consulted)
 *   cgroup_v   2 or 1: which hierarchy supplied it (0 when none)
 *   limit_path the control file whose limit binds, for the banner */
typedef struct {
    double bytes;
    int    known;
    int    malformed;
    double host;
    double cgroup;
    double cgroup_limit;     /* the binding limit itself, for the banner */
    double cgroup_usage;     /* usage charged against that limit */
    int    cgroup_v;
    char   limit_path[640];  /* wide enough for any probe path built below */
    char   detail[1024];
} K3MemAvail;

/* ---- the cgroup walk -----------------------------------------------------------
 * Plain stdio over explicit paths, so it compiles everywhere and tests/unit/test_mem.c
 * can point it at a fake /proc + /sys tree on any platform; only the live Linux
 * probe below actually calls it against the real one. The membership path from
 * /proc/self/cgroup is mapped into the mounted tree; a leaf without memory.* files
 * (the controller enabled only on an ancestor) walks up to the nearest that has
 * them, and a container runtime that mounts the process's own subgroup as the root
 * while /proc still reports the host path falls back to the root. Every walk is
 * bounded. */
#define K3_MEM_MAX_ANCESTORS 32

/* One control file. Returns 1 with *v set, 0 when the file is absent (ENOENT/ENOTDIR
 * only), and -1 when it is present but unreadable or not a non-negative integer.
 * "max" is the v2 spelling of unlimited and comes back as 1 with *unlimited set. */
static int k3_mem_read_u64(const char *path, double *v, int *unlimited, K3MemAvail *out)
{
    if (unlimited) *unlimited = 0;
    FILE *f = fopen(path, "r");
    if (!f) {
        if (errno == ENOENT || errno == ENOTDIR) return 0;
        snprintf(out->detail, sizeof out->detail, "cannot read %s: %s", path, strerror(errno));
        return -1;
    }
    char line[64];
    if (!fgets(line, sizeof line, f)) {
        fclose(f);
        snprintf(out->detail, sizeof out->detail, "%s is empty", path);
        return -1;
    }
    fclose(f);
    size_t n = strlen(line);
    while (n && (line[n - 1] == '\n' || line[n - 1] == '\r' || line[n - 1] == ' ')) line[--n] = 0;
    if (!strcmp(line, "max")) { if (unlimited) *unlimited = 1; *v = -1.0; return 1; }
    if (n == 0 || line[0] < '0' || line[0] > '9') {
        snprintf(out->detail, sizeof out->detail, "%s holds '%s', not a byte count", path, line);
        return -1;
    }
    char *end = NULL;
    errno = 0;
    unsigned long long u = strtoull(line, &end, 10);
    if (errno || !end || *end) {
        snprintf(out->detail, sizeof out->detail, "%s holds '%s', not a byte count", path, line);
        return -1;
    }
    *v = (double)u;
    return 1;
}

static int k3_mem_file_exists(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0;
}

/* Strip the last path component in place; returns 0 at the top. */
static int k3_mem_parent(char *path, const char *root)
{
    if (!strcmp(path, root)) return 0;
    char *s = strrchr(path, '/');
    if (!s || s == path) return 0;
    *s = 0;
    if (strlen(path) < strlen(root)) return 0;   /* walked above the mount */
    return 1;
}

/* One hierarchy. Returns 1 if a memory controller was seen anywhere on the walk (so
 * the caller treats this hierarchy as authoritative), 0 if not, -1 on a malformed
 * file. *remaining is the tightest finite (limit - usage), or -1. */
static int k3_mem_walk(const char *root, const char *member, const char *limit_name,
                       const char *usage_name, double unlimited_at, double *remaining,
                       K3MemAvail *out)
{
    char dir[512], probe[600];
    *remaining = -1.0;
    if (member && member[0] == '/')
        snprintf(dir, sizeof dir, "%s%s", root, strcmp(member, "/") ? member : "");
    else
        snprintf(dir, sizeof dir, "%s", root);
    /* trim a trailing slash so parent() and the root comparison agree */
    { size_t n = strlen(dir); while (n > 1 && dir[n - 1] == '/') dir[--n] = 0; }

    /* Nearest directory that actually carries the limit file. */
    int found = 0;
    for (int i = 0; i < K3_MEM_MAX_ANCESTORS; i++) {
        snprintf(probe, sizeof probe, "%s/%s", dir, limit_name);
        if (k3_mem_file_exists(probe)) { found = 1; break; }
        if (!k3_mem_parent(dir, root)) break;
    }
    if (!found) {
        /* Container runtimes mount the process's own subgroup as the root while
         * /proc still reports the host-side path. */
        snprintf(probe, sizeof probe, "%s/%s", root, limit_name);
        if (!k3_mem_file_exists(probe)) return 0;
        snprintf(dir, sizeof dir, "%s", root);
    }

    int seen = 0;
    for (int i = 0; i < K3_MEM_MAX_ANCESTORS; i++) {
        double lim = -1.0, use = 0.0;
        int unl = 0;
        snprintf(probe, sizeof probe, "%s/%s", dir, limit_name);
        int r = k3_mem_read_u64(probe, &lim, &unl, out);
        if (r < 0) return -1;
        if (r > 0) {
            seen = 1;
            if (!unl && lim < unlimited_at) {
                char up[600];
                snprintf(up, sizeof up, "%s/%s", dir, usage_name);
                int ru = k3_mem_read_u64(up, &use, NULL, out);
                if (ru < 0) return -1;
                if (ru == 0) use = 0.0;
                double head = lim - use;
                if (head < 0.0) head = 0.0;
                if (*remaining < 0.0 || head < *remaining) {
                    *remaining = head;
                    out->cgroup_limit = lim;
                    out->cgroup_usage = use;
                    snprintf(out->limit_path, sizeof out->limit_path, "%s", probe);
                }
            }
        }
        if (!k3_mem_parent(dir, root)) break;
    }
    return seen;
}

/* /proc/self/cgroup: `0::/path` is the v2 membership; `N:ctl,ctl:/path` with
 * `memory` among the controllers is the v1 one. Both may appear on a hybrid host,
 * and the v1 line is often FIRST, so match on the prefix rather than the line. */
static int k3_mem_memberships(const char *proc_cgroup, char *v2, size_t v2n,
                              char *v1, size_t v1n, K3MemAvail *out)
{
    v2[0] = 0; v1[0] = 0;
    FILE *f = fopen(proc_cgroup, "r");
    if (!f) return 0;
    char line[600];
    int any = 0;
    while (fgets(line, sizeof line, f)) {
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
        char *c1 = strchr(line, ':');
        char *c2 = c1 ? strchr(c1 + 1, ':') : NULL;
        if (!c1 || !c2) {
            fclose(f);
            snprintf(out->detail, sizeof out->detail, "%s: unparseable line '%s'", proc_cgroup, line);
            return -1;
        }
        any = 1;
        *c1 = 0; *c2 = 0;
        const char *ctls = c1 + 1, *path = c2 + 1;
        if (!strcmp(line, "0") && !*ctls) { snprintf(v2, v2n, "%s", path); continue; }
        /* v1: controllers are comma separated; look for the whole word `memory` */
        const char *p = ctls;
        while (*p) {
            const char *e = strchr(p, ',');
            size_t len = e ? (size_t)(e - p) : strlen(p);
            if (len == 6 && !strncmp(p, "memory", 6)) { snprintf(v1, v1n, "%s", path); break; }
            if (!e) break;
            p = e + 1;
        }
    }
    fclose(f);
    return any;
}

/* One "Field:" line of a /proc/meminfo-shaped file, in bytes; -1 if unreadable or
 * absent. */
static double k3_meminfo_field_at(const char *meminfo, const char *field)
{
    FILE *f = fopen(meminfo, "r");
    if (!f) return -1.0;
    const size_t n = strlen(field);
    char line[256];
    double kb = -1.0;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, field, n)) { kb = atof(line + n); break; }
    fclose(f);
    return kb < 0.0 ? -1.0 : kb * 1024.0;
}

/* The Linux probe against explicit paths, so a test can point it at a fake tree.
 * Returns 0 when something was learned (out->known), -1 when nothing was. */
static int k3_mem_available_at(const char *meminfo, const char *proc_cgroup,
                               const char *cgroup_root, K3MemAvail *out)
{
    memset(out, 0, sizeof *out);
    out->host = -1.0; out->cgroup = -1.0; out->cgroup_limit = -1.0;

    out->host = k3_meminfo_field_at(meminfo, "MemAvailable:");

    char v2[512], v1[512];
    double remaining = -1.0;
    int r = k3_mem_memberships(proc_cgroup, v2, sizeof v2, v1, sizeof v1, out);
    if (r < 0) goto closed;
    if (r > 0) {
        if (v2[0]) {
            int w = k3_mem_walk(cgroup_root, v2, "memory.max", "memory.current",
                                1e300, &remaining, out);
            if (w < 0) goto closed;
            if (w > 0) { out->cgroup_v = 2; out->cgroup = remaining; }
        }
        if (!out->cgroup_v && v1[0]) {
            /* PAGE_COUNTER_MAX (2^63 - PAGE_SIZE) is v1's unlimited; anything at
             * or above 2^62 is the kernel saying "none", not a real limit. */
            const double v1_unlimited = 4611686018427387904.0;
            char sub[600];
            snprintf(sub, sizeof sub, "%s/memory", cgroup_root);
            int w = k3_mem_walk(sub, v1, "memory.limit_in_bytes", "memory.usage_in_bytes",
                                v1_unlimited, &remaining, out);
            if (w == 0)
                w = k3_mem_walk(cgroup_root, v1, "memory.limit_in_bytes",
                                "memory.usage_in_bytes", v1_unlimited, &remaining, out);
            if (w < 0) goto closed;
            if (w > 0) { out->cgroup_v = 1; out->cgroup = remaining; }
        }
    }

    if (out->host < 0.0 && out->cgroup < 0.0) { out->known = 0; out->bytes = 0.0; return -1; }
    out->known = 1;
    if (out->host < 0.0)        out->bytes = out->cgroup;
    else if (out->cgroup < 0.0) out->bytes = out->host;
    else                        out->bytes = out->host < out->cgroup ? out->host : out->cgroup;
    return 0;

closed:
    /* A present but untrustworthy control file: fail closed. Known, and zero. */
    out->known = 1; out->malformed = 1; out->bytes = 0.0;
    return 0;
}

#if !defined(_WIN32) && !defined(__APPLE__)
/* One "Field:" line of /proc/meminfo, in bytes; 0 if absent. */
static double k3_meminfo_field(const char *field)
{
    const double b = k3_meminfo_field_at("/proc/meminfo", field);
    return b < 0.0 ? 0.0 : b;
}
#endif

/* The probe against the live system, with the breakdown. Fills *out and returns
 * out->bytes. On Windows and Darwin there is no cgroup, so host is the whole answer;
 * on Linux it is MemAvailable capped by the tightest cgroup limit, and a present but
 * unreadable control file comes back as malformed with bytes 0. */
static double k3_mem_probe(K3MemAvail *out)
{
    K3MemAvail tmp;
    if (!out) out = &tmp;
    memset(out, 0, sizeof *out);
    out->host = -1.0; out->cgroup = -1.0; out->cgroup_limit = -1.0;
#if defined(_WIN32)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof ms;
    if (!GlobalMemoryStatusEx(&ms)) return 0.0;
    out->host = (double)(ms.ullAvailPhys < ms.ullAvailPageFile ? ms.ullAvailPhys
                                                               : ms.ullAvailPageFile);
    out->bytes = out->host;
    out->known = 1;
    return out->bytes;
#elif defined(__APPLE__)
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page = 0;
    const mach_port_t host = mach_host_self();
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &count) != KERN_SUCCESS)
        return 0.0;
    if (host_page_size(host, &page) != KERN_SUCCESS || page == 0) return 0.0;
    out->host = ((double)vm.free_count + (double)vm.inactive_count + (double)vm.purgeable_count)
                * (double)page;
    out->bytes = out->host;
    out->known = 1;
    return out->bytes;
#else
    k3_mem_available_at("/proc/meminfo", "/proc/self/cgroup", "/sys/fs/cgroup", out);
    return out->bytes;
#endif
}

/* What the machine can hand out, as one number: 0 when unknown, and also 0 when a
 * cgroup control file is present but unreadable. Callers that must tell those two
 * apart (the admission refusals in k3_run.c) use k3_mem_probe() directly. */
static inline double k3_mem_available_bytes(void)
{
    K3MemAvail m;
    return k3_mem_probe(&m);
}

static inline double k3_mem_total_bytes(void)
{
#if defined(_WIN32)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof ms;
    if (!GlobalMemoryStatusEx(&ms)) return 0.0;
    return (double)ms.ullTotalPhys;
#elif defined(__APPLE__)
    uint64_t bytes = 0;
    size_t len = sizeof bytes;
    if (sysctlbyname("hw.memsize", &bytes, &len, NULL, 0) != 0) return 0.0;
    return (double)bytes;
#else
    return k3_meminfo_field("MemTotal:");
#endif
}

#endif /* K3_SYSMEM_H */
