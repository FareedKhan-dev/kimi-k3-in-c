/* test_mem.c - the memory probe must see a cgroup cap, not just MemAvailable.
 *
 * Companion to test_sysmem.c: that one checks the live probe returns a real number
 * on every platform; this one checks the Linux cgroup walk in k3_sysmem.h against
 * trees it builds itself, then exercises the CLI admission callers with that
 * fixture probe, so it runs the same on every platform too.
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

/* Exercise the CLI's actual admission callers with the fixture probe below.
 * Allocation failure stops an admitted chat turn before it needs model weights. */
static K3MemAvail admission_memory;
static int fail_chat_alloc;
static char admission_model[800];
static double admission_probe(K3MemAvail *out)
{
    *out = admission_memory;
    return out->bytes;
}
static void *admission_malloc(size_t n)
{
    return fail_chat_alloc ? NULL : malloc(n);
}
#define k3_mem_probe admission_probe
#define malloc admission_malloc
#define main k3_cli_main
#include "../../src/cli/k3_run.c"
#undef main
#undef malloc
#undef k3_mem_probe

#ifdef _WIN32
#define dup _dup
#define dup2 _dup2
#define close _close
#else
#include <unistd.h>
#endif

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

/* Check the reason as well as the status: a missing checkpoint also exits nonzero.
 * Run the real chat caller with all allocations refused, so an unknown reading
 * reaches the buffer error while a known zero must stop earlier, naming KV. */
static void admission_case(int mode, int want_rc,
                           const char *reason, const char *what)
{
    FILE *log = tmpfile();
    int saved = dup(fileno(stderr));
    if (!log || saved < 0 || dup2(fileno(log), fileno(stderr)) < 0) {
        fprintf(stderr, "cannot capture admission diagnostic\n");
        exit(1);
    }
    int rc;
    if (mode == 1) {
        K3ChatHistory history; k3_chat_history_init(&history);
        K3Cfg c; int fa[128]; real_cfg_hardcoded(&c, fa);
        Weights w; memset(&w, 0, sizeof w);
        int *prompt = NULL, *seq = NULL, tmax = 0;
        float *h = NULL, *br = NULL, *sc = NULL;
        fail_chat_alloc = 1;
        rc = chat_run(NULL, NULL, &history, NULL, NULL, &prompt, 1, 1,
                      1, 1, 1.0, 0.95, 0, 1, &w, &c, NULL,
                      c.n_layers, &tmax, &h, &br, NULL, &sc, NULL, &seq,
                      NULL, 1, 0, NULL);
        fail_chat_alloc = 0;
    } else {
        char *args[] = {"k3", "tests/fixtures/st", "--config",
                        "tests/fixtures/cfg/flat.json", "--ids", "1",
                        "--incremental", "--layers", "1", NULL};
        char *auto_args[] = {"k3", "fake", "--ids", "1", "--preset", "auto", NULL};
        char *plan_args[] = {"k3", admission_model, "--config",
                             "tests/fixtures/cfg/flat.json", "--ids", "1",
                             "--layers", "1", "--cache-gb", "0", NULL};
        if (mode == 2)
            rc = k3_cli_main((int)(sizeof auto_args / sizeof *auto_args) - 1, auto_args);
        else if (mode == 3)
            rc = k3_cli_main((int)(sizeof plan_args / sizeof *plan_args) - 1, plan_args);
        else
            rc = k3_cli_main((int)(sizeof args / sizeof *args) - 1, args);
    }
    fflush(stderr);
    dup2(saved, fileno(stderr)); close(saved);
    rewind(log);
    char detail[8192];
    size_t n = fread(detail, 1, sizeof detail - 1, log);
    detail[n] = 0; fclose(log);
    int ok = rc == want_rc && strstr(detail, reason) != NULL;
    ck(ok, what, ok ? NULL : detail);
}

/* One KDA/dense layer matching cfg/flat.json, with zero F32 payloads. It lets
 * non-incremental runs reach the whole-plan guard without mocking the binder.
 * Model-level tensors are deliberately absent: an admitted plan stops there. */
static int plan_fixture(const char *work)
{
    static const struct { const char *name; int elements; } tensors[] = {
        {"input_layernorm.weight", 128}, {"post_attention_layernorm.weight", 128},
        {"self_attention_res_norm.weight", 128}, {"self_attention_res_proj.weight", 128},
        {"mlp_res_norm.weight", 128}, {"mlp_res_proj.weight", 128},
        {"self_attn.q_proj.weight", 8192}, {"self_attn.k_proj.weight", 8192},
        {"self_attn.v_proj.weight", 8192}, {"self_attn.g_proj.weight", 8192},
        {"self_attn.o_proj.weight", 8192},
        {"self_attn.q_conv1d.weight", 256}, {"self_attn.k_conv1d.weight", 256},
        {"self_attn.v_conv1d.weight", 256},
        {"self_attn.f_a_proj.weight", 2048}, {"self_attn.f_b_proj.weight", 1024},
        {"self_attn.b_proj.weight", 512}, {"self_attn.A_log", 16},
        {"self_attn.dt_bias", 64}, {"self_attn.o_norm.weight", 16},
        {"mlp.gate_proj.weight", 12288}, {"mlp.up_proj.weight", 12288},
        {"mlp.down_proj.weight", 12288}
    };
    snprintf(admission_model, sizeof admission_model, "%s/plan-model", work);
    mkdirs(admission_model);
    char path[900]; snprintf(path, sizeof path, "%s/model.safetensors", admission_model);
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    char header[8192]; size_t n = 0; int bytes = 0;
    for (size_t i = 0; i < sizeof tensors / sizeof *tensors; i++) {
        int end = bytes + tensors[i].elements * 4;
        n += (size_t)snprintf(header + n, sizeof header - n,
            "%s\"language_model.model.layers.0.%s\":{\"dtype\":\"F32\","
            "\"shape\":[%d],\"data_offsets\":[%d,%d]}",
            i ? "," : "{", tensors[i].name, tensors[i].elements, bytes, end);
        bytes = end;
    }
    header[n++] = '}';
    while (n % 8) header[n++] = ' ';
    for (int i = 0; i < 8; i++) fputc((int)(((uint64_t)n >> (8 * i)) & 255), f);
    fwrite(header, 1, n, f);
    for (int i = 0; i < bytes; i++) fputc(0, f);
    int bad = ferror(f);
    if (fclose(f)) bad = 1;
    return bad ? -1 : 0;
}

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

    /* (k) limit == usage is a measured zero, not permission to skip admission. */
    {
        Tree t; tree(&t, work, "k", 1);
        put(t.root, "cgroup", "0::/\n");
        put(t.sys, "memory.max", "1073741824\n");
        put(t.sys, "memory.current", "1073741824\n");
        int rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &admission_memory);
        ck(rc == 0 && admission_memory.known && !admission_memory.malformed
           && admission_memory.bytes == 0.0, "full cgroup: known zero headroom", NULL);
        ck(plan_fixture(work) == 0, "whole-plan model fixture written", NULL);
        admission_case(0, 2, "REFUSING: the KV cache", "known zero: startup refuses KV by name");
        admission_case(1, 1, "chat: KV cache", "known zero: chat refuses KV by name");
        admission_case(2, 2, "--preset auto could not read", "known zero: auto still refuses");
        admission_case(3, 1, "REFUSING TO START: this needs", "known zero: whole plan refuses");
    }
    /* (l) with neither host nor cgroup measurements, explicit budgets still pass
     * the KV guard and reach the unrelated missing-model / buffer diagnostics. */
    {
        Tree t; tree(&t, work, "l", 0);
        int rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &admission_memory);
        ck(rc != 0 && !admission_memory.known && !admission_memory.malformed,
           "no sources: admission remains unknown", NULL);
        admission_case(0, 1, "layers are missing tensors", "unknown: startup skips KV check");
        admission_case(1, 1, "chat: buffer allocation failed", "unknown: chat skips KV check");
        admission_case(2, 2, "--preset auto could not read", "unknown: auto still refuses");
        admission_case(3, 1, "missing tensor language_model.model.embed_tokens.weight",
                       "unknown: whole plan skips check");
    }

    /* (m) File cache can be reclaimed before the cgroup kills an allocation.
     * v1's total_* counters include descendants, just like usage_in_bytes. */
    for (int v = 2; v >= 1; v--) {
        static const struct {
            const char *tag, *inactive, *active, *extra;
            double room;
        } cases[] = {
            {"both file lists", "8589934592", "6442450944", "", 18},
            {"inactive only", "8589934592", NULL, "", 12},
            {"active only", NULL, "6442450944", "", 10},
            {"empty memory.stat", NULL, NULL, "", 4},
            {"absent memory.stat", NULL, NULL, NULL, 4},
            {"anon/file/shmem are not room", "0", "0",
             "anon 21474836480\nfile 21474836480\nshmem 21474836480\n", 4},
            {"cache above usage", "17179869184", "8589934592", "", 24},
            {"cache sum cannot wrap", "18446744073709551615", "18446744073709551615", "", 24},
            {"negative inactive_file", "-8589934592", "6442450944", "", 4},
            {"signed active_file", "8589934592", "+6442450944", "", 4},
            {"malformed active_file", "8589934592", "garbage", "", 4},
            {"active_file trailing junk", "8589934592", "6442450944x", "", 4},
            {"active_file overflow", "8589934592", "18446744073709551616", "", 4},
            {"fractional active_file", "8589934592", "1.5", "", 4},
            {"missing active_file value", "8589934592", "", "", 4},
            {"extra active_file value", "8589934592", "6442450944 1", "", 4},
            {"duplicate inactive_file", "8589934592", "6442450944", "%sinactive_file 1\n", 4},
            {"duplicate active_file", "8589934592", "6442450944", "%sactive_file 1\n", 4},
            {"malformed later line", "8589934592", "6442450944", "broken\n", 4},
            {"malformed unrelated count", "8589934592", "6442450944", "anon -1\n", 4},
            {"overlong later line", "8589934592", "6442450944",
             "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
             "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
             "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
             "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz 1\n", 4},
        };
        const char *prefix = v == 1 ? "total_" : "";
        const char *limit = v == 1 ? "memory.limit_in_bytes" : "memory.max";
        const char *usage = v == 1 ? "memory.usage_in_bytes" : "memory.current";
        const double gib = 1073741824.0;
        for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
            char tag[80]; snprintf(tag, sizeof tag, "k-v%d-%zu", v, i);
            Tree t; tree(&t, work, tag, 1);
            put(t.root, "cgroup", v == 1 ? "4:memory:/\n" : "0::/\n");
            put(t.sys, limit, "25769803776\n");  /* 24 GiB */
            put(t.sys, usage, "21474836480\n");  /* 20 GiB */
            char a[128] = "", b[128] = "", extra[512] = "", stat[1024];
            if (cases[i].inactive)
                snprintf(a, sizeof a, "%sinactive_file %s\n", prefix, cases[i].inactive);
            if (cases[i].active)
                snprintf(b, sizeof b, "%sactive_file %s\n", prefix, cases[i].active);
            if (cases[i].extra) {
                snprintf(extra, sizeof extra, cases[i].extra, prefix);
                snprintf(stat, sizeof stat, "%s%s%s", a, b, extra);
                put(t.sys, "memory.stat", stat);
            }
            K3MemAvail m;
            int rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
            char d[2048], what[128];
            snprintf(what, sizeof what, "v%d %s", v, cases[i].tag);
            snprintf(d, sizeof d, "room %.0f GiB, expected %.0f", m.bytes / gib, cases[i].room);
            ck(rc == 0 && m.known && !m.malformed && m.cgroup_v == v
               && m.bytes == cases[i].room * gib && m.cgroup == m.bytes
               && m.cgroup_limit == 24 * gib && m.cgroup_usage == (24 - cases[i].room) * gib,
               what, d);
        }

        /* Each ancestor uses its own cache; the binding path may change. */
        Tree t; tree(&t, work, v == 1 ? "k-v1-nested" : "k-v2-nested", 1);
        put(t.root, "cgroup", v == 1 ? "4:memory:/parent/leaf\n" : "0::/parent/leaf\n");
        put(t.sys, limit, v == 1 ? "9223372036854771712\n" : "max\n");
        char leaf[900], mid[900], stat[256];
        snprintf(leaf, sizeof leaf, "%s/parent/leaf", t.sys); mkdirs(leaf);
        snprintf(mid, sizeof mid, "%s/parent", t.sys);
        put(leaf, limit, "25769803776\n"); put(leaf, usage, "21474836480\n");
        snprintf(stat, sizeof stat, "%sinactive_file 8589934592\n%sactive_file 6442450944\n", prefix, prefix);
        put(leaf, "memory.stat", stat);
        put(mid, limit, "17179869184\n"); put(mid, usage, "15032385536\n");
        snprintf(stat, sizeof stat, "%sinactive_file 10737418240\n%sactive_file 2147483648\n", prefix, prefix);
        put(mid, "memory.stat", stat);
        K3MemAvail m;
        int rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
        char what[128]; snprintf(what, sizeof what, "v%d parent cache: 14 GiB binds", v);
        ck(rc == 0 && !m.malformed && m.bytes == 14 * gib
           && strstr(m.limit_path, "/parent/memory.") != NULL, what, m.limit_path);

        put(t.sys, limit, "12884901888\n"); put(t.sys, usage, "11811160064\n");
        snprintf(stat, sizeof stat, "%sinactive_file 6442450944\n%sactive_file 2147483648\n", prefix, prefix);
        put(t.sys, "memory.stat", stat);
        rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
        snprintf(what, sizeof what, "v%d mount-root cache: 9 GiB binds", v);
        char path[900]; snprintf(path, sizeof path, "%s/%s", t.sys, limit);
        ck(rc == 0 && !m.malformed && m.bytes == 9 * gib
           && !strcmp(m.limit_path, path), what, m.limit_path);

        put(t.root, "meminfo", "MemAvailable: 8388608 kB\n");
        rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
        snprintf(what, sizeof what, "v%d host still caps reclaimed room", v);
        ck(rc == 0 && !m.malformed && m.bytes == 8 * gib && m.cgroup == 9 * gib, what, NULL);

        /* Only counters from this hierarchy count, even when both spellings exist. */
        snprintf(stat, sizeof stat,
                 "inactive_file %s\nactive_file %s\ntotal_inactive_file %s\ntotal_active_file %s\n",
                 v == 1 ? "11811160064" : "6442450944", v == 1 ? "11811160064" : "2147483648",
                 v == 1 ? "6442450944" : "11811160064", v == 1 ? "2147483648" : "11811160064");
        put(t.sys, "memory.stat", stat);
        rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
        snprintf(what, sizeof what, "v%d ignores other hierarchy's counters", v);
        ck(rc == 0 && !m.malformed && m.cgroup == 9 * gib, what, NULL);

        /* Reclaim before clamping: raw usage can be higher than the limit. */
        put(t.sys, usage, "13958643712\n");   /* 13 GiB - 8 GiB cache = 5 GiB */
        rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
        snprintf(what, sizeof what, "v%d raw usage above limit can reclaim", v);
        ck(rc == 0 && !m.malformed && m.cgroup == 7 * gib && m.bytes == 7 * gib, what, NULL);
        put(t.sys, usage, "22548578304\n");   /* 21 GiB - 8 GiB cache > 12 GiB */
        rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
        snprintf(what, sizeof what, "v%d working set above limit stays zero", v);
        ck(rc == 0 && m.known && !m.malformed && m.cgroup == 0 && m.bytes == 0, what, NULL);

        put(t.sys, limit, "garbage\n");
        rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
        snprintf(what, sizeof what, "v%d bad limit with cache fails closed", v);
        ck(rc == 0 && m.known && m.malformed && m.bytes == 0 && strstr(m.detail, limit), what, m.detail);

        put(t.sys, limit, "12884901888\n"); put(t.sys, usage, "-1\n");
        rc = k3_mem_available_at(t.meminfo, t.cgroup, t.sys, &m);
        snprintf(what, sizeof what, "v%d bad usage with cache fails closed", v);
        ck(rc == 0 && m.known && m.malformed && m.bytes == 0 && strstr(m.detail, usage), what, m.detail);
    }

    printf("\n%s\n", g_fail ? "MEMORY PROBE TESTS FAILED" : "MEMORY PROBE TESTS PASSED");
    return g_fail ? 1 : 0;
}
