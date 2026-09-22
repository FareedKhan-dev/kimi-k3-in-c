/* json.h - a minimal header-only JSON parser.
 *
 * Sized for exactly what this project reads and no more:
 *   - safetensors headers (the tensor index at the front of each shard)
 *   - config.json and tokenizer_config.json
 *   - the reference fixtures under tests/fixtures/ (prompt_ids / full_ids)
 *
 * It is deliberately incomplete: no \uXXXX escapes, no exotic number syntax. Those do
 * not occur in the inputs above, and supporting them would add parsing surface to code
 * that runs against files fetched from the network.
 */
#ifndef JSON_H
#define JSON_H
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype;

typedef struct jval {
    jtype t;
    double num;            /* J_NUM */
    int    boolean;        /* J_BOOL */
    char  *str;            /* J_STR (NUL-terminated, inside the arena) */
    /* array: children in [0..len). object: keys[] and kids[] run in parallel */
    struct jval **kids;
    char        **keys;    /* J_OBJ only    */
    int           len;
} jval;

typedef struct {
    const char *s;
    char       *arena;     /* backing buffer for the parsed strings */
    size_t      acap, aoff;
    int         depth;     /* current nesting depth: the bound that stops a hostile
                            * JSON like [[[[...]]]] from overflowing this recursive-descent
                            * parser's stack */
    int         oom;       /* latched on any allocation failure: parsing unwinds
                            * and json_parse returns NULL instead of crashing
                            * through a NULL pointer a few frames later */
} jparser;

/* Nesting ceiling. Safetensors headers and config files are shallow (depth ~3), so
 * 1024 is far above anything legitimate while staying well below the stack limit.
 * The cap exists so that a hostile or corrupt file cannot recurse the parser into a
 * stack overflow. */
#define J_MAX_DEPTH 1024

static char *j_dup(jparser *p, const char *b, int n) {
    /* Every string gets its own allocation. An arena that reallocs would move the
     * buffer and invalidate every pointer already handed out (use-after-free). */
    char *d = (char *)malloc((size_t)n + 1);
    if (!d) { p->oom = 1; return NULL; }
    memcpy(d, b, n); d[n] = 0;
    return d;
}

static void j_ws(jparser *p) { while (*p->s && isspace((unsigned char)*p->s)) p->s++; }

static jval *j_new(jparser *p, jtype t) {
    jval *v = (jval *)calloc(1, sizeof(jval));
    if (!v) { p->oom = 1; return NULL; }
    v->t = t; return v;
}

static jval *j_parse_val(jparser *p);

static char *j_parse_str_raw(jparser *p) {
    /* assume *p->s == '"' */
    p->s++;
    /* A GROWING heap buffer, deliberately not a fixed stack array. A fixed 64 KB
     * buffer silently truncates the long strings that appear in tokenizer and config
     * files, which corrupts values instead of reporting an error, and costs 64 KB of
     * stack on every call. */
    size_t cap = 64, n = 0; char *tmp = (char *)malloc(cap);
    if (!tmp) { p->oom = 1; return NULL; }
    /* A library must not exit(1) on OOM: the CLI distinguishes misuse (2)
     * from failure, and every json_parse caller already handles NULL. */
    #define J_PUT(ch) do{ if (n + 1 >= cap) { cap *= 2; char *nt = (char *)realloc(tmp, cap); \
        if (!nt) { free(tmp); p->oom = 1; return NULL; } tmp = nt; } tmp[n++] = (char)(ch); }while(0)
    while (*p->s && *p->s != '"') {
        char c = *p->s++;
        if (c == '\\' && *p->s) {
            char e = *p->s++;
            switch (e) {
                case 'n': c = '\n'; break; case 't': c = '\t'; break;
                case 'r': c = '\r'; break; case 'b': c = '\b'; break;
                case 'f': c = '\f'; break; case '/': c = '/'; break;
                case '\\': c = '\\'; break; case '"': c = '"'; break;
                case 'u': {  /* \uXXXX -> codepoint UTF-8 (con coppie surrogate) */
                    if (!p->s[0]||!p->s[1]||!p->s[2]||!p->s[3]) { c='?'; break; }   /* truncated \u: do not read past the NUL */
                    unsigned cp = (unsigned)strtoul((char[]){p->s[0],p->s[1],p->s[2],p->s[3],0}, NULL, 16);
                    p->s += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && p->s[0]=='\\' && p->s[1]=='u'
                        && p->s[2] && p->s[3] && p->s[4] && p->s[5]) {
                        unsigned lo = (unsigned)strtoul((char[]){p->s[2],p->s[3],p->s[4],p->s[5],0}, NULL, 16);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) { cp = 0x10000 + ((cp-0xD800)<<10) + (lo-0xDC00); p->s += 6; }
                    }
                    if (cp < 0x80) { J_PUT(cp); }
                    else if (cp < 0x800) { J_PUT(0xC0|(cp>>6)); J_PUT(0x80|(cp&0x3F)); }
                    else if (cp < 0x10000) { J_PUT(0xE0|(cp>>12)); J_PUT(0x80|((cp>>6)&0x3F)); J_PUT(0x80|(cp&0x3F)); }
                    else { J_PUT(0xF0|(cp>>18)); J_PUT(0x80|((cp>>12)&0x3F)); J_PUT(0x80|((cp>>6)&0x3F)); J_PUT(0x80|(cp&0x3F)); }
                    continue;
                }
                default: c = e; break;
            }
        }
        J_PUT(c);
    }
    #undef J_PUT
    if (*p->s == '"') p->s++;
    char *out = j_dup(p, tmp, (int)n); free(tmp);
    return out;   /* NULL with oom latched on failure */
}

/* Grow helpers: realloc-or-latch, never assigning NULL over the live
 * pointer (the old code wrote realloc's result straight back, so one OOM
 * wrote through NULL two lines later). Objects grow keys[] and kids[] under
 * ONE cap update: doubling per array (8->16->32) while growing each once
 * overruns the first array past its allocation. */
static int j_grow(jparser *p, void **pp, int *cap, size_t elem) {
    int ncap = (*cap) * 2;
    void *np = realloc(*pp, (size_t)ncap * elem);
    if (!np) { p->oom = 1; return 0; }
    *pp = np; *cap = ncap;
    return 1;
}

static int j_grow2(jparser *p, void **a, size_t ae, void **b, size_t be,
                   int *cap) {
    int ncap = (*cap) * 2;
    void *na = realloc(*a, (size_t)ncap * ae);
    if (!na) { p->oom = 1; return 0; }
    *a = na;
    void *nb = realloc(*b, (size_t)ncap * be);
    if (!nb) { p->oom = 1; return 0; }   /* *a already grown; unwind frees it */
    *b = nb;
    *cap = ncap;
    return 1;
}

static jval *j_parse_val(jparser *p);
static void json_free_tree(jval *v);

static jval *j_parse_obj(jparser *p) {
    if (++p->depth > J_MAX_DEPTH) { p->depth--; return j_new(p, J_NULL); }
    p->s++; jval *v = j_new(p, J_OBJ);
    if (!v) { p->depth--; return NULL; }
    int cap = 8;
    v->keys = malloc((size_t)cap * sizeof(char*));
    v->kids = malloc((size_t)cap * sizeof(jval*));
    if (!v->keys || !v->kids) {
        free(v->keys); free(v->kids); free(v); p->oom = 1; p->depth--;
        return NULL;
    }
    j_ws(p);
    if (*p->s == '}') { p->s++; p->depth--; return v; }
    for (;;) {
        j_ws(p);
        char *key = j_parse_str_raw(p);
        j_ws(p); if (*p->s == ':') p->s++;
        jval *val = j_parse_val(p);
        /* Every allocated node is owned by exactly one frame's kids[], so
         * freeing this frame's partial v closes the whole failed subtree;
         * only the in-flight key/val need separate handling. */
        if (!key || !val || p->oom) {
            free(key); json_free_tree(val); json_free_tree(v);
            p->depth--;
            return NULL;
        }
        if (v->len == cap &&
            !j_grow2(p, (void **)&v->keys, sizeof(char*),
                        (void **)&v->kids, sizeof(jval*), &cap)) {
            free(key); json_free_tree(val); json_free_tree(v);
            p->depth--;
            return NULL;
        }
        v->keys[v->len] = key; v->kids[v->len] = val; v->len++;
        j_ws(p);
        if (*p->s == ',') { p->s++; continue; }
        if (*p->s == '}') { p->s++; break; }
        break;
    }
    p->depth--;
    return v;
}

static jval *j_parse_arr(jparser *p) {
    if (++p->depth > J_MAX_DEPTH) { p->depth--; return j_new(p, J_NULL); }
    p->s++; jval *v = j_new(p, J_ARR);
    if (!v) { p->depth--; return NULL; }
    int cap = 8;
    v->kids = malloc((size_t)cap * sizeof(jval*));
    if (!v->kids) { free(v); p->oom = 1; p->depth--; return NULL; }
    j_ws(p);
    if (*p->s == ']') { p->s++; p->depth--; return v; }
    for (;;) {
        jval *val = j_parse_val(p);
        if (!val || p->oom) { json_free_tree(val); json_free_tree(v); p->depth--; return NULL; }
        if (v->len == cap &&
            !j_grow(p, (void **)&v->kids, &cap, sizeof(jval*))) {
            json_free_tree(val); json_free_tree(v);
            p->depth--;
            return NULL;
        }

        v->kids[v->len++] = val;
        j_ws(p);
        if (*p->s == ',') { p->s++; continue; }
        if (*p->s == ']') { p->s++; break; }
        break;
    }
    p->depth--;
    return v;
}

static jval *j_parse_val(jparser *p) {
    j_ws(p);
    char c = *p->s;
    if (c == '"') {
        jval *v = j_new(p, J_STR);
        if (!v) return NULL;
        v->str = j_parse_str_raw(p);
        if (!v->str) { free(v); return NULL; }
        return v;
    }
    if (c == '{') return j_parse_obj(p);
    if (c == '[') return j_parse_arr(p);
    if (c == 't' && !strncmp(p->s, "true", 4))  { p->s += 4; jval *v = j_new(p, J_BOOL); if (!v) return NULL; v->boolean = 1; return v; }
    if (c == 'f' && !strncmp(p->s, "false", 5)) { p->s += 5; jval *v = j_new(p, J_BOOL); if (!v) return NULL; v->boolean = 0; return v; }
    if (c == 'n' && !strncmp(p->s, "null", 4))  { p->s += 4; return j_new(p, J_NULL); }
    /* numero */
    { char *end; double d = strtod(p->s, &end); p->s = end; jval *v = j_new(p, J_NUM); if (!v) return NULL; v->num = d; return v; }
}

/* API */
static jval *json_parse(const char *text, char **arena_out) {
    jparser p = { text, NULL, 0, 0, 0, 0 };
    jval *v = j_parse_val(&p);
    if (p.oom) { json_free_tree(v); v = NULL; }
    if (arena_out) *arena_out = p.arena; else free(p.arena);
    return v;
}

static jval *json_get(jval *o, const char *key) {
    if (!o || o->t != J_OBJ) return NULL;
    for (int i = 0; i < o->len; i++) if (strcmp(o->keys[i], key) == 0) return o->kids[i];
    return NULL;
}

/* Free a parsed tree: nodes, key/value arrays, and every string. Every string
 * the parser hands out is individually heap-owned (see j_dup), including
 * object keys and J_STR values, so freeing after extraction leaks nothing.
 * The one exception a caller must respect: anything it ALIASED out of the
 * tree (k3_trunk.c tensor names, k3_tok.h special-token strings) must be
 * strdup'd first (trunk) or deliberately process-lifetime (tokenizer). */
static void json_free_tree(jval *v) {
    int i;
    if (!v) return;
    if (v->t == J_OBJ) {
        for (i = 0; i < v->len; i++) {
            free(v->keys[i]);
            json_free_tree(v->kids[i]);
        }
        free(v->keys);
        free(v->kids);
    } else if (v->t == J_ARR) {
        for (i = 0; i < v->len; i++) json_free_tree(v->kids[i]);
        free(v->kids);
    } else if (v->t == J_STR) {
        free(v->str);
    }
    free(v);
}

#endif
