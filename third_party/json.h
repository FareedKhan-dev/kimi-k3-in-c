/* json.h - a minimal header-only JSON parser.
 *
 * Sized for exactly what this project reads and no more:
 *   - safetensors headers (the tensor index at the front of each shard)
 *   - config.json and tokenizer_config.json
 *   - trunk.json manifests
 *   - the reference fixtures under tests/fixtures/ (prompt_ids / full_ids)
 *
 * It is strict on purpose: a truncated or malformed document is refused with
 * NULL rather than parsed as if it were complete. Every input here can be
 * corrupt or hostile, and a wrong answer that looks right is the worst
 * failure mode of this engine. Callers already handle NULL.
 */
#ifndef JSON_H
#define JSON_H
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>

typedef enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } jtype;

typedef struct jval {
    jtype t;
    double num;            /* J_NUM */
    int    boolean;        /* J_BOOL */
    char  *str;            /* J_STR (NUL-terminated, individually owned) */
    /* array: children in [0..len). object: keys[] and kids[] run in parallel */
    struct jval **kids;
    char        **keys;    /* J_OBJ only    */
    int           len;
} jval;

typedef struct {
    const char *s;
    char       *arena;     /* unused, kept for the json_parse(text, &arena) shape */
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
 * stack overflow. Past the ceiling the document is refused outright. */
#define J_MAX_DEPTH 1024

static char *j_dup(jparser *p, const char *b, int n) {
    /* Every string gets its own allocation. An arena that reallocs would move the
     * buffer and invalidate every pointer already handed out (use-after-free). */
    char *d = (char *)malloc((size_t)n + 1);
    if (!d) { p->oom = 1; return NULL; }
    memcpy(d, b, n); d[n] = 0;
    return d;
}

/* JSON whitespace only: space, tab, LF, CR. isspace() also accepts \v and \f,
 * which JSON does not, so a document starting with \v must not parse. */
static void j_ws(jparser *p) {
    while (*p->s == ' ' || *p->s == '\t' || *p->s == '\n' || *p->s == '\r') p->s++;
}

static jval *j_new(jparser *p, jtype t) {
    jval *v = (jval *)calloc(1, sizeof(jval));
    if (!v) { p->oom = 1; return NULL; }
    v->t = t; return v;
}

static jval *j_parse_val(jparser *p);

static int j_hex(const char *s, unsigned *out) {
    unsigned v = 0;
    int i;
    for (i = 0; i < 4; i++) {
        char c = s[i];
        unsigned d;
        if (c >= '0' && c <= '9') d = (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') d = (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') d = (unsigned)(c - 'A' + 10);
        else return 0;
        v = (v << 4) | d;
    }
    *out = v;
    return 1;
}

static char *j_parse_str_raw(jparser *p) {
    /* The caller guarantees the opening quote; a NUL here means truncation. */
    if (*p->s != '"') return NULL;
    p->s++;
    /* A GROWING heap buffer, deliberately not a fixed stack array. A fixed 64 KB
     * buffer silently truncates the long strings that appear in tokenizer and config
     * files, which corrupts values instead of reporting an error, and costs 64 KB of
     * stack on every call. */
    size_t cap = 64, n = 0; char *tmp = (char *)malloc(cap);
    if (!tmp) { p->oom = 1; return NULL; }
    /* A library must not exit(1) on OOM: the CLI distinguishes misuse (2)
     * from failure, and every json_parse caller already handles NULL. */
    #define J_PUT(ch) do{ if (n + 1 >= cap) { \
        if (cap > (size_t)0x40000000) { free(tmp); p->oom = 1; return NULL; } \
        cap *= 2; char *nt = (char *)realloc(tmp, cap); \
        if (!nt) { free(tmp); p->oom = 1; return NULL; } tmp = nt; } \
        tmp[n++] = (char)(ch); }while(0)
    for (;;) {
        char c = *p->s;
        if (c == '\0') { free(tmp); return NULL; }   /* unterminated */
        if (c == '"') { p->s++; break; }             /* closing quote */
        if ((unsigned char)c < 0x20) { free(tmp); return NULL; }  /* raw control */
        if (c != '\\') { p->s++; J_PUT(c); continue; }
        /* Escape. *p->s == '\\' and the next byte decides; NUL ends the input. */
        p->s++;
        c = *p->s;
        if (c == '\0') { free(tmp); return NULL; }
        switch (c) {
            case '"':  p->s++; J_PUT('"'); break;
            case '\\': p->s++; J_PUT('\\'); break;
            case '/':  p->s++; J_PUT('/'); break;
            case 'b':  p->s++; J_PUT('\b'); break;
            case 'f':  p->s++; J_PUT('\f'); break;
            case 'n':  p->s++; J_PUT('\n'); break;
            case 'r':  p->s++; J_PUT('\r'); break;
            case 't':  p->s++; J_PUT('\t'); break;
            case 'u': {
                unsigned cp, lo;
                /* Four hex digits must follow; the NUL check inside j_hex's
                 * caller matters because p->s[1..4] may run into the end. */
                if (p->s[1] == '\0' || p->s[2] == '\0' ||
                    p->s[3] == '\0' || p->s[4] == '\0') { free(tmp); return NULL; }
                if (!j_hex(p->s + 1, &cp)) { free(tmp); return NULL; }
                p->s += 5;
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    /* High surrogate: a low surrogate must follow, else the
                     * document is malformed rather than encodable. */
                    if (p->s[0] != '\\' || p->s[1] != 'u' ||
                        p->s[2] == '\0' || p->s[3] == '\0' ||
                        p->s[4] == '\0' || p->s[5] == '\0' ||
                        !j_hex(p->s + 2, &lo) || lo < 0xDC00 || lo > 0xDFFF) {
                        free(tmp); return NULL;
                    }
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    p->s += 6;
                } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                    free(tmp); return NULL;   /* lone low surrogate */
                } else if (cp == 0) {
                    free(tmp); return NULL;   /* NUL cannot live in a C string */
                }
                if (cp < 0x80) { J_PUT(cp); }
                else if (cp < 0x800) { J_PUT(0xC0|(cp>>6)); J_PUT(0x80|(cp&0x3F)); }
                else if (cp < 0x10000) { J_PUT(0xE0|(cp>>12)); J_PUT(0x80|((cp>>6)&0x3F)); J_PUT(0x80|(cp&0x3F)); }
                else { J_PUT(0xF0|(cp>>18)); J_PUT(0x80|((cp>>12)&0x3F)); J_PUT(0x80|((cp>>6)&0x3F)); J_PUT(0x80|(cp&0x3F)); }
                break;
            }
            default: free(tmp); return NULL;   /* no \q, \a, or friends */
        }
    }
    #undef J_PUT
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

static jval *j_parse_num(jparser *p) {
    /* Strict JSON numbers: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?.
     * strtod alone accepts "+1", "01", "NaN", "Infinity" and hex, none of
     * which this parser's inputs may contain. NaN/Infinity would also turn
     * into UB one cast later, where a double becomes an int64 offset. */
    const char *s = p->s;
    if (*s == '-') s++;
    if (*s == '0') {
        s++;
        if (*s >= '0' && *s <= '9') return NULL;   /* no leading zeros */
    } else if (*s >= '1' && *s <= '9') {
        do s++; while (*s >= '0' && *s <= '9');
    } else {
        return NULL;
    }
    if (*s == '.') {
        s++;
        if (*s < '0' || *s > '9') return NULL;
        do s++; while (*s >= '0' && *s <= '9');
    }
    if (*s == 'e' || *s == 'E') {
        s++;
        if (*s == '+' || *s == '-') s++;
        if (*s < '0' || *s > '9') return NULL;
        do s++; while (*s >= '0' && *s <= '9');
    }
    {
        char *end = NULL;
        double d = strtod(p->s, &end);
        if (end != s || !isfinite(d)) return NULL;
        p->s = s;
        jval *v = j_new(p, J_NUM);
        if (!v) return NULL;
        v->num = d;
        return v;
    }
}

static jval *j_parse_val(jparser *p);
static void json_free_tree(jval *v);

static jval *j_parse_obj(jparser *p) {
    if (*p->s != '{') return NULL;
    if (++p->depth > J_MAX_DEPTH) { p->depth--; return NULL; }
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
    if (*p->s == '\0') { json_free_tree(v); p->depth--; return NULL; }
    for (;;) {
        j_ws(p);
        /* Keys are always quoted strings. Anything else, including NUL for
         * a truncated "{", is a refusal rather than a scan past the end. */
        if (*p->s != '"') { json_free_tree(v); p->depth--; return NULL; }
        char *key = j_parse_str_raw(p);
        j_ws(p);
        /* The colon is mandatory: {"a" 1} must not parse as {"a":1}. */
        if (!key || p->oom || *p->s != ':') {
            free(key); json_free_tree(v); p->depth--;
            return NULL;
        }
        p->s++;
        jval *val = j_parse_val(p);
        /* Every allocated node is owned by exactly one frame's kids[], so
         * freeing this frame's partial v closes the whole failed subtree;
         * only the in-flight key/val need separate handling. */
        if (!val || p->oom) {
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
        /* Missing comma or brace, or truncation: {"a":1 and {"a":1 "b":2}. Key and value are already owned by v. */
        json_free_tree(v); p->depth--;
        return NULL;
    }
    p->depth--;
    return v;
}

static jval *j_parse_arr(jparser *p) {
    if (*p->s != '[') return NULL;
    if (++p->depth > J_MAX_DEPTH) { p->depth--; return NULL; }
    p->s++; jval *v = j_new(p, J_ARR);
    if (!v) { p->depth--; return NULL; }
    int cap = 8;
    v->kids = malloc((size_t)cap * sizeof(jval*));
    if (!v->kids) { free(v); p->oom = 1; p->depth--; return NULL; }
    j_ws(p);
    if (*p->s == ']') { p->s++; p->depth--; return v; }
    if (*p->s == '\0') { json_free_tree(v); p->depth--; return NULL; }
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
        /* [1 2], [1, and [1 all end here. A trailing comma fails on the
         * next value parse ([1,] -> ']' is not a value). */
        json_free_tree(v); p->depth--;
        return NULL;
    }
    p->depth--;
    return v;
}

static jval *j_parse_val(jparser *p) {
    j_ws(p);
    {
        char c = *p->s;
        jval *v;
        if (c == '\0') return NULL;
        if (c == '"') {
            v = j_new(p, J_STR);
            if (!v) return NULL;
            v->str = j_parse_str_raw(p);
            if (!v->str) { free(v); return NULL; }
            return v;
        }
        if (c == '{') return j_parse_obj(p);
        if (c == '[') return j_parse_arr(p);
        /* Literals must end at a delimiter: "truth" is not "true". */
        if (c == 't' && !strncmp(p->s, "true", 4) &&
            !isalnum((unsigned char)p->s[4]) && p->s[4] != '_') {
            p->s += 4; v = j_new(p, J_BOOL); if (!v) return NULL; v->boolean = 1; return v;
        }
        if (c == 'f' && !strncmp(p->s, "false", 5) &&
            !isalnum((unsigned char)p->s[5]) && p->s[5] != '_') {
            p->s += 5; v = j_new(p, J_BOOL); if (!v) return NULL; v->boolean = 0; return v;
        }
        if (c == 'n' && !strncmp(p->s, "null", 4) &&
            !isalnum((unsigned char)p->s[4]) && p->s[4] != '_') {
            p->s += 4; return j_new(p, J_NULL);
        }
        if (c == '-' || (c >= '0' && c <= '9')) return j_parse_num(p);
        return NULL;
    }
}

/* API */
static jval *json_parse(const char *text, char **arena_out) {
    jparser p = { text, NULL, 0, 0, 0, 0 };
    jval *v = j_parse_val(&p);
    if (!v || p.oom) { json_free_tree(v); v = NULL; }
    else {
        /* One value and nothing else: trailing bytes mean truncation or
         * concatenation, never a document. {"a":1} garbage is refused. */
        j_ws(&p);
        if (*p.s != '\0') { json_free_tree(v); v = NULL; }
    }
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
 * tree (k3_tok.h special-token strings) must be strdup'd first (trunk) or
 * deliberately process-lifetime (tokenizer). */
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
