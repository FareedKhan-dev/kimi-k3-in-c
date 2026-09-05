/* k3_chat.c - transcript, XTML rendering, and assistant boundary parsing. */
#define _POSIX_C_SOURCE 200809L
#include "k3_chat.h"

#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void fail(char *err, size_t n, const char *fmt, ...)
{
    if (!err || n == 0) return;
    va_list ap; va_start(ap, fmt); vsnprintf(err, n, fmt, ap); va_end(ap);
}

static char *dup_(const char *s)
{
    size_t n = s ? strlen(s) : 0;
    char *d = (char *)malloc(n + 1);
    if (!d) return NULL;
    if (n) memcpy(d, s, n);
    d[n] = 0;
    return d;
}

static const char *role_name(int role)
{
    if (role == K3_CHAT_SYSTEM) return "system";
    if (role == K3_CHAT_USER) return "user";
    if (role == K3_CHAT_ASSISTANT) return "assistant";
    return NULL;
}

static int role_id(const char *s)
{
    if (!strcmp(s, "system")) return K3_CHAT_SYSTEM;
    if (!strcmp(s, "user")) return K3_CHAT_USER;
    if (!strcmp(s, "assistant")) return K3_CHAT_ASSISTANT;
    return -1;
}

void k3_chat_message_free(K3ChatMessage *m)
{
    if (!m) return;
    free(m->content); free(m->reasoning_content);
    memset(m, 0, sizeof *m);
}

void k3_chat_history_init(K3ChatHistory *h) { memset(h, 0, sizeof *h); }

void k3_chat_history_free(K3ChatHistory *h)
{
    if (!h) return;
    for (int i = 0; i < h->n; i++) k3_chat_message_free(&h->v[i]);
    free(h->v); memset(h, 0, sizeof *h);
}

int k3_chat_history_reset(K3ChatHistory *h, char *err, size_t err_n)
{
    char *system = NULL;
    if (h->n && h->v[0].role == K3_CHAT_SYSTEM) {
        system = dup_(h->v[0].content);
        if (!system) { fail(err, err_n, "out of memory preserving system message"); return -1; }
    }
    k3_chat_history_free(h); k3_chat_history_init(h);
    if (system) {
        int rc = k3_chat_history_add(h, K3_CHAT_SYSTEM, system, NULL, err, err_n);
        free(system);
        return rc;
    }
    return 0;
}

int k3_chat_history_validate(const K3ChatHistory *h, char *err, size_t err_n)
{
    int want = K3_CHAT_USER;
    for (int i = 0; i < h->n; i++) {
        const K3ChatMessage *m = &h->v[i];
        if (!role_name(m->role) || !m->content) {
            fail(err, err_n, "history message %d has an invalid role or missing content", i + 1);
            return -1;
        }
        if (m->role == K3_CHAT_SYSTEM) {
            if (i != 0) { fail(err, err_n, "system message must be first (line %d)", i + 1); return -1; }
            continue;
        }
        if (m->role != want) {
            fail(err, err_n, "message %d has role '%s', expected '%s'", i + 1,
                 role_name(m->role), role_name(want));
            return -1;
        }
        if (m->role != K3_CHAT_ASSISTANT && m->reasoning_content) {
            fail(err, err_n, "message %d has reasoning_content but is not assistant", i + 1);
            return -1;
        }
        want = want == K3_CHAT_USER ? K3_CHAT_ASSISTANT : K3_CHAT_USER;
    }
    return 0;
}

int k3_chat_history_add(K3ChatHistory *h, int role, const char *content,
                        const char *reasoning, char *err, size_t err_n)
{
    if (!role_name(role) || !content || (role != K3_CHAT_ASSISTANT && reasoning)) {
        fail(err, err_n, "invalid chat message"); return -1;
    }
    if (h->n == h->cap) {
        int cap = h->cap ? h->cap * 2 : 8;
        K3ChatMessage *v = (K3ChatMessage *)realloc(h->v, (size_t)cap * sizeof(*v));
        if (!v) { fail(err, err_n, "out of memory growing history"); return -1; }
        h->v = v; h->cap = cap;
    }
    K3ChatMessage m; memset(&m, 0, sizeof m); m.role = role;
    m.content = dup_(content);
    m.reasoning_content = reasoning ? dup_(reasoning) : NULL;
    if (!m.content || (reasoning && !m.reasoning_content)) {
        k3_chat_message_free(&m); fail(err, err_n, "out of memory copying message"); return -1;
    }
    h->v[h->n++] = m;
    if (k3_chat_history_validate(h, err, err_n) != 0) {
        k3_chat_message_free(&h->v[--h->n]); return -1;
    }
    return 0;
}

/* ---- strict JSON lines ----------------------------------------------------- */
typedef struct { const char *p; } JScan;

static void jws(JScan *s) { while (*s->p == ' ' || *s->p == '\t' || *s->p == '\r') s->p++; }

static int putc_(char **out, size_t *n, size_t *cap, unsigned char c)
{
    if (*n + 1 >= *cap) {
        size_t nc = *cap ? *cap * 2 : 64;
        char *v = (char *)realloc(*out, nc);
        if (!v) return -1;
        *out = v; *cap = nc;
    }
    (*out)[(*n)++] = (char)c; return 0;
}

static int hex_(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int jstr(JScan *s, char **out, char *err, size_t err_n)
{
    if (*s->p++ != '"') { fail(err, err_n, "expected JSON string"); return -1; }
    char *v = NULL; size_t n = 0, cap = 0;
    while (*s->p && *s->p != '"') {
        unsigned char c = (unsigned char)*s->p++;
        if (c < 0x20) { free(v); fail(err, err_n, "control byte in JSON string"); return -1; }
        if (c != '\\') { if (putc_(&v, &n, &cap, c)) goto oom; continue; }
        char e = *s->p++;
        if (!e) { free(v); fail(err, err_n, "truncated JSON escape"); return -1; }
        switch (e) {
        case '"': case '\\': case '/': if (putc_(&v, &n, &cap, (unsigned char)e)) goto oom; break;
        case 'b': if (putc_(&v, &n, &cap, '\b')) goto oom; break;
        case 'f': if (putc_(&v, &n, &cap, '\f')) goto oom; break;
        case 'n': if (putc_(&v, &n, &cap, '\n')) goto oom; break;
        case 'r': if (putc_(&v, &n, &cap, '\r')) goto oom; break;
        case 't': if (putc_(&v, &n, &cap, '\t')) goto oom; break;
        case 'u': {
            int a = hex_(s->p[0]), b = hex_(s->p[1]), c1 = hex_(s->p[2]), d = hex_(s->p[3]);
            if (a < 0 || b < 0 || c1 < 0 || d < 0) { free(v); fail(err, err_n, "bad \\u escape"); return -1; }
            unsigned cp = (unsigned)((a << 12) | (b << 8) | (c1 << 4) | d);
            s->p += 4;
            if (cp >= 0xd800 && cp <= 0xdbff) {
                if (s->p[0] != '\\' || s->p[1] != 'u') { free(v); fail(err, err_n, "unpaired high surrogate in JSON string"); return -1; }
                int lo0 = hex_(s->p[2]), lo1 = hex_(s->p[3]);
                int lo2 = hex_(s->p[4]), lo3 = hex_(s->p[5]);
                if (lo0 < 0 || lo1 < 0 || lo2 < 0 || lo3 < 0) { free(v); fail(err, err_n, "bad low-surrogate escape"); return -1; }
                unsigned lo = (unsigned)((lo0 << 12) | (lo1 << 8) | (lo2 << 4) | lo3);
                if (lo < 0xdc00 || lo > 0xdfff) { free(v); fail(err, err_n, "unpaired high surrogate in JSON string"); return -1; }
                cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                s->p += 6;
            } else if (cp >= 0xdc00 && cp <= 0xdfff) {
                free(v); fail(err, err_n, "unpaired low surrogate in JSON string"); return -1;
            }
            if (cp == 0) { free(v); fail(err, err_n, "NUL is not supported in chat JSON strings"); return -1; }
            if (cp < 0x80) { if (putc_(&v, &n, &cap, (unsigned char)cp)) goto oom; }
            else if (cp < 0x800) {
                if (putc_(&v, &n, &cap, 0xc0 | (cp >> 6)) || putc_(&v, &n, &cap, 0x80 | (cp & 63))) goto oom;
            } else if (cp < 0x10000) {
                if (putc_(&v, &n, &cap, 0xe0 | (cp >> 12)) || putc_(&v, &n, &cap, 0x80 | ((cp >> 6) & 63)) || putc_(&v, &n, &cap, 0x80 | (cp & 63))) goto oom;
            } else {
                if (putc_(&v, &n, &cap, 0xf0 | (cp >> 18)) || putc_(&v, &n, &cap, 0x80 | ((cp >> 12) & 63)) || putc_(&v, &n, &cap, 0x80 | ((cp >> 6) & 63)) || putc_(&v, &n, &cap, 0x80 | (cp & 63))) goto oom;
            }
            break;
        }
        default: free(v); fail(err, err_n, "unknown JSON escape"); return -1;
        }
    }
    if (*s->p != '"') { free(v); fail(err, err_n, "unterminated JSON string"); return -1; }
    s->p++;
    if (putc_(&v, &n, &cap, 0)) goto oom;
    *out = v; return 0;
oom:
    free(v); fail(err, err_n, "out of memory parsing JSON string"); return -1;
}

static int parse_record(const char *line, K3ChatMessage *out, char *err, size_t err_n)
{
    JScan s = {line}; char *role = NULL, *content = NULL, *reasoning = NULL;
    int have_role = 0, have_content = 0, have_reason = 0;
    memset(out, 0, sizeof *out); jws(&s);
    if (*s.p++ != '{') { fail(err, err_n, "record is not a JSON object"); goto bad; }
    for (;;) {
        jws(&s); if (*s.p == '}') { s.p++; break; }
        char *key = NULL, *value = NULL;
        if (jstr(&s, &key, err, err_n) || (jws(&s), *s.p++ != ':') || (jws(&s), jstr(&s, &value, err, err_n))) {
            free(key); free(value); goto bad;
        }
        if (!strcmp(key, "role") && !have_role) { role = value; have_role = 1; value = NULL; }
        else if (!strcmp(key, "content") && !have_content) { content = value; have_content = 1; value = NULL; }
        else if (!strcmp(key, "reasoning_content") && !have_reason) { reasoning = value; have_reason = 1; value = NULL; }
        else { free(key); free(value); fail(err, err_n, "unknown or duplicate JSONL field"); goto bad; }
        free(key); free(value); jws(&s);
        if (*s.p == ',') { s.p++; continue; }
        if (*s.p == '}') { s.p++; break; }
        fail(err, err_n, "expected ',' or '}' in JSON object"); goto bad;
    }
    jws(&s); if (*s.p) { fail(err, err_n, "trailing data after JSON object"); goto bad; }
    int rid = role ? role_id(role) : -1;
    if (rid < 0 || !content || (rid != K3_CHAT_ASSISTANT && reasoning)) {
        fail(err, err_n, "record needs role/content; reasoning_content is assistant-only"); goto bad;
    }
    out->role = rid; out->content = content; out->reasoning_content = reasoning;
    free(role); return 0;
bad:
    free(role); free(content); free(reasoning); return -1;
}

int k3_chat_history_load(K3ChatHistory *h, const char *path, char *err, size_t err_n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { if (errno == ENOENT) return 0; fail(err, err_n, "cannot open %s: %s", path, strerror(errno)); return -1; }
    char *line = NULL; size_t n = 0, cap = 0; int line_no = 0, ch, rc = -1;
    while ((ch = fgetc(f)) != EOF) {
        if (ch == '\n') {
            line_no++; if (!n) { fail(err, err_n, "%s:%d is blank", path, line_no); goto done; }
            if (putc_(&line, &n, &cap, 0)) { fail(err, err_n, "out of memory reading history"); goto done; }
            K3ChatMessage m;
            if (parse_record(line, &m, err, err_n)) { char msg[256]; snprintf(msg, sizeof msg, "%s", err); fail(err, err_n, "%s:%d: %s", path, line_no, msg); goto done; }
            if (h->n == h->cap) { int nc = h->cap ? h->cap * 2 : 8; K3ChatMessage *v = (K3ChatMessage *)realloc(h->v, (size_t)nc * sizeof(*v)); if (!v) { k3_chat_message_free(&m); fail(err, err_n, "out of memory growing history"); goto done; } h->v = v; h->cap = nc; }
            h->v[h->n++] = m; n = 0; continue;
        }
        if (n >= 16 * 1024 * 1024) { fail(err, err_n, "%s:%d exceeds 16 MiB", path, line_no + 1); goto done; }
        if (putc_(&line, &n, &cap, (unsigned char)ch)) { fail(err, err_n, "out of memory reading history"); goto done; }
    }
    if (ferror(f)) { fail(err, err_n, "failed reading %s", path); goto done; }
    if (n) { line_no++; if (putc_(&line, &n, &cap, 0)) { fail(err, err_n, "out of memory reading history"); goto done; } K3ChatMessage m; if (parse_record(line, &m, err, err_n)) { char msg[256]; snprintf(msg, sizeof msg, "%s", err); fail(err, err_n, "%s:%d: %s", path, line_no, msg); goto done; } if (h->n == h->cap) { int nc = h->cap ? h->cap * 2 : 8; K3ChatMessage *v = (K3ChatMessage *)realloc(h->v, (size_t)nc * sizeof(*v)); if (!v) { k3_chat_message_free(&m); fail(err, err_n, "out of memory growing history"); goto done; } h->v = v; h->cap = nc; } h->v[h->n++] = m; }
    rc = k3_chat_history_validate(h, err, err_n);
done:
    free(line); fclose(f); if (rc) k3_chat_history_free(h); return rc;
}

static int jwrite(FILE *f, const char *s)
{
    if (fputc('"', f) == EOF) return -1;
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { if (fputc('\\', f) == EOF || fputc(c, f) == EOF) return -1; }
        else if (c == '\n') { if (fputs("\\n", f) == EOF) return -1; }
        else if (c == '\r') { if (fputs("\\r", f) == EOF) return -1; }
        else if (c == '\t') { if (fputs("\\t", f) == EOF) return -1; }
        else if (c < 0x20) { if (fprintf(f, "\\u%04x", c) < 0) return -1; }
        else if (fputc(c, f) == EOF) return -1;
    }
    return fputc('"', f) == EOF ? -1 : 0;
}

int k3_chat_history_save(const K3ChatHistory *h, const char *path, char *err, size_t err_n)
{
    if (k3_chat_history_validate(h, err, err_n)) return -1;
    char tmp[4096]; if (snprintf(tmp, sizeof tmp, "%s.tmp.XXXXXX", path) >= (int)sizeof tmp) { fail(err, err_n, "history path too long"); return -1; }
    int fd = mkstemp(tmp); if (fd < 0) { fail(err, err_n, "cannot create history temp file: %s", strerror(errno)); return -1; }
    FILE *f = fdopen(fd, "wb"); if (!f) { close(fd); unlink(tmp); fail(err, err_n, "cannot open history temp file"); return -1; }
    int rc = 0;
    for (int i = 0; i < h->n && !rc; i++) {
        const K3ChatMessage *m = &h->v[i];
        if (fputs("{\"role\":", f) == EOF || jwrite(f, role_name(m->role)) || fputs(",\"content\":", f) == EOF || jwrite(f, m->content)) rc = -1;
        if (!rc && m->role == K3_CHAT_ASSISTANT && m->reasoning_content) if (fputs(",\"reasoning_content\":", f) == EOF || jwrite(f, m->reasoning_content)) rc = -1;
        if (!rc && fputs("}\n", f) == EOF) rc = -1;
    }
    if (!rc && fflush(f) != 0) rc = -1;
    if (!rc && fsync(fd) != 0) rc = -1;
    if (fclose(f) != 0) rc = -1;
    if (!rc && rename(tmp, path) != 0) rc = -1;
    if (rc) { unlink(tmp); fail(err, err_n, "failed atomically writing %s: %s", path, strerror(errno)); return -1; }
    return 0;
}

/* ---- XTML segments --------------------------------------------------------- */
void k3_chat_segments_init(K3ChatSegments *s) { memset(s, 0, sizeof *s); }
void k3_chat_segments_free(K3ChatSegments *s)
{
    for (int i = 0; i < s->n; i++) free(s->v[i].text);
    free(s->v); memset(s, 0, sizeof *s);
}

static int seg(K3ChatSegments *s, const char *text, int special, char *err, size_t err_n)
{
    int len = (int)strlen(text); if (!len) return 0;
    if (s->n == s->cap) { int nc = s->cap ? s->cap * 2 : 32; K3ChatSegment *v = (K3ChatSegment *)realloc(s->v, (size_t)nc * sizeof(*v)); if (!v) { fail(err, err_n, "out of memory rendering XTML"); return -1; } s->v = v; s->cap = nc; }
    s->v[s->n].text = dup_(text); if (!s->v[s->n].text) { fail(err, err_n, "out of memory rendering XTML"); return -1; }
    s->v[s->n].len = len; s->v[s->n].allow_special = special; s->n++; return 0;
}
static int ctl(K3ChatSegments *s, const char *x, char *e, size_t n) { return seg(s, x, 1, e, n); }
static int txt(K3ChatSegments *s, const char *x, char *e, size_t n) { return seg(s, x, 0, e, n); }
static int open_tag(K3ChatSegments *s, const char *tag, const char *attrs, char *e, size_t n)
{ return ctl(s, "<|open|>", e, n) || txt(s, tag, e, n) || (attrs && txt(s, attrs, e, n)) || ctl(s, "<|sep|>", e, n); }
static int close_tag(K3ChatSegments *s, const char *tag, char *e, size_t n)
{ return ctl(s, "<|close|>", e, n) || txt(s, tag, e, n) || ctl(s, "<|sep|>", e, n); }
static int eom(K3ChatSegments *s, char *e, size_t n) { return ctl(s, "<|end_of_msg|>", e, n); }

int k3_chat_template_init(Tok *tok, K3ChatTemplate *t, char *err, size_t err_n)
{
    t->open_id = tok_id_of(tok, "<|open|>"); t->close_id = tok_id_of(tok, "<|close|>");
    t->sep_id = tok_id_of(tok, "<|sep|>"); t->eom_id = tok_id_of(tok, "<|end_of_msg|>");
    t->eos_id = tok_id_of(tok, "[EOS]");
    if (t->open_id < 0 || t->close_id < 0 || t->sep_id < 0 || t->eom_id < 0) {
        fail(err, err_n, "K3 tokenizer is missing required XTML control tokens"); return -1;
    }
    if (t->eom_id != 163586) { fail(err, err_n, "K3 end_of_msg id is %d, expected official id 163586", t->eom_id); return -1; }
    if (t->eos_id != 163585) { fail(err, err_n, "K3 [EOS] id is %d, expected official id 163585", t->eos_id); return -1; }
    return 0;
}

int k3_chat_render(const K3ChatHistory *h, int add_generation_prompt,
                   K3ChatSegments *out, char *err, size_t err_n)
{
    static const char thinking[] =
        "`thinking_effort` guides on how much to think in your thinking channel (not including the response channel), "
        "supported values include `low`, `medium`, `high`, and `max`.\n"
        "Now the system is invoked with `thinking_effort=max`.";
    k3_chat_segments_init(out);
    if (k3_chat_history_validate(h, err, err_n)) return -1;
    if (open_tag(out, "message", " role=\"system\" type=\"thinking-effort\"", err, err_n) || txt(out, thinking, err, err_n) || close_tag(out, "message", err, err_n) || eom(out, err, err_n)) goto bad;
    for (int i = 0; i < h->n; i++) {
        const K3ChatMessage *m = &h->v[i]; char attrs[64];
        snprintf(attrs, sizeof attrs, " role=\"%s\"", role_name(m->role));
        if (open_tag(out, "message", attrs, err, err_n)) goto bad;
        if (m->role == K3_CHAT_ASSISTANT) {
            if (open_tag(out, "think", NULL, err, err_n) || txt(out, m->reasoning_content ? m->reasoning_content : "", err, err_n) || close_tag(out, "think", err, err_n) || open_tag(out, "response", NULL, err, err_n) || txt(out, m->content, err, err_n) || close_tag(out, "response", err, err_n)) goto bad;
        } else if (txt(out, m->content, err, err_n)) goto bad;
        if (close_tag(out, "message", err, err_n) || eom(out, err, err_n)) goto bad;
    }
    if (add_generation_prompt && (open_tag(out, "message", " role=\"assistant\"", err, err_n) || open_tag(out, "think", NULL, err, err_n))) goto bad;
    return 0;
bad:
    k3_chat_segments_free(out); return -1;
}

int k3_chat_segments_text(const K3ChatSegments *s, char **text_out, int *len_out, char *err, size_t err_n)
{
    size_t n = 0; for (int i = 0; i < s->n; i++) n += (size_t)s->v[i].len;
    if (n > (size_t)INT32_MAX) { fail(err, err_n, "rendered XTML is too long"); return -1; }
    char *out = (char *)malloc(n + 1); if (!out) { fail(err, err_n, "out of memory joining XTML"); return -1; }
    size_t at = 0; for (int i = 0; i < s->n; i++) { memcpy(out + at, s->v[i].text, (size_t)s->v[i].len); at += (size_t)s->v[i].len; }
    out[at] = 0; *text_out = out; *len_out = (int)at; return 0;
}

int k3_chat_encode(Tok *tok, const K3ChatSegments *s, int *ids, int max, char *err, size_t err_n)
{
    int n = 0;
    for (int i = 0; i < s->n; i++) {
        int room = max - n; if (room <= 0) { fail(err, err_n, "rendered XTML exceeds token buffer"); return -1; }
        int got = tok_encode_mode(tok, s->v[i].text, s->v[i].len, ids + n, room, s->v[i].allow_special);
        n += got;
    }
    return n;
}

static int ids_for(Tok *tok, const char *text, int *out)
{ return tok_encode_mode(tok, text, (int)strlen(text), out, 64, 0); }
static int match(const int *ids, int n, int at, const int *pat, int pn)
{ return at + pn <= n && !memcmp(ids + at, pat, (size_t)pn * sizeof(*ids)); }
static char *decode_ids(Tok *tok, const int *ids, int n)
{
    size_t cap = 1;
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= tok->n_ids || !tok->id2str[ids[i]]) return NULL;
        cap += strlen(tok->id2str[ids[i]]);
        if (cap > (size_t)INT32_MAX) return NULL;
    }
    char *s = (char *)malloc(cap);
    if (!s) return NULL;
    int got = tok_decode(tok, ids, n, s, (int)cap - 1);
    s[got] = 0; return s;
}

int k3_chat_parse_assistant(Tok *tok, const K3ChatTemplate *t, const int *ids, int n,
                            K3ChatMessage *out, char *err, size_t err_n)
{
    int think[64], response[64], message[64], a[128], z[128];
    int nt = ids_for(tok, "think", think), nr = ids_for(tok, "response", response), nm = ids_for(tok, "message", message);
    int na = 0, nz = 0;
    a[na++] = t->close_id; memcpy(a + na, think, (size_t)nt * sizeof(int)); na += nt; a[na++] = t->sep_id; a[na++] = t->open_id; memcpy(a + na, response, (size_t)nr * sizeof(int)); na += nr; a[na++] = t->sep_id;
    z[nz++] = t->close_id; memcpy(z + nz, response, (size_t)nr * sizeof(int)); nz += nr; z[nz++] = t->sep_id; z[nz++] = t->close_id; memcpy(z + nz, message, (size_t)nm * sizeof(int)); nz += nm; z[nz++] = t->sep_id;
    int split = -1, end = -1;
    for (int i = 0; i < n; i++) if (match(ids, n, i, a, na)) { split = i; break; }
    if (split < 0) { fail(err, err_n, "assistant output has no <think> to <response> boundary"); return -1; }
    for (int i = split + na; i < n; i++) if (match(ids, n, i, z, nz)) { end = i; break; }
    /* The closure is followed by exactly one end id, and either declared id is accepted:
     * the released model ends its turn with [EOS], the template inserts <|end_of_msg|>. */
    if (end < 0 || end + nz != n - 1 || (ids[n - 1] != t->eom_id && ids[n - 1] != t->eos_id)) {
        fail(err, err_n, "assistant output is missing the official response/message closure and end id"); return -1;
    }
    /* Stored transcript text is re-encoded with special tokens disabled.  Accepting a
     * control id in either payload would therefore make a restart differ from the live
     * turn, so treat it as malformed instead of silently changing the conversation. */
    for (int i = 0; i < split; i++)
        if (ids[i] >= 0 && ids[i] < tok->n_ids && tok->id_added[ids[i]]) {
            fail(err, err_n, "assistant reasoning contains an unexpected control token"); return -1;
        }
    for (int i = split + na; i < end; i++)
        if (ids[i] >= 0 && ids[i] < tok->n_ids && tok->id_added[ids[i]]) {
            fail(err, err_n, "assistant response contains an unexpected control token"); return -1;
        }
    memset(out, 0, sizeof *out); out->role = K3_CHAT_ASSISTANT;
    out->reasoning_content = decode_ids(tok, ids, split);
    out->content = decode_ids(tok, ids + split + na, end - (split + na));
    if (!out->reasoning_content || !out->content) { k3_chat_message_free(out); fail(err, err_n, "out of memory decoding assistant output"); return -1; }
    return 0;
}
