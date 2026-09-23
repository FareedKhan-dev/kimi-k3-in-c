/* test_json.c - truncated and malformed JSON must be refused loudly.
 *
 * WHY THIS FILE EXISTS
 *   third_party/json.h reads bytes the engine did not write itself
 *   (safetensors headers, configs, trunk manifests). A truncated file that
 *   parses as if it were complete is a silent-wrong-model case, and a read
 *   past the input terminator is a memory-safety bug. Each case below feeds
 *   the parser an exact-size copy so an overread is visible to sanitizers.
 *
 * usage: test_json
 *   needs no weights, no tokenizer files, no fixtures.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int allocations, fail_at, live;

static void *test_malloc(size_t n)
{
    if (++allocations == fail_at) return NULL;
    void *p = malloc(n);
    if (p) live++;
    return p;
}

static void *test_calloc(size_t n, size_t size)
{
    if (++allocations == fail_at) return NULL;
    void *p = calloc(n, size);
    if (p) live++;
    return p;
}

static void *test_realloc(void *p, size_t n)
{
    if (++allocations == fail_at) return NULL;
    const int was_null = p == NULL;
    void *q = realloc(p, n);
    if (q && was_null) live++;
    return q;
}

static void test_free(void *p)
{
    if (p) live--;
    free(p);
}

#define malloc test_malloc
#define calloc test_calloc
#define realloc test_realloc
#define free test_free
#include "json.h"
#undef malloc
#undef calloc
#undef realloc
#undef free

static int fails;
static void check(int ok, const char *what)
{
    if (!ok) { fprintf(stderr, "FAIL: %s\n", what); fails++; }
}

static void reject(const char *text, size_t n)
{
    /* Exact allocation: a read past the terminator must be visible to ASan. */
    char *copy = malloc(n + 1);
    if (!copy) exit(2);
    memcpy(copy, text, n);
    copy[n] = '\0';
    jval *v = json_parse(copy, NULL);
    if (v != NULL) fprintf(stderr, "FAIL: accepted %s\n", copy);
    else if (live != 0) fprintf(stderr, "FAIL: leak after refusing %s\n", copy);
    if (v != NULL || live != 0) fails++;
    json_free_tree(v);
    if (live != 0) fails++;
    free(copy);
}

int main(void)
{
    static const char *bad[] = {
        "", " ", "{", "{\"a\"", "{\"a\":", "{\"a\":1", "{\"a\":1,}",
        "{a:1}", "{\"a\" 1}", "{\"a\":1 \"b\":2}", "[", "[1", "[1,]",
        "[1 2]", "{\"a\":1} garbage", "true false", "nul", "truth", "NaN",
        "Infinity", "+1", "01", "-01", "-", ".1", "1.", "1e", "1e+",
        "1e9999", "\"unterminated", "\"\\", "\"\\q\"", "\"\\u\"",
        "\"\\u123\"", "\"\\u12xz\"", "\"\\uD800\"", "\"\\uDC00\"",
        "\"\\uD800\\u0041\"", "\"\\u0000\"", "\vnull"
    };
    /* Raw control bytes inside a string are refused (built at runtime so the
     * C source stays readable). */
    char raw_nl[] = { '"', 'l', 'i', 'n', 'e', '\n', 'b', 'r', 'e', 'a', 'k', '"', '\0' };
    char raw_tab[] = { '"', 'a', '\x01', 'b', '"', '\0' };

    size_t i;
    for (i = 0; i < sizeof bad / sizeof bad[0]; i++)
        reject(bad[i], strlen(bad[i]));
    reject(raw_nl, strlen(raw_nl));
    reject(raw_tab, strlen(raw_tab));

    static const char doc[] =
        "{\"a\":[0,1,2,3,4,5,6,7,8,9],\"b\":\""
        "abcdefghijklmnopqrstuvwxyz0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        "abcdefghijklmnopqrstuvwxyz0123456789\","
        "\"c\":{\"k0\":null,\"k1\":true,\"k2\":false,\"k3\":-1.25e+2,"
        "\"k4\":[],\"k5\":{},\"k6\":0,\"k7\":1,\"k8\":2},"
        "\"unicode\":\"\\u0041\\u00e9\\u20ac\\ud83d\\ude80\"}";
    for (i = 0; i < strlen(doc); i++) reject(doc, i);

    allocations = 0;
    fail_at = 0;
    jval *v = json_parse(doc, NULL);
    check(v != NULL, "complete document parses");
    const int count = allocations;
    if (v) {
        jval *a = json_get(v, "a"), *u = json_get(v, "unicode");
        jval *c = json_get(v, "c");
        jval *number = c ? json_get(c, "k3") : NULL;
        check(a && a->t == J_ARR && a->len == 10 && a->kids[9]->num == 9,
              "array values survive growth");
        check(number && number->t == J_NUM && number->num == -125,
              "signed fraction with exponent");
        check(u && u->t == J_STR &&
              !strcmp(u->str, "A\xc3\xa9\xe2\x82\xac\xf0\x9f\x9a\x80"),
              "unicode and surrogate pairs decode to UTF-8");
    }
    json_free_tree(v);
    check(live == 0, "successful tree can be freed");

    /* Every allocation failure must propagate as a clean refusal. */
    for (i = 1; i <= (size_t)count; i++) {
        allocations = 0;
        fail_at = (int)i;
        v = json_parse(doc, NULL);
        if (v != NULL) {
            fprintf(stderr, "FAIL: allocation %u did not propagate\n", (unsigned)i);
            fails++;
        }
        json_free_tree(v);
        if (live != 0) {
            fprintf(stderr, "FAIL: leak after allocation %u failed\n", (unsigned)i);
            fails++;
        }
    }
    fail_at = 0;

    /* Valid escapes still work. */
    v = json_parse("{\"e\":\"X\\\"Y\\\\Z\\/W\\b\\f\\n\\r\\t\"}", NULL);
    check(v != NULL, "valid escapes parse");
    if (v) {
        jval *e = json_get(v, "e");
        check(e && e->t == J_STR && !strcmp(e->str, "X\"Y\\Z/W\b\f\n\r\t"),
              "escape values decode");
    }
    json_free_tree(v);

    /* Nesting past the ceiling is refused instead of substituted. */
    {
        char deep[2 * (J_MAX_DEPTH + 1) + 2];
        memset(deep, '[', J_MAX_DEPTH + 1);
        deep[J_MAX_DEPTH + 1] = '0';
        memset(deep + J_MAX_DEPTH + 2, ']', J_MAX_DEPTH + 1);
        deep[sizeof deep - 1] = '\0';
        reject(deep, strlen(deep));
    }
    static const char *good[] = {"null", "true", "false", "0", "-0", "1e-3",
                                 "[]", "{}", "\"\"", " \t\r\n[1,2]\n"};
    for (i = 0; i < sizeof good / sizeof good[0]; i++) {
        v = json_parse(good[i], NULL);
        if (!v) {
            fprintf(stderr, "FAIL: refused valid %s\n", good[i]);
            fails++;
        }
        json_free_tree(v);
    }
    check(live == 0, "no parser allocations remain");
    printf("JSON: %d allocation failures exercised; %s\n", count,
           fails ? "FAILED" : "PASSED");
    return fails ? 1 : 0;
}
