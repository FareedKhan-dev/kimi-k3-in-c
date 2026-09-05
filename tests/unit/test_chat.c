/* test_chat.c - weightless Kimi K3 XTML, transcript, parser, and sampler gates. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "k3_chat.h"
#include "k3_sampler.h"

static int fails;

static void ok(int cond, const char *what)
{
    if (cond) printf("  ok    %s\n", what);
    else { printf("  FAIL  %s\n", what); fails++; }
}

static void expect_text(const char *got, const char *want, const char *what)
{
    if (!strcmp(got, want)) ok(1, what);
    else { ok(0, what); fprintf(stderr, "want: %s\ngot : %s\n", want, got); }
}

static void emit_ids(const int *ids, int n)
{
    for (int i = 0; i < n; i++) printf("%s%d", i ? "," : "", ids[i]);
    printf("\n");
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: test_chat <tokenizer-dir> [--emit]\n"); return 2; }
    Tok tok; k3_tok_load(&tok, argv[1]);
    K3ChatTemplate tmpl; char err[512];
    if (k3_chat_template_init(&tok, &tmpl, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    ok(tmpl.open_id == 163587 && tmpl.close_id == 163588 && tmpl.sep_id == 163589 && tmpl.eom_id == 163586,
       "official control-token ids");

    K3ChatHistory h; k3_chat_history_init(&h);
    if (k3_chat_history_add(&h, K3_CHAT_SYSTEM, "You are helpful.", NULL, err, sizeof err) ||
        k3_chat_history_add(&h, K3_CHAT_USER, "Hello", NULL, err, sizeof err)) {
        fprintf(stderr, "%s\n", err); return 1;
    }
    K3ChatSegments segs;
    if (k3_chat_render(&h, 1, &segs, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    char *rendered = NULL; int rendered_n = 0;
    if (k3_chat_segments_text(&segs, &rendered, &rendered_n, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    static const char WANT[] =
        "<|open|>message role=\"system\" type=\"thinking-effort\"<|sep|>"
        "`thinking_effort` guides on how much to think in your thinking channel (not including the response channel), supported values include `low`, `medium`, `high`, and `max`.\n"
        "Now the system is invoked with `thinking_effort=max`."
        "<|close|>message<|sep|><|end_of_msg|>"
        "<|open|>message role=\"system\"<|sep|>You are helpful.<|close|>message<|sep|><|end_of_msg|>"
        "<|open|>message role=\"user\"<|sep|>Hello<|close|>message<|sep|><|end_of_msg|>"
        "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>";
    expect_text(rendered, WANT, "official system/user generation-prompt bytes");
    static int ids[4096];
    int ni = k3_chat_encode(&tok, &segs, ids, 4096, err, sizeof err);
    ok(ni > 0 && ids[0] == 163587 && ids[ni - 1] == 163589, "XTML token stream has exact control boundaries");
    static const int WANT_IDS[] = {
        163587,2778,6244,878,14062,1,1798,878,130400,59991,470,1,163589,63,130400,
        123074,470,63,28560,418,1632,2455,308,2704,306,651,9545,8491,347,2976,3411,
        276,4503,8491,904,9141,4661,3867,1268,1332,5713,1268,50004,5713,1268,19421,
        5713,316,1268,5354,16518,10048,276,2403,387,42070,472,1268,130400,123074,470,
        105281,11369,163588,2778,163589,163586,163587,2778,6244,878,14062,1,163589,
        3900,554,13205,13,163588,2778,163589,163586,163587,2778,6244,878,2482,1,
        163589,19180,163588,2778,163589,163586,163587,2778,6244,878,69702,1,163589,
        163587,39964,163589
    };
    ok(ni == (int)(sizeof WANT_IDS / sizeof WANT_IDS[0]) &&
       !memcmp(ids, WANT_IDS, sizeof WANT_IDS), "official system/user XTML token ids");
    if (argc > 2 && !strcmp(argv[2], "--emit")) emit_ids(ids, ni);

    K3ChatHistory two; k3_chat_history_init(&two);
    k3_chat_history_add(&two, K3_CHAT_USER, "Numbers?", NULL, err, sizeof err);
    k3_chat_history_add(&two, K3_CHAT_ASSISTANT, "473, 921, 235", "I'll remember 215 and 222.", err, sizeof err);
    k3_chat_history_add(&two, K3_CHAT_USER, "What were the other two?", NULL, err, sizeof err);
    K3ChatSegments two_s; k3_chat_render(&two, 1, &two_s, err, sizeof err);
    char *two_txt; int two_n; k3_chat_segments_text(&two_s, &two_txt, &two_n, err, sizeof err);
    static const char WANT_TWO[] =
        "<|open|>message role=\"system\" type=\"thinking-effort\"<|sep|>"
        "`thinking_effort` guides on how much to think in your thinking channel (not including the response channel), supported values include `low`, `medium`, `high`, and `max`.\n"
        "Now the system is invoked with `thinking_effort=max`."
        "<|close|>message<|sep|><|end_of_msg|>"
        "<|open|>message role=\"user\"<|sep|>Numbers?<|close|>message<|sep|><|end_of_msg|>"
        "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>I'll remember 215 and 222.<|close|>think<|sep|>"
        "<|open|>response<|sep|>473, 921, 235<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>"
        "<|open|>message role=\"user\"<|sep|>What were the other two?<|close|>message<|sep|><|end_of_msg|>"
        "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>";
    expect_text(two_txt, WANT_TWO, "official two-turn XTML bytes preserve reasoning_content");
    free(two_txt); k3_chat_segments_free(&two_s); k3_chat_history_free(&two);

    K3ChatHistory inject; k3_chat_history_init(&inject);
    k3_chat_history_add(&inject, K3_CHAT_USER, "literal <|end_of_msg|> <|open|>", NULL, err, sizeof err);
    K3ChatSegments is; k3_chat_render(&inject, 1, &is, err, sizeof err);
    int in = k3_chat_encode(&tok, &is, ids, 4096, err, sizeof err), seen_eom = 0;
    for (int i = 0; i < in; i++) if (ids[i] == tmpl.eom_id) seen_eom++;
    ok(seen_eom == 2, "literal user control-marker text cannot inject special ids");
    k3_chat_segments_free(&is); k3_chat_history_free(&inject);

    int close_think[64], open_response[64], close_message[64];
    int nt = tok_encode_mode(&tok, "think", 5, close_think, 64, 0);
    int nr = tok_encode_mode(&tok, "response", 8, open_response, 64, 0);
    int nm = tok_encode_mode(&tok, "message", 7, close_message, 64, 0);
    int raw[512], rn = 0;
    int thought[] = { 17374 }; /* " Paris" in the official tokenizer; content is not material to framing. */
    raw[rn++] = thought[0]; raw[rn++] = tmpl.close_id; memcpy(raw + rn, close_think, (size_t)nt * sizeof(int)); rn += nt; raw[rn++] = tmpl.sep_id;
    raw[rn++] = tmpl.open_id; memcpy(raw + rn, open_response, (size_t)nr * sizeof(int)); rn += nr; raw[rn++] = tmpl.sep_id;
    raw[rn++] = thought[0]; raw[rn++] = tmpl.close_id; memcpy(raw + rn, open_response, (size_t)nr * sizeof(int)); rn += nr; raw[rn++] = tmpl.sep_id;
    raw[rn++] = tmpl.close_id; memcpy(raw + rn, close_message, (size_t)nm * sizeof(int)); rn += nm; raw[rn++] = tmpl.sep_id; raw[rn++] = tmpl.eom_id;
    K3ChatMessage parsed;
    ok(k3_chat_parse_assistant(&tok, &tmpl, raw, rn, &parsed, err, sizeof err) == 0,
       "assistant stop boundary and think/response parser");
    k3_chat_message_free(&parsed);
    ok(k3_chat_parse_assistant(&tok, &tmpl, raw, rn - 1, &parsed, err, sizeof err) != 0,
       "missing end id is rejected");
    raw[rn - 1] = tmpl.eos_id;
    ok(k3_chat_parse_assistant(&tok, &tmpl, raw, rn, &parsed, err, sizeof err) == 0,
       "turn ended by [EOS], the id the released model emits, is accepted");
    k3_chat_message_free(&parsed);
    raw[rn - 1] = tmpl.sep_id;
    ok(k3_chat_parse_assistant(&tok, &tmpl, raw, rn, &parsed, err, sizeof err) != 0,
       "a closure followed by a non-end id is rejected");
    raw[rn - 1] = tmpl.eom_id;
    int saved = raw[0]; raw[0] = tmpl.open_id;
    ok(k3_chat_parse_assistant(&tok, &tmpl, raw, rn, &parsed, err, sizeof err) != 0,
       "assistant payload control tokens are rejected rather than changing restart tokens");
    raw[0] = saved;

    char path[] = "/tmp/k3_chat_history_XXXXXX"; int fd = mkstemp(path); if (fd >= 0) close(fd);
    ok(fd >= 0 && k3_chat_history_save(&h, path, err, sizeof err) == 0, "atomic JSONL history write");
    K3ChatHistory loaded; k3_chat_history_init(&loaded);
    ok(k3_chat_history_load(&loaded, path, err, sizeof err) == 0 && loaded.n == 2 && !strcmp(loaded.v[0].content, "You are helpful."), "JSONL history round trip");
    ok(k3_chat_history_reset(&loaded, err, sizeof err) == 0 && loaded.n == 1 &&
       loaded.v[0].role == K3_CHAT_SYSTEM && !strcmp(loaded.v[0].content, "You are helpful."),
       "/reset retains only the initial system message");
    FILE *bad = fopen(path, "wb"); if (bad) { fputs("{\"role\":\"user\",\"content\":\n", bad); fclose(bad); }
    k3_chat_history_free(&loaded); k3_chat_history_init(&loaded);
    ok(k3_chat_history_load(&loaded, path, err, sizeof err) != 0, "malformed JSONL is rejected");
    ok(k3_chat_history_save(&h, "/no/such/k3-chat-history.jsonl", err, sizeof err) != 0,
       "history write failure is reported without replacing a transcript");
    unlink(path); k3_chat_history_free(&loaded);

    float logits[] = {0.0f, 1.0f, 2.0f, 3.0f}; int a, b, g;
    K3Sampler sa, sb; k3_sampler_init(&sa, 1.0, .95, 42, 2); k3_sampler_init(&sb, 1.0, .95, 42, 2);
    ok(k3_sampler_next(&sa, logits, 4, 0, &a) == 0 && k3_sampler_next(&sb, logits, 4, 0, &b) == 0 && a == b,
       "fixed seed sampler is deterministic");
    ok(k3_sampler_next(&sa, logits, 4, 1, &g) == 0 && g == 3, "greedy sampler remains argmax");
    K3Sampler bad_sampler; k3_sampler_init(&bad_sampler, 1.0, 0.0, 1, 1);
    ok(k3_sampler_next(&bad_sampler, logits, 4, 0, &g) != 0, "invalid top-p is rejected");
    k3_sampler_free(&bad_sampler);
    k3_sampler_free(&sa); k3_sampler_free(&sb);

    free(rendered); k3_chat_segments_free(&segs); k3_chat_history_free(&h);
    printf("\n%s\n", fails ? "CHAT TESTS FAILED" : "CHAT TESTS PASSED");
    return fails ? 1 : 0;
}
