/* test_sampler.c - the chat sampler, without the tokenizer.
 *
 * WHY THIS FILE EXISTS
 *   The sampler gates used to live at the end of test_chat.c, after the
 *   tokenizer load. Without the released vocabulary the whole binary,
 *   sampler checks included, was skipped: the one component with no
 *   legitimate reason to need a checkpoint was gated behind one. These
 *   checks need nothing but the sampler itself, so they run on every
 *   machine, in both build systems and under sanitizers.
 *
 * WHAT IS CHECKED
 *   Every property is deterministic, never statistical: a seed that must
 *   replay exactly, greedy decoding that never touches the stream, masked
 *   logits that can never win, and invalid configurations that must be
 *   refused rather than sampled from.
 *
 * usage: test_sampler
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "k3_sampler.h"

static int fails;

static void ok(int cond, const char *what)
{
    if (cond) printf("  ok    %s\n", what);
    else { printf("  FAIL  %s\n", what); fails++; }
}

int main(void)
{
    float logits[] = {0.0f, 1.0f, 2.0f, 3.0f};
    int a, b, g, i;

    /* Same seed and turn replays the same 50-draw sequence exactly. */
    {
        K3Sampler sa, sb;
        int same = 1;
        k3_sampler_init(&sa, 0.7, 0.9, 1234, 5);
        k3_sampler_init(&sb, 0.7, 0.9, 1234, 5);
        for (i = 0; i < 50; i++) {
            k3_sampler_next(&sa, logits, 4, 0, &a);
            k3_sampler_next(&sb, logits, 4, 0, &b);
            if (a != b) same = 0;
        }
        ok(same, "fixed seed sampler replays a 50-draw sequence");
        k3_sampler_free(&sa);
        k3_sampler_free(&sb);
    }

    /* Greedy is argmax, and never advances the stream: two greedy draws
     * followed by a sampled one match a single sampled draw. */
    {
        K3Sampler sa, sb;
        int x[3], y;
        k3_sampler_init(&sa, 1.0, 0.95, 42, 2);
        k3_sampler_init(&sb, 1.0, 0.95, 42, 2);
        ok(k3_sampler_next(&sa, logits, 4, 1, &g) == 0 && g == 3,
           "greedy sampler remains argmax");
        k3_sampler_next(&sa, logits, 4, 1, &x[0]);
        k3_sampler_next(&sa, logits, 4, 1, &x[1]);
        k3_sampler_next(&sa, logits, 4, 0, &x[2]);
        k3_sampler_next(&sb, logits, 4, 0, &y);
        ok(x[2] == y, "greedy draws do not advance the stream");
        k3_sampler_free(&sa);
        k3_sampler_free(&sb);
    }

    /* A masked logit has zero mass and sorts last, so it can never win,
     * however many tokens are drawn. */
    {
        float masked[] = {0.0f, 1.0f, 2.0f, -1e30f};
        K3Sampler s;
        int picked = 0;
        k3_sampler_init(&s, 1.0, 1.0, 7, 0);
        for (i = 0; i < 20000; i++) {
            k3_sampler_next(&s, masked, 4, 0, &a);
            if (a == 3) picked = 1;
        }
        ok(!picked, "masked -inf logit is never selected in 20000 draws");
        k3_sampler_free(&s);
    }

    /* One logit is one outcome. */
    {
        float one[] = {5.0f};
        K3Sampler s;
        k3_sampler_init(&s, 1.0, 0.95, 1, 1);
        ok(k3_sampler_next(&s, one, 1, 0, &a) == 0 && a == 0,
           "single-logit sampler always picks it");
        k3_sampler_free(&s);
    }

    /* Invalid configurations are refused, never sampled from. */
    {
        K3Sampler bad;
        k3_sampler_init(&bad, 1.0, 0.0, 1, 1);
        ok(k3_sampler_next(&bad, logits, 4, 0, &g) != 0, "top-p 0 is rejected");
        k3_sampler_free(&bad);
        k3_sampler_init(&bad, 1.0, 1.5, 1, 1);
        ok(k3_sampler_next(&bad, logits, 4, 0, &g) != 0, "top-p above 1 is rejected");
        k3_sampler_free(&bad);
        k3_sampler_init(&bad, 0.0, 0.95, 1, 1);
        ok(k3_sampler_next(&bad, logits, 4, 0, &g) != 0, "temperature 0 is rejected");
        k3_sampler_free(&bad);
        k3_sampler_init(&bad, NAN, 0.95, 1, 1);
        ok(k3_sampler_next(&bad, logits, 4, 0, &g) != 0, "NaN temperature is rejected");
        k3_sampler_free(&bad);
        k3_sampler_init(&bad, 1.0, 0.95, 1, 1);
        ok(k3_sampler_next(&bad, logits, 0, 0, &g) != 0, "empty vocabulary is rejected");
        ok(k3_sampler_next(&bad, NULL, 4, 0, &g) != 0, "NULL logits are rejected");
        ok(k3_sampler_next(&bad, logits, 4, 0, NULL) != 0, "NULL out is rejected");
        k3_sampler_free(&bad);
    }

    printf("\n%s\n", fails ? "SAMPLER TESTS FAILED" : "SAMPLER TESTS PASSED");
    return fails ? 1 : 0;
}
