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
        k3_sampler_init(&sa, 0.7, 0.9, 0, 1234, 5);
        k3_sampler_init(&sb, 0.7, 0.9, 0, 1234, 5);
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
        k3_sampler_init(&sa, 1.0, 0.95, 0, 42, 2);
        k3_sampler_init(&sb, 1.0, 0.95, 0, 42, 2);
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
        k3_sampler_init(&s, 1.0, 1.0, 0, 7, 0);
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
        k3_sampler_init(&s, 1.0, 0.95, 0, 1, 1);
        ok(k3_sampler_next(&s, one, 1, 0, &a) == 0 && a == 0,
           "single-logit sampler always picks it");
        k3_sampler_free(&s);
    }

    /* Top-k: only the K most probable ids stay eligible for the nucleus that follows.
     * Off, wider than the vocabulary, or negative must replay the disabled stream
     * exactly, so adding the option changes nothing for anyone who does not ask. */
    {
        K3Sampler s_full, s_wide, s_neg, s_one, s_two, s_narrow;
        int same = 1, same_neg = 1, only_best = 1, in_window = 1, seen2 = 0, seen3 = 0;
        k3_sampler_init(&s_full, 1.0, 0.95, 0, 42, 2);
        k3_sampler_init(&s_wide, 1.0, 0.95, 99, 42, 2);
        k3_sampler_init(&s_neg, 1.0, 0.95, -5, 42, 2);
        for (i = 0; i < 20; i++) {
            k3_sampler_next(&s_full, logits, 4, 0, &a);
            k3_sampler_next(&s_wide, logits, 4, 0, &b);
            k3_sampler_next(&s_neg, logits, 4, 0, &g);
            if (a != b) same = 0;
            if (a != g) same_neg = 0;
        }
        ok(same, "top-k wider than the vocabulary replays disabled exactly");
        ok(same_neg, "negative top-k replays disabled exactly");
        k3_sampler_free(&s_full); k3_sampler_free(&s_wide); k3_sampler_free(&s_neg);

        k3_sampler_init(&s_one, 1.0, 1.0, 1, 7, 0);
        for (i = 0; i < 200; i++) {
            k3_sampler_next(&s_one, logits, 4, 0, &a);
            if (a != 3) only_best = 0;
        }
        ok(only_best, "top-k 1 always picks the argmax");
        k3_sampler_free(&s_one);

        k3_sampler_init(&s_two, 1.0, 1.0, 2, 7, 0);
        for (i = 0; i < 2000; i++) {
            k3_sampler_next(&s_two, logits, 4, 0, &a);
            if (a != 2 && a != 3) in_window = 0;
            if (a == 2) seen2 = 1;
            if (a == 3) seen3 = 1;
        }
        ok(in_window, "top-k 2 never picks outside its window");
        ok(seen2 && seen3, "top-k 2 still samples both survivors");
        k3_sampler_free(&s_two);

        k3_sampler_init(&s_narrow, 1.0, 0.5, 2, 7, 0);
        only_best = 1;
        for (i = 0; i < 200; i++) {
            k3_sampler_next(&s_narrow, logits, 4, 0, &a);
            if (a != 3) only_best = 0;
        }
        ok(only_best, "a top-p nucleus stays inside the top-k window");
        k3_sampler_free(&s_narrow);
    }

    /* Nucleus mass is relative to the surviving top-k distribution. With
     * four tied logits, top-k=2 gives each survivor mass 1/2, so top-p=.4
     * must retain exactly the first one, rather than both at original 1/4. */
    {
        float tied[] = {0.0f, 0.0f, 0.0f, 0.0f};
        K3Sampler s;
        int only_first = 1;
        k3_sampler_init(&s, 1.0, 0.4, 2, 7, 0);
        for (i = 0; i < 200; i++) {
            if (k3_sampler_next(&s, tied, 4, 0, &a) || a != 0)
                only_first = 0;
        }
        ok(only_first, "top-p is relative to the retained top-k mass");
        k3_sampler_free(&s);

        /* Unequal mass: original probabilities are 4/10, 3/10, 2/10,
         * 1/10. After top-k=2 the best token has mass 4/7 > .5. */
        float weighted[] = {logf(4.0f), logf(3.0f), logf(2.0f), 0.0f};
        only_first = 1;
        k3_sampler_init(&s, 1.0, 0.5, 2, 7, 0);
        for (i = 0; i < 200; i++) {
            if (k3_sampler_next(&s, weighted, 4, 0, &a) || a != 0)
                only_first = 0;
        }
        ok(only_first, "normalized nucleus handles unequal top-k probabilities");
        k3_sampler_free(&s);
    }

    /* Invalid configurations are refused, never sampled from. */
    {
        K3Sampler bad;
        k3_sampler_init(&bad, 1.0, 0.0, 0, 1, 1);
        ok(k3_sampler_next(&bad, logits, 4, 0, &g) != 0, "top-p 0 is rejected");
        k3_sampler_free(&bad);
        k3_sampler_init(&bad, 1.0, 1.5, 0, 1, 1);
        ok(k3_sampler_next(&bad, logits, 4, 0, &g) != 0, "top-p above 1 is rejected");
        k3_sampler_free(&bad);
        k3_sampler_init(&bad, 0.0, 0.95, 0, 1, 1);
        ok(k3_sampler_next(&bad, logits, 4, 0, &g) != 0, "temperature 0 is rejected");
        k3_sampler_free(&bad);
        k3_sampler_init(&bad, NAN, 0.95, 0, 1, 1);
        ok(k3_sampler_next(&bad, logits, 4, 0, &g) != 0, "NaN temperature is rejected");
        k3_sampler_free(&bad);
        k3_sampler_init(&bad, 1.0, 0.95, 0, 1, 1);
        ok(k3_sampler_next(&bad, logits, 0, 0, &g) != 0, "empty vocabulary is rejected");
        ok(k3_sampler_next(&bad, NULL, 4, 0, &g) != 0, "NULL logits are rejected");
        ok(k3_sampler_next(&bad, logits, 4, 0, NULL) != 0, "NULL out is rejected");
        k3_sampler_free(&bad);
    }

    printf("\n%s\n", fails ? "SAMPLER TESTS FAILED" : "SAMPLER TESTS PASSED");
    return fails ? 1 : 0;
}
