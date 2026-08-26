/* test_model_stream.c - streamed model-level matrices must equal resident BF16 math.
 *
 * The full tables are 4.70 GB on Kimi K3, so the production path reads embedding rows
 * and lm_head chunks on demand. This fixture is intentionally tiny, but it crosses the
 * same safetensors/aligned-read boundary and demands bit-identical float32 outputs. */
#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3.h"
#include "k3_bind.h"
#include "k3_st.h"

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: test_model_stream <safetensors_fixture_dir>\n");
        return 2;
    }

    K3St st;
    if (k3_st_open(&st, argv[1]) != 0) return 1;

    K3Cfg c;
    memset(&c, 0, sizeof c);
    c.hidden = 7;
    c.vocab = 11;

    K3ModelStream ms;
    if (k3_model_stream_init(&ms, &st, &c) != 0) {
        k3_st_close(&st);
        return 1;
    }

    const K3Tensor *et = k3_st_find(&st, "language_model.model.embed_tokens.weight");
    const K3Tensor *ht = k3_st_find(&st, "language_model.lm_head.weight");
    uint16_t *embed = (uint16_t *)malloc((size_t)et->nbytes);
    uint16_t *head = (uint16_t *)malloc((size_t)ht->nbytes);
    if (!embed || !head || k3_st_read(&st, et, embed) != et->nbytes ||
        k3_st_read(&st, ht, head) != ht->nbytes) {
        fprintf(stderr, "could not read resident comparison matrices\n");
        free(embed); free(head); k3_model_stream_free(&ms); k3_st_close(&st);
        return 1;
    }

    int bad = 0;
    float got_row[7], ref_row[7];
    const int row = 6;
    if (k3_model_stream_embed_row(&ms, got_row, row) != 0) bad++;
    k3_embed_row(ref_row, embed, K3_WBF16, row, c.hidden);
    if (memcmp(got_row, ref_row, sizeof got_row) != 0) {
        fprintf(stderr, "streamed embedding row differs from resident gather\n");
        bad++;
    }

    const float x[7] = { 0.125f, -0.25f, 0.5f, 1.0f, -1.5f, 0.0625f, 0.75f };
    float got_logits[11], ref_logits[11];
    if (k3_model_stream_project(&ms, got_logits, x) != 0) bad++;
    k3_matmul_bf16(ref_logits, x, head, c.hidden, c.vocab);
    if (memcmp(got_logits, ref_logits, sizeof got_logits) != 0) {
        fprintf(stderr, "streamed lm_head logits differ from resident BF16 matmul\n");
        bad++;
    }

    if (ms.embed_bytes_read != (uint64_t)c.hidden * 2 ||
        ms.lm_head_bytes_read != (uint64_t)c.hidden * c.vocab * 2) {
        fprintf(stderr, "wrong I/O accounting: embed %llu head %llu\n",
                (unsigned long long)ms.embed_bytes_read,
                (unsigned long long)ms.lm_head_bytes_read);
        bad++;
    }
    if (k3_model_stream_embed_row(&ms, got_row, -1) == 0 ||
        k3_model_stream_embed_row(&ms, got_row, c.vocab) == 0) {
        fprintf(stderr, "out-of-range embedding row was accepted\n");
        bad++;
    }

    /* A corrupt index offset must become a hard short-read failure. The copy leaves the
     * opened index and fixture untouched while exercising the same aligned pread path a
     * truncated checkpoint would take; accounting must not claim bytes that never
     * arrived. */
    K3Tensor corrupt = *et;
    corrupt.off = INT64_C(1) << 60;
    ms.embed = &corrupt;
    const uint64_t before_short = ms.embed_bytes_read;
    if (k3_model_stream_embed_row(&ms, got_row, row) == 0 ||
        ms.embed_bytes_read != before_short) {
        fprintf(stderr, "short embedding read was accepted or counted\n");
        bad++;
    }
    ms.embed = et;

    /* Shape mismatch must fail before allocating or reading. */
    K3Cfg wrong = c;
    wrong.vocab++;
    K3ModelStream reject;
    if (k3_model_stream_init(&reject, &st, &wrong) == 0) {
        fprintf(stderr, "model-stream shape mismatch was accepted\n");
        k3_model_stream_free(&reject);
        bad++;
    }

    free(embed);
    free(head);
    k3_model_stream_free(&ms);
    k3_st_close(&st);
    printf("model stream parity: %s\n", bad ? "FAILED" : "PASSED");
    return bad ? 1 : 0;
}
