/* k3_sampler.h - deterministic top-p/top-k sampling for chat only. */
#ifndef K3_SAMPLER_H
#define K3_SAMPLER_H

#include <stdint.h>

typedef struct {
    double temperature, top_p;
    int top_k;   /* <= 0 disables: the whole vocabulary stays eligible */
    uint64_t state, inc;
    int *ids;
    double *prob;
    int cap;
} K3Sampler;

void k3_sampler_init(K3Sampler *s, double temperature, double top_p, int top_k,
                     uint64_t seed, uint64_t turn);
void k3_sampler_free(K3Sampler *s);
int  k3_sampler_next(K3Sampler *s, const float *logits, int n_vocab,
                     int greedy, int *token_out);

#endif
