/* k3_chat.h - exact Kimi K3 XTML text-chat helpers. */
#ifndef K3_CHAT_H
#define K3_CHAT_H

#include <stddef.h>
#include "k3_tok.h"

enum { K3_CHAT_SYSTEM, K3_CHAT_USER, K3_CHAT_ASSISTANT };

typedef struct {
    int role;
    char *content;
    char *reasoning_content; /* assistant only; NULL means an empty think channel */
} K3ChatMessage;

typedef struct {
    K3ChatMessage *v;
    int n, cap;
} K3ChatHistory;

typedef struct {
    int open_id, close_id, sep_id, eom_id;
    int eos_id; /* tokenizer_config's [EOS]. The released checkpoint declares TWO end ids that
                 * disagree -- config.json says <|end_of_msg|> (163586), tokenizer_config.json
                 * says [EOS] (163585) -- and the model emits 163585 at the end of a turn. A
                 * turn may legitimately end with either.                                    */
} K3ChatTemplate;

/* Rendering policy, mirroring the two switches the checkpoint's own encoder exposes
 * (encoding_k3.build_chat_segments: `thinking`, `thinking_effort`).  The defaults are
 * what tokenization_kimi.apply_chat_template does when given nothing: thinking on with
 * thinking_effort="max".  With thinking off the encoder emits no thinking-effort system
 * message, renders prior assistant turns with no think channel at all, and opens the
 * response channel in the generation prompt, so the model answers directly.         */
typedef struct {
    int thinking;                /* 1: think channel + thinking-effort message; 0: response only */
    const char *thinking_effort; /* "low", "high" or "max"; only read when thinking is 1 */
} K3ChatOptions;

K3ChatOptions k3_chat_options_default(void);
int  k3_chat_options_validate(const K3ChatOptions *o, char *err, size_t err_n);

typedef struct {
    char *text;
    int len;
    int allow_special;
} K3ChatSegment;

typedef struct {
    K3ChatSegment *v;
    int n, cap;
} K3ChatSegments;

void k3_chat_history_init(K3ChatHistory *h);
void k3_chat_history_free(K3ChatHistory *h);
/* Discard all turns while retaining an initial system message, if any. */
int  k3_chat_history_reset(K3ChatHistory *h, char *err, size_t err_n);
int  k3_chat_history_add(K3ChatHistory *h, int role, const char *content,
                         const char *reasoning, char *err, size_t err_n);
int  k3_chat_history_load(K3ChatHistory *h, const char *path, char *err, size_t err_n);
int  k3_chat_history_save(const K3ChatHistory *h, const char *path, char *err, size_t err_n);
int  k3_chat_history_validate(const K3ChatHistory *h, char *err, size_t err_n);

int  k3_chat_template_init(Tok *tok, K3ChatTemplate *tmpl, char *err, size_t err_n);
void k3_chat_segments_init(K3ChatSegments *s);
void k3_chat_segments_free(K3ChatSegments *s);
int  k3_chat_render(const K3ChatHistory *h, int add_generation_prompt,
                    K3ChatSegments *out, char *err, size_t err_n); /* default options */
int  k3_chat_render_opts(const K3ChatHistory *h, int add_generation_prompt,
                         const K3ChatOptions *o, K3ChatSegments *out,
                         char *err, size_t err_n);
int  k3_chat_segments_text(const K3ChatSegments *s, char **text_out, int *len_out,
                           char *err, size_t err_n);
int  k3_chat_encode(Tok *tok, const K3ChatSegments *s, int *ids, int max,
                    char *err, size_t err_n);

/* Parse exactly one assistant completion after the generated channel opening: the
 * <think> opening with thinking on, the <response> opening with thinking off, in which
 * case reasoning_content is NULL. */
int  k3_chat_parse_assistant(Tok *tok, const K3ChatTemplate *tmpl,
                             const int *ids, int n, K3ChatMessage *out,
                             char *err, size_t err_n); /* default options */
int  k3_chat_parse_assistant_opts(Tok *tok, const K3ChatTemplate *tmpl,
                                  const K3ChatOptions *o, const int *ids, int n,
                                  K3ChatMessage *out, char *err, size_t err_n);
void k3_chat_message_free(K3ChatMessage *m);

#endif
