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
} K3ChatTemplate;

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
                    K3ChatSegments *out, char *err, size_t err_n);
int  k3_chat_segments_text(const K3ChatSegments *s, char **text_out, int *len_out,
                           char *err, size_t err_n);
int  k3_chat_encode(Tok *tok, const K3ChatSegments *s, int *ids, int max,
                    char *err, size_t err_n);

/* Parse exactly one assistant completion after the generated <think> opening. */
int  k3_chat_parse_assistant(Tok *tok, const K3ChatTemplate *tmpl,
                             const int *ids, int n, K3ChatMessage *out,
                             char *err, size_t err_n);
void k3_chat_message_free(K3ChatMessage *m);

#endif
