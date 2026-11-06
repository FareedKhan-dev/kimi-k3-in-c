/* k3_stream.h — pread(), but the bytes come from an object store.
 *
 * The upstream engine reads the trunk and each expert with a positioned
 * read. On a phone those bytes are not in a file. They are in S3, R2, B2,
 * or a gateway that already speaks S3 in front of Telegram file parts.
 * This is that read: one HTTP Range per slice, connection kept open,
 * body dropped by the caller after the matmul.
 *
 * Cleartext HTTP/1.1 on purpose. A gateway you run (the usual shape of a
 * "Telegram S3" front) terminates TLS. On device, the same Range header
 * goes through libcurl if you want TLS from the phone itself. Offsets do
 * not change.
 *
 * part <= 0 means "one range for the whole pread". Telegram's own
 * upload.getFile will not serve more than about 1 MiB, so pass
 * K3_ANDROID_PART_TELEGRAM there. An S3 gateway can take 8 MiB or more.
 */
#ifndef K3_STREAM_H
#define K3_STREAM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char    host[256];
    char    path[1024];
    int     port;
    int     fd;
    int64_t part;
} K3Stream;

/* Writes "bytes=START-END" (END inclusive), the header value only. */
int k3_range_header(int64_t off, int64_t n, char *buf, int buflen);

int k3_stream_open(K3Stream *s, const char *host, int port,
                   const char *path, int64_t part);
/* Like pread: exactly n bytes at off, or -1. A part cap splits the call
 * into several Range GETs on the same connection. */
int64_t k3_stream_pread(K3Stream *s, void *dst, int64_t n, int64_t off);
void k3_stream_close(K3Stream *s);

#ifdef __cplusplus
}
#endif

#endif /* K3_STREAM_H */
