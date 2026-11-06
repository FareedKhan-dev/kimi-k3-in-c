#define _POSIX_C_SOURCE 200112L

#include "k3_stream.h"

#include <errno.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

int k3_range_header(int64_t off, int64_t n, char *buf, int buflen)
{
    if (!buf || buflen < 8 || off < 0 || n <= 0) return -1;
    const int wr = snprintf(buf, (size_t)buflen, "bytes=%lld-%lld",
                            (long long)off, (long long)(off + n - 1));
    if (wr < 0 || wr >= buflen) return -1;
    return 0;
}

static int tainted(const char *s)
{
    return !s || strchr(s, '\r') || strchr(s, '\n') || strchr(s, ' ');
}

int k3_stream_open(K3Stream *s, const char *host, int port,
                   const char *path, int64_t part)
{
    if (!s || tainted(host) || tainted(path) || port <= 0 || port > 65535)
        return -1;
    if (strlen(host) >= sizeof s->host || strlen(path) >= sizeof s->path)
        return -1;
    if (path[0] != '/') return -1;
    memset(s, 0, sizeof *s);
    memcpy(s->host, host, strlen(host) + 1);
    memcpy(s->path, path, strlen(path) + 1);
    s->port = port;
    s->fd = -1;
    s->part = part;
    return 0;
}

void k3_stream_close(K3Stream *s)
{
    if (!s) return;
    if (s->fd >= 0) close(s->fd);
    s->fd = -1;
}

static int dial(K3Stream *s)
{
    char port[8];
    snprintf(port, sizeof port, "%d", s->port);

    struct addrinfo hints;
    memset(&hints, 0, sizeof hints);
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;

    struct addrinfo *res = NULL;
    if (getaddrinfo(s->host, port, &hints, &res) != 0) return -1;

    int fd = -1;
    for (struct addrinfo *p = res; p; p = p->ai_next) {
        fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        struct timeval tv;
        tv.tv_sec = 5;
        tv.tv_usec = 0;
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        if (connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) return -1;
    s->fd = fd;
    return 0;
}

static int send_all(int fd, const char *buf, int n)
{
    int sent = 0;
    while (sent < n) {
        const ssize_t r = send(fd, buf + sent, (size_t)(n - sent), MSG_NOSIGNAL);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return -1;
        sent += (int)r;
    }
    return 0;
}

static int read_full(int fd, unsigned char *dst, int64_t n)
{
    int64_t got = 0;
    while (got < n) {
        const size_t chunk = (size_t)((n - got) > (1 << 20) ? (1 << 20) : (n - got));
        const ssize_t r = recv(fd, dst + got, chunk, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return -1;
        got += r;
    }
    return 0;
}

static int64_t content_length(const char *hdr)
{
    const char *p = hdr;
    while (*p) {
        if (strncasecmp(p, "Content-Length:", 15) == 0)
            return strtoll(p + 15, NULL, 10);
        const char *nl = strstr(p, "\r\n");
        if (!nl) break;
        p = nl + 2;
    }
    return -1;
}

/* One Range GET. Returns 0 on success and leaves the connection up. */
static int one_range(K3Stream *s, unsigned char *dst, int64_t n, int64_t off)
{
    char range[64];
    char req[1400];
    char hdr[8192];
    if (k3_range_header(off, n, range, (int)sizeof range) != 0) return -1;

    const int wr = snprintf(req, sizeof req,
                            "GET %s HTTP/1.1\r\n"
                            "Host: %s:%d\r\n"
                            "Range: %s\r\n"
                            "Connection: keep-alive\r\n"
                            "User-Agent: k3-handset\r\n"
                            "\r\n",
                            s->path, s->host, s->port, range);
    if (wr < 0 || wr >= (int)sizeof req) return -1;

    for (int attempt = 0; attempt < 2; attempt++) {
        if (s->fd < 0 && dial(s) != 0) return -1;
        if (send_all(s->fd, req, wr) != 0) {
            k3_stream_close(s);
            continue;
        }

        int hn = 0;
        int bad = 0;
        while (hn < (int)sizeof hdr - 1) {
            const ssize_t r = recv(s->fd, hdr + hn, 1, 0);
            if (r < 0 && errno == EINTR) continue;
            if (r <= 0) { bad = 1; break; }
            hn += (int)r;
            hdr[hn] = 0;
            if (hn >= 4 && memcmp(hdr + hn - 4, "\r\n\r\n", 4) == 0) break;
        }
        if (bad || hn < 12 || strstr(hdr, "\r\n\r\n") == NULL) {
            k3_stream_close(s);
            continue;
        }

        int status = 0;
        sscanf(hdr, "HTTP/%*s %d", &status);
        const int64_t len = content_length(hdr);
        /* 206 is the only acceptable answer. A 200 would be the whole
         * object, which on this model is over a terabyte. */
        if (status != 206 || len != n) {
            k3_stream_close(s);
            return -1;
        }
        if (read_full(s->fd, dst, n) != 0) {
            k3_stream_close(s);
            continue;
        }
        return 0;
    }
    return -1;
}

int64_t k3_stream_pread(K3Stream *s, void *dst, int64_t n, int64_t off)
{
    if (!s || !dst || n < 0 || off < 0) return -1;
    if (n == 0) return 0;

    unsigned char *out = dst;
    int64_t got = 0;
    while (got < n) {
        int64_t chunk = n - got;
        if (s->part > 0 && chunk > s->part) chunk = s->part;
        if (one_range(s, out + got, chunk, off + got) != 0)
            return -1;
        got += chunk;
    }
    return got;
}
