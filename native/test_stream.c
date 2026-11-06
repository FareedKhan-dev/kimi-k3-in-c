#include "k3_stream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;

static void expect(int cond, const char *msg)
{
    if (!cond) {
        fprintf(stderr, "FAIL %s\n", msg);
        g_fail++;
    }
}

static unsigned char at(int i)
{
    return (unsigned char)((i * 7 + 3) & 255);
}

int main(int argc, char **argv)
{
    char hdr[64];
    expect(k3_range_header(1000, 64, hdr, (int)sizeof hdr) == 0, "header writes");
    expect(strcmp(hdr, "bytes=1000-1063") == 0, "inclusive end");
    expect(k3_range_header(-1, 4, hdr, (int)sizeof hdr) != 0, "reject negative");
    expect(k3_stream_open(NULL, "127.0.0.1", 80, "/x", 0) != 0, "null stream");

    if (argc < 4) {
        fprintf(stderr, "usage: test_stream host port path\n");
        return 1;
    }

    K3Stream s;
    const int port = atoi(argv[2]);
    /* 1000-byte parts, so a 2500-byte read is three Range GETs. */
    expect(k3_stream_open(&s, argv[1], port, argv[3], 1000) == 0, "open");

    unsigned char small[64];
    const int64_t n1 = k3_stream_pread(&s, small, 64, 1000);
    expect(n1 == 64, "first pread length");
    int mismatch = 0;
    for (int i = 0; i < 64; i++)
        if (small[i] != at(1000 + i)) mismatch++;
    expect(mismatch == 0, "first pread bytes");

    unsigned char big[2500];
    const int64_t n2 = k3_stream_pread(&s, big, 2500, 50);
    expect(n2 == 2500, "split pread length");
    mismatch = 0;
    for (int i = 0; i < 2500; i++)
        if (big[i] != at(50 + i)) mismatch++;
    expect(mismatch == 0, "split pread bytes, three ranges");

    /* Connection was reused or redialed. Either is correct. A third read
     * at the end of the object must still match. */
    unsigned char tail[4];
    const int64_t n3 = k3_stream_pread(&s, tail, 4, 199996);
    expect(n3 == 4, "tail length");
    expect(tail[0] == at(199996) && tail[3] == at(199999), "tail bytes");

    k3_stream_close(&s);

    if (g_fail) {
        fprintf(stderr, "%d failure(s)\n", g_fail);
        return 1;
    }
    printf("k3_stream: range pread matched\n");
    return 0;
}
