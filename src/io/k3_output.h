/* k3_output.h - buffered writes are successful only after closing the stream. */
#ifndef K3_OUTPUT_H
#define K3_OUTPUT_H
#include <stdio.h>

static int k3_output_close(FILE *stream)
{
    int failed = ferror(stream);
    if (fclose(stream) != 0) failed = 1;
    return failed ? -1 : 0;
}
#endif
