/* Exercise buffered flush failures and pre-existing stream errors. */
#include <stdio.h>
#include <unistd.h>
#include "k3_output.h"

int main(int argc, char **argv)
{
    int fails = 0;
    FILE *f = tmpfile();
    if (!f) return 2;
    fputs("complete output", f);
    if (k3_output_close(f) != 0) fails++;

    f = tmpfile();
    if (!f) return 2;
    fputs("buffered output", f);
    /* Closing the backing descriptor injects a real flush error at fclose. */
    if (close(fileno(f)) != 0) return 2;
    if (k3_output_close(f) == 0) fails++;

    f = fopen(argc > 1 ? argv[1] : "tests/fixtures/ref_k3.json", "rb");
    if (!f) return 2;
    if (fputs("not writable", f) != EOF) return 2;
    if (k3_output_close(f) == 0) fails++;
    printf("output close contract: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
