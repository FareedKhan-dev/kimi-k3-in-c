/* Scalar MXFP4 matvec, grouped path.
 *
 * Same contract as k3_matmul_mxfp4's non-AVX group loop in
 * FareedKhan-dev/kimi-k3-in-c src/core/k3_ops.c:
 *   each group of 32 is dotted in double, then multiplied by its E8M0 scale.
 * The Android port does not change this kernel. This witness lets a browser
 * recompute the identical rounding and show the delta.
 *
 * E2M1 (OCP MX): low nibble = even element, high nibble = odd.
 *   0, 0.5, 1, 1.5, 2, 3, 4, 6, and the negatives.
 * E8M0 byte 255 is NaN and zeroes the group. Others are 2^(byte-127).
 */
#include <stdint.h>
#include <stdio.h>

static const float E2M1[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static uint32_t xs32(uint32_t *s)
{
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

/* 2^(sb-127) as an exact f32: the exponent field is sb itself. */
static float e8m0(unsigned char sb)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)sb << 23;
    return v.f;
}

static uint32_t fbits(float f)
{
    union { uint32_t u; float f; } v;
    v.f = f;
    return v.u;
}

enum { ROWS = 8, IN = 128, GROUP = 32 };

int main(void)
{
    unsigned char packed[ROWS][IN / 2];
    unsigned char scales[ROWS][IN / GROUP];
    float x[IN];
    float y[ROWS];

    uint32_t s = 0xA11CEu;
    for (int i = 0; i < IN; i++) {
        const uint32_t r = xs32(&s);
        x[i] = ((int)(r & 2047) - 1024) / 1024.0f;
    }
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < IN / 2; c++)
            packed[r][c] = (unsigned char)(xs32(&s) & 255);
        for (int g = 0; g < IN / GROUP; g++) {
            unsigned char sc = (unsigned char)(120 + (xs32(&s) & 15));
            if (((r * 4 + g) % 11) == 0) sc = 255;
            scales[r][g] = sc;
        }
    }

    double xd[IN];
    for (int i = 0; i < IN; i++) xd[i] = (double)x[i];

    for (int r = 0; r < ROWS; r++) {
        double acc = 0.0;
        for (int g = 0; g < IN / GROUP; g++) {
            const unsigned char sb = scales[r][g];
            const double scale = (sb == 255) ? 0.0 : (double)e8m0(sb);
            double sum = 0.0;
            const int base = g * GROUP;
            for (int i = 0; i < GROUP; i += 2) {
                const unsigned char byte = packed[r][(base + i) / 2];
                sum += (double)E2M1[byte & 0x0F] * xd[base + i];
                sum += (double)E2M1[byte >> 4] * xd[base + i + 1];
            }
            acc += sum * scale;
        }
        y[r] = (float)acc;
    }

    printf("{\n  \"rows\": %d,\n  \"in\": %d,\n  \"group\": %d,\n  \"seed\": 660462,\n  \"yBits\": [",
           ROWS, IN, GROUP);
    for (int r = 0; r < ROWS; r++)
        printf("%s%u", r ? ", " : "", fbits(y[r]));
    printf("]\n}\n");
    return 0;
}
