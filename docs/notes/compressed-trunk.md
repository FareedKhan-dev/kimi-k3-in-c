# Compressed trunk: investigated, shelved, with the measurement that decided it

## The idea

The trunk is bf16. A bf16 value's high byte (sign + exponent) carries about 2.8 bits of
real information on this checkpoint: roughly a dozen byte values cover 99.9% of all
weights. Entropy-coding just that byte plane, and storing the mantissa byte raw (it
measures as noise), shrinks the packed trunk to about 67% of its size. Because the decoded
bytes are the checkpoint's own bytes, it is lossless and the engine's output is unchanged.
At a streamed budget, fewer trunk bytes is fewer seconds per token.

## What was built and proven

A canonical length-limited Huffman codec, encoder in Python (`huf_encode.py.shelved`) and a
decode-only C header (`k3_huf.h.shelved`). On real trunk bytes it round-trips byte-exactly,
Python-encoded and C-decoded, at a measured **1.45x** (ratio 0.689) on a 4 MB sample. The
format, the Kraft check, and the canonical code assignment all work.

## The measurement that shelved it

Decode speed. A hand-rolled scalar Huffman decoder reached only **~0.31 GB/s per core** after
two optimisation passes. To keep a compressed trunk fed on a 3 GB/s laptop SSD at that rate
needs roughly ten cores decoding, which laptops do not have. So compression built this way
**helps many-core machines, which have the RAM to not need it, and starves the laptops that
do**. That is backwards from the goal.

A fast entropy decoder exists: FSE/Huff0 measured 1724 MB/s on the same exponent-plane bytes.
But even at that speed it needs two dedicated cores to match a laptop SSD, and vendoring it is
about two thousand lines against a project whose identity is a 176 KB binary with no
dependencies. The payoff did not justify the weight, for the machines that matter.

## If revisited

The unlock is a decoder in the 1 to 1.5 GB/s/core range that stays small: four interleaved
bitstreams per stripe for instruction-level parallelism, or a double-symbol table, are the
standard routes and would let the encoder stay as-is. The prototypes here are a correct
starting point; the format round-trips today. Until a decoder that fast is in hand, the
streamed-trunk speed lever is pinning (RAM-first `--preset auto`) and the resident int8 draft,
not compression.
