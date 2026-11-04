#!/usr/bin/env python
"""
devbw.py - measure a device the way the ENGINE actually reads it.

WHY NOT dd
    dd is one sequential stream at queue depth 1. The expert path is random 17.55 MB
    O_DIRECT reads issued up to 16 at a time by the MoE block's batch prefetch. Those are
    different workloads, and on network-backed storage they differ by a large factor, so
    a dd number cannot tell you whether a device is the bottleneck for THIS engine.

WHAT IT REPORTS
    Achieved MB/s for expert-sized random O_DIRECT reads at queue depth 1 and at queue
    depth 16, the depths the engine uses without and with the batch prefetch. The gap
    between the two is the headroom the prefetch can convert into speed on this device;
    if they are equal, the device does not reward depth and the prefetch buys nothing.

usage: devbw.py <file> [<file> ...]
"""
from __future__ import annotations

import ctypes
import os
import random
import sys
import threading
import time

SZ = 17547264            # one real Kimi K3 routed expert, to the byte
ALIGN = 4096
N = 40                   # reads per measurement


def aligned_buf(n: int):
    """O_DIRECT needs the destination aligned too, not just offset and length."""
    raw = ctypes.create_string_buffer(n + ALIGN)
    addr = ctypes.addressof(raw)
    off = (-addr) % ALIGN
    return raw, (ctypes.c_char * n).from_buffer(raw, off)


def measure(path: str, qd: int) -> float:
    if qd < 1:
        raise ValueError("queue depth must be positive")
    if getattr(os, "O_DIRECT", None) is None or not callable(getattr(os, "preadv", None)):
        raise RuntimeError("O_DIRECT and preadv are required; use Linux for this benchmark")
    fsz = os.path.getsize(path)
    if fsz < SZ:
        raise ValueError("file is smaller than one expert-sized read")
    span = (fsz - SZ) // ALIGN + 1
    offs = [random.randrange(0, span) * ALIGN for _ in range(N)]
    err = []

    def worker(k: int):
        try:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        except OSError as e:
            err.append(e); return
        try:
            _raw, buf = aligned_buf(SZ)
            mv = memoryview(buf).cast("B")
            for i in range(k, N, qd):
                got, want = 0, SZ
                while got < want:
                    r = os.preadv(fd, [mv[got:want]], offs[i] + got)
                    if r <= 0:
                        raise OSError("short read: expected %d bytes, got %d" % (want, got))
                    got += r
        except Exception as e:
            err.append(e)
        finally:
            os.close(fd)

    t0 = time.perf_counter()
    th = [threading.Thread(target=worker, args=(k,)) for k in range(qd)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    dt = time.perf_counter() - t0
    if err:
        raise err[0]
    return N * SZ / dt / 1e6


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    print("expert-sized (%d B) random O_DIRECT reads, %d per measurement\n" % (SZ, N))
    failed = False
    for path in sys.argv[1:]:
        for qd in (1, 4, 16):
            try:
                print("  %-46s QD%-3d %7.0f MB/s" % (path, qd, measure(path, qd)))
            except Exception as e:
                failed = True
                print("  %-46s QD%-3d FAILED: %s" % (path, qd, e))
        print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
