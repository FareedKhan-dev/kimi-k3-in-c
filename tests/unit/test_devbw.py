"""Adversarial checks for the device benchmark, with no large-file I/O."""
import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "devbw", Path(__file__).resolve().parents[2] / "tools" / "devbw.py"
)
devbw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(devbw)


class DeviceBandwidth(unittest.TestCase):
    def measure(self, returns, size=16, qd=1):
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"x" * size)
            f.flush()
            with patch.object(devbw, "SZ", 16), patch.object(devbw, "N", 1), \
                 patch.object(devbw.os, "O_DIRECT", 0, create=True), \
                 patch.object(devbw.os, "preadv", side_effect=returns, create=True), \
                 patch.object(devbw.time, "perf_counter", side_effect=[1.0, 3.0]):
                return devbw.measure(f.name, qd)

    def test_eof_is_not_full_throughput(self):
        with self.assertRaisesRegex(OSError, "short read"):
            self.measure([8, 0])

    def test_small_file_is_refused(self):
        with self.assertRaisesRegex(ValueError, "smaller"):
            self.measure([0], size=4)

    def test_zero_queue_depth_is_refused(self):
        with self.assertRaisesRegex(ValueError, "queue depth"):
            self.measure([], qd=0)

    def test_partial_reads_are_completed(self):
        self.assertEqual(self.measure([8, 8]), 16 / 2 / 1e6)

    def test_buffered_fallback_is_refused(self):
        with patch.object(devbw.os, "O_DIRECT", None, create=True):
            with self.assertRaisesRegex(RuntimeError, "O_DIRECT"):
                devbw.measure("unused", 1)

    def test_main_reports_failure_status(self):
        with patch.object(devbw.sys, "argv", ["devbw.py", "fixture"]), \
             patch.object(devbw, "measure", side_effect=OSError("read failed")), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(devbw.main(), 1)


if __name__ == "__main__":
    unittest.main()
