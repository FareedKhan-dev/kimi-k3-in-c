"""Refuse incomplete replay data instead of computing misleading cache figures."""
import importlib.util
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is required for replay")
class ReplayInputs(unittest.TestCase):
    def run_replay(self, raw, *extra):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.bin"
            trace.write_bytes(raw)
            return subprocess.run([sys.executable, str(ROOT / "tools/sim_cache.py"),
                                   str(trace), *extra], capture_output=True, text=True, check=False)

    def test_bad_trace_and_measurement_inputs_are_refused(self):
        valid = struct.pack("<iiiiii", 1, 2, 1, 2, 1, 3)
        cases = [(b"", []), (valid + b"x", []), (struct.pack("<ii", -1, 2), []),
                 (struct.pack("<ii", 1, -1), []), (struct.pack("<ii", 1, 1 << 20), [])]
        cases += [(valid, ["--expert-bytes", value]) for value in ("0", "-1")]
        cases += [(valid, ["--disk-mbs", value]) for value in ("0", "-1", "nan", "inf")]
        for raw, extra in cases:
            with self.subTest(raw=raw, extra=extra):
                result = self.run_replay(raw, *extra)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertNotIn("LRU", result.stdout)
                self.assertNotIn("Traceback", result.stderr)

    def test_valid_trace_still_replays(self):
        result = self.run_replay(struct.pack("<iiiiii", 1, 2, 1, 2, 1, 3))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("3 requests, 2 distinct experts", result.stdout)
        self.assertIn("BELADY", result.stdout)


if __name__ == "__main__":
    unittest.main()
