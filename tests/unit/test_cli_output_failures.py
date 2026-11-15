"""Optional tiny-checkpoint CLI checks for requested artifact write failures.

Set K3_TINY_CHECKPOINT to tools/make_tiny_checkpoint.py's generated directory.
"""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

CHECKPOINT = os.environ.get("K3_TINY_CHECKPOINT")


@unittest.skipUnless(CHECKPOINT, "set K3_TINY_CHECKPOINT for synthetic CLI integration")
class OutputFailures(unittest.TestCase):
    def run_model(self, output, *extra):
        return subprocess.run(
            [os.environ.get("K3_TEST_BIN", "./bin/k3"), CHECKPOINT,
             "--ids", "3,7", "--gen", "1", "--cache-gb", "0.01",
             "--out", str(output), *map(str, extra)],
            capture_output=True, text=True, check=False,
        )

    def test_successful_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traces = root / "traces"
            traces.mkdir()
            result = self.run_model(root / "run.json", "--dump-logits", root / "logits.bin",
                                    "--dump-cache-trace", traces)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(json.loads((root / "run.json").read_text())["generated_ids"]), 1)
            self.assertEqual((root / "logits.bin").stat().st_size, 256 * 4)
            self.assertTrue((traces / "expert_hist.json").is_file())
            self.assertGreater((traces / "expert_trace.bin").stat().st_size, 0)

    def test_unwritable_requested_paths_exit_three(self):
        for kind in ("out", "logits", "trace"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                missing = root / "missing" / "artifact"
                output = missing if kind == "out" else root / "run.json"
                extra = [] if kind == "out" else ["--dump-logits" if kind == "logits"
                                                  else "--dump-cache-trace", missing]
                result = self.run_model(output, *extra)
                self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
                self.assertIn("cannot", result.stderr)
                self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
