"""Optional synthetic check: diagnostics must not change generation or state."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

CHECKPOINT = os.environ.get("K3_TINY_CHECKPOINT")


@unittest.skipUnless(CHECKPOINT, "set K3_TINY_CHECKPOINT for synthetic CLI integration")
class TraceOptIn(unittest.TestCase):
    def test_tracing_preserves_tokens_logits_and_saved_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = []
            for enabled in (False, True):
                arm = root / str(enabled)
                arm.mkdir()
                command = [os.environ.get("K3_TEST_BIN", "./bin/k3"), CHECKPOINT,
                           "--ids", "3,7", "--gen", "2", "--cache-gb", "0.01",
                           "--incremental", "--save-state", str(arm / "state.bin"),
                           "--dump-logits", str(arm / "logits.bin"), "--out", str(arm / "run.json")]
                if enabled:
                    command += ["--dump-cache-trace", str(arm)]
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                results.append(json.loads((arm / "run.json").read_text()))
                self.assertEqual((arm / "expert_trace.bin").exists(), enabled)
            for key in ("generated_ids", "full_ids", "layers_completed", "expert_drops"):
                self.assertEqual(results[0][key], results[1][key])
            for name in ("logits.bin", "state.bin"):
                self.assertEqual((root / "False" / name).read_bytes(),
                                 (root / "True" / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
