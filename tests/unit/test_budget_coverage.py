"""Hardware estimates must stay within the shard coverage actually observed."""
import json
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class BudgetCoverage(unittest.TestCase):
    def run_budget(self, root):
        return subprocess.run([sys.executable, str(ROOT / "tools/budget.py"), str(root)],
                              capture_output=True, text=True, check=False)

    def test_no_experts_is_valid_partial_coverage(self):
        result = self.run_budget(ROOT / "tests/fixtures/st")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("no complete routed expert", result.stdout)
        self.assertNotIn("cached (", result.stdout)

    def test_fixture_uses_observed_expert_count_and_caps_capacity(self):
        result = self.run_budget(ROOT / "tests/fixtures/cache")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("24 observed experts", result.stdout)
        self.assertNotIn("896", result.stdout)
        rows = re.findall(r"of (\d+) cached \(([0-9.]+)%\)", result.stdout)
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(int(total) == 24 and float(percent) <= 100 for total, percent in rows))

    def test_incomplete_expert_does_not_project_a_full_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            header = {"language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_packed":
                      {"dtype": "U8", "shape": [8, 1], "data_offsets": [0, 8]}}
            raw = json.dumps(header).encode()
            (Path(tmp) / "partial.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 8)
            result = self.run_budget(tmp)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("incomplete", result.stdout)
            self.assertNotIn("cached (", result.stdout)


if __name__ == "__main__":
    unittest.main()
