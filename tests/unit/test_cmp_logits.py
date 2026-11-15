"""Exercise the comparison verdict with synthetic vectors, not model weights."""
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

spec = importlib.util.spec_from_file_location(
    "cmp_logits", Path(__file__).resolve().parents[2] / "tools" / "cmp_logits.py"
)
cmp_logits = importlib.util.module_from_spec(spec)
if np is not None:
    spec.loader.exec_module(cmp_logits)


@unittest.skipIf(np is None, "NumPy is required for the logit comparator")
class LogitComparison(unittest.TestCase):
    def compare(self, values, extra=b"", json_input=False, reference=None):
        with tempfile.TemporaryDirectory() as tmp:
            vector = np.array(values, dtype=np.float32)
            bits = vector.view(np.uint32).tolist()
            ref = Path(tmp) / "ref.json"
            ref_bits = bits if reference is None else np.array(reference, dtype=np.float32).view(np.uint32).tolist()
            ref.write_text(json.dumps({"logits_bits": ref_bits, "prompt_ids": [1]}))
            actual = Path(tmp) / ("actual.json" if json_input else "actual.bin")
            if json_input:
                actual.write_text(json.dumps({"logits_bits": bits, "prompt_ids": [1]}))
            else:
                actual.write_bytes(vector.tobytes() + extra)
            output = io.StringIO()
            with patch.object(sys, "argv", ["cmp_logits.py", str(actual), str(ref)]), \
                 redirect_stdout(output):
                rc = cmp_logits.main()
            return rc, output.getvalue()

    def test_synthetic_match_is_scoped_to_supplied_vectors(self):
        rc, output = self.compare([1, 2, 3])
        self.assertEqual(rc, 0)
        self.assertIn("VERIFIED", output)
        self.assertNotIn("93-layer", output)
        self.assertNotIn("real checkpoint", output)
        self.assertIn("supplied", output)

    def test_different_vectors_are_not_verified(self):
        for reference in [[3, 2, 1], [1, 2], [float("nan"), 2, 3]]:
            with self.subTest(reference=reference):
                rc, output = self.compare([1, 2, 3], reference=reference)
                self.assertEqual(rc, 1)
                self.assertNotIn("VERIFIED", output)

    def test_incomplete_float_is_not_ignored(self):
        rc, output = self.compare([1, 2, 3], extra=b"x")
        self.assertEqual(rc, 1)
        self.assertIn("float32", output)

    def test_empty_nonfinite_and_multidimensional_are_refused(self):
        for values in [[], [float("nan"), 1], [float("inf"), 1], [[1, 2], [3, 4]]]:
            with self.subTest(values=values):
                rc, output = self.compare(values, json_input=True)
                self.assertEqual(rc, 1)
                self.assertNotIn("VERIFIED", output)


if __name__ == "__main__":
    unittest.main()
