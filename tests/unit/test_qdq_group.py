"""A requested quantization grouping must not silently become per-row."""
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None
spec = importlib.util.spec_from_file_location(
    "qdq_trunk", Path(__file__).resolve().parents[2] / "tools" / "qdq_trunk.py"
)
qdq = importlib.util.module_from_spec(spec)
if np is not None:
    spec.loader.exec_module(qdq)


@unittest.skipIf(np is None, "NumPy is required for quantization tools")
class QuantizationGroup(unittest.TestCase):
    def test_invalid_group_is_not_a_row_fallback(self):
        for group in (-1, 2, 4):
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, "group"):
                qdq.qdq(np.ones((2, 3), dtype=np.float32), 8, group)

    def test_valid_group_and_row_modes_are_distinct(self):
        values = np.array([[1, 2, 100, 200]], dtype=np.float32)
        row = qdq.qdq(values, 8, 0)
        grouped = qdq.qdq(values, 8, 2)
        self.assertFalse(np.array_equal(row, grouped))
        self.assertEqual(grouped[0, 1], 2)
        self.assertEqual(grouped[0, 3], 200)

    def test_cli_rejects_before_copying_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "src", Path(tmp) / "dst"
            src.mkdir()
            manifest = {"layers": [{"tensors": {
                "matrix": {"dtype": "BF16", "shape": [2, 3], "nbytes": 12, "off": 0}
            }, "file_off": 0}]}
            (src / "trunk.json").write_text(json.dumps(manifest))
            (src / "trunk.bin").write_bytes(b"\0" * 12)
            with patch.object(sys, "argv", ["qdq_trunk.py", str(src), str(dst), "--group", "2"]), \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
                qdq.main()
            self.assertEqual(result.exception.code, 2)
            self.assertFalse(dst.exists())


if __name__ == "__main__":
    unittest.main()
