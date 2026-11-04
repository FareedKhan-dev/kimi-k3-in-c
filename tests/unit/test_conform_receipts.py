"""Conformance needs a new matching dump and actual finite stage evidence."""
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "conform_all", Path(__file__).resolve().parents[2] / "tools/conform_all.py"
)
conform = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conform)


class ConformanceReceipts(unittest.TestCase):
    def test_old_or_mismatched_dump_cannot_certify_current_run(self):
        modes = ("missing", "wrong-layer", "wrong-T", "invalid-kind", "no-stages",
                 "nan-stage", "mismatch-stage", "valid")
        for mode in modes:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                stale = root / "real_layer.json"
                stale.write_text('{"layer":0,"T":4,"is_mla":1}')
                old = stale.read_bytes()

                def child(args, mode=mode, stale=stale, **kwargs):
                    if str(args[0]).endswith("test_real_layer"):
                        if mode != "missing":
                            value = {"layer": 1 if mode == "wrong-layer" else 0,
                                     "T": 8 if mode == "wrong-T" else 4,
                                     "is_mla": 5 if mode == "invalid-kind" else 1}
                            # The producer's optional final argument is the new path.
                            target = Path(args[5]) if len(args) > 5 else stale
                            target.write_text(json.dumps(value))
                        return SimpleNamespace(returncode=0, stdout="", stderr="")
                    output = ("nothing evaluated" if mode == "no-stages" else
                              "  x_norm max|diff| nan rel nan budget 1 OK" if mode == "nan-stage" else
                              "  x_norm max|diff| 1 rel 1 budget 1 MISMATCH" if mode == "mismatch-stage" else
                              "  x_norm max|diff| 0 rel 0 budget 1 OK")
                    return SimpleNamespace(returncode=0, stdout=output, stderr="")

                with patch.object(sys, "argv", ["conform_all.py", "unused", tmp, "python", "0", "0"]), \
                     patch.object(conform.subprocess, "run", side_effect=child) as run, \
                     redirect_stdout(io.StringIO()):
                    self.assertEqual(conform.main(), 0 if mode == "valid" else 1)
                self.assertEqual(stale.read_bytes(), old)
                result = json.loads((root / "conform_all.json").read_text(),
                                    parse_constant=lambda value: self.fail("nonstandard JSON: " + value))
                row = result["rows"][0]
                self.assertNotEqual(Path(row["dump_path"]), stale)
                self.assertTrue(Path(row["dump_path"]).exists())
                self.assertEqual(run.call_count, 1 if mode in modes[:4] else 2)


if __name__ == "__main__":
    unittest.main()
