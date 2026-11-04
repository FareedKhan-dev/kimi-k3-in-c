"""Conformance runner must execute a nonempty, valid released-layer interval."""
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "conform_all", Path(__file__).resolve().parents[2] / "tools" / "conform_all.py"
)
conform = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conform)


class ConformanceRange(unittest.TestCase):
    def test_invalid_range_cannot_certify_empty_work(self):
        for first, last in [(2, 1), (-1, 2), (0, 93), (93, 93), ("x", 4)]:
            with self.subTest(first=first, last=last), tempfile.TemporaryDirectory() as tmp:
                with patch.object(sys, "argv", ["conform_all.py", "unused", tmp,
                                               "unused-python", str(first), str(last)]), \
                     patch.object(conform.subprocess, "run") as run, \
                     redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(conform.main(), 2)
                    run.assert_not_called()
                self.assertFalse((Path(tmp) / "conform_all.json").exists())

    def test_one_layer_interval_still_runs_both_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "real_layer.json").write_text('{"is_mla":1}')
            responses = [SimpleNamespace(returncode=0, stdout="", stderr=""),
                         SimpleNamespace(returncode=0, stdout=
                                         "  attention max|diff| 0 rel 0 budget 1 OK", stderr="")]
            with patch.object(sys, "argv", ["conform_all.py", "unused", tmp,
                                           "unused-python", "92", "92"]), \
                 patch.object(conform.subprocess, "run", side_effect=responses) as run, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(conform.main(), 0)
                self.assertEqual(run.call_count, 2)
            result = json.loads((Path(tmp) / "conform_all.json").read_text())
            self.assertEqual(result["passed"], 1)
            self.assertEqual(result["rows"][0]["layer"], 92)


if __name__ == "__main__":
    unittest.main()
