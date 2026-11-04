"""Resolve harness input paths before changing the subprocess working directory."""
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "conform_all", Path(__file__).resolve().parents[2] / "tools" / "conform_all.py"
)
conform = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conform)


class ConformancePaths(unittest.TestCase):
    def test_relative_paths_keep_callers_meaning(self):
        for python in ("python3", ".venv/bin/python"):
            with self.subTest(python=python), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                (root / "bin").mkdir()
                (root / "bin/real_layer.json").write_text('{"is_mla":1}')
                def child(args, **kwargs):
                    if str(args[0]).endswith("test_real_layer"):
                        target = Path(args[5]) if len(args) > 5 else Path(args[0]).parent / "real_layer.json"
                        target.write_text(json.dumps({"layer": int(args[2]), "T": int(args[3]), "is_mla": 1},
                                                     separators=(",", ":")))
                        return SimpleNamespace(returncode=0, stdout="", stderr="")
                    stages = ("x_norm", "attn_out", "dense_out")
                    output = "\n".join(f"  {name} max|diff| 0 rel 0 budget 1 OK" for name in stages)
                    return SimpleNamespace(returncode=0, stdout=output, stderr="")
                old_cwd = os.getcwd()
                try:
                    os.chdir(root)
                    with patch.object(sys, "argv", ["conform_all.py", "weights", "bin",
                                                   python, "0", "0"]), \
                         patch.object(conform.subprocess, "run", side_effect=child) as run, \
                         redirect_stdout(io.StringIO()):
                        self.assertEqual(conform.main(), 0)
                    first, second = run.call_args_list
                    self.assertEqual(Path(first.args[0][0]), root / "bin/test_real_layer")
                    self.assertEqual(Path(first.args[0][1]), root / "weights")
                    self.assertEqual(Path(first.kwargs["cwd"]), root / "bin")
                    expected_python = str(root / python) if "/" in python else python
                    self.assertEqual(second.args[0][0], expected_python)
                    self.assertEqual(Path(second.args[0][2]), root / "weights")
                finally:
                    os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
