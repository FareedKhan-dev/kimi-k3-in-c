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
    def test_fresh_identity_and_complete_finite_stage_contract(self):
        modes = ("missing", "wrong-layer", "wrong-T", "invalid-kind", "no-stages",
                 "missing-stage", "nan-stage", "nan-abs", "negative-abs", "mismatch-stage",
                 "router-mismatch", "valid", "rounded")
        for layer in (0, 1):
            for mode in modes:
                with self.subTest(layer=layer, mode=mode), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    stale = root / "real_layer.json"
                    stale.write_text(json.dumps({"layer": layer, "T": 4, "is_mla": 1}))
                    old = stale.read_bytes()

                    def child(args, mode=mode, stale=stale, layer=layer, **kwargs):
                        if str(args[0]).endswith("test_real_layer"):
                            if mode != "missing":
                                value = {"layer": layer + 1 if mode == "wrong-layer" else layer,
                                         "T": 8 if mode == "wrong-T" else 4,
                                         "is_mla": 5 if mode == "invalid-kind" else 1}
                                target = Path(args[5]) if len(args) > 5 else stale
                                target.write_text(json.dumps(value))
                            return SimpleNamespace(returncode=0, stdout="", stderr="")
                        labels = (["x_norm", "attn_out", "dense_out"] if layer == 0 else
                                  ["x_norm", "attn_out", "route_wt (id-sorted)", "moe_out"])
                        lines = [f"  {name} max|diff| 0 rel 0 budget 1 OK" for name in labels]
                        if mode == "no-stages":
                            lines = []
                        elif mode == "missing-stage":
                            lines = lines[:1]
                        elif mode == "nan-stage":
                            lines[0] = "  x_norm max|diff| 0 rel nan budget 1 OK"
                        elif mode == "nan-abs":
                            lines[0] = "  x_norm max|diff| nan rel 0 budget 1 OK"
                        elif mode == "negative-abs":
                            lines[0] = "  x_norm max|diff| -1 rel 0 budget 1 OK"
                        elif mode == "mismatch-stage":
                            lines[0] = "  x_norm max|diff| 1 rel 1 budget 1 MISMATCH"
                        elif mode == "router-mismatch":
                            lines.append("  route_wt (id-sorted) max|diff| 1 rel 1 budget 1 MISMATCH")
                        elif mode == "rounded":
                            lines[0] = "  x_norm max|diff| 0.001 rel 1.23e-4 budget 1.2e-4 OK"
                        return SimpleNamespace(returncode=0, stdout="\n".join(lines), stderr="")

                    with patch.object(sys, "argv", ["conform_all.py", "unused", tmp, "python", str(layer), str(layer)]), \
                         patch.object(conform.subprocess, "run", side_effect=child) as run, \
                         redirect_stdout(io.StringIO()):
                        self.assertEqual(conform.main(), 0 if mode in ("valid", "rounded") else 1)
                    self.assertEqual(stale.read_bytes(), old)
                    result = json.loads((root / "conform_all.json").read_text(),
                                        parse_constant=lambda value: self.fail("nonstandard JSON: " + value))
                    row = result["rows"][0]
                    self.assertNotEqual(Path(row["dump_path"]), stale)
                    self.assertTrue(Path(row["dump_path"]).exists())
                    self.assertEqual(run.call_count, 1 if mode in modes[:4] else 2)
                    if mode == "rounded":
                        self.assertGreater(row["worst_ratio"], 1)


if __name__ == "__main__":
    unittest.main()
