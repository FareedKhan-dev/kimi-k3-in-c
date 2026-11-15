"""Quantization output must never truncate an aliased source container."""
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required for quantization")
class SourceOwnership(unittest.TestCase):
    def fixture(self, src):
        src.mkdir()
        manifest = {"n_layers": 1, "align": 4096, "layers": [{"layer": 0,
                    "file_off": 0, "nbytes": 4096, "tensors": {"matrix": {
                    "off": 0, "nbytes": 4, "dtype": "BF16", "shape": [1, 2]}}}]}
        (src / "trunk.json").write_text(json.dumps(manifest))
        (src / "trunk.bin").write_bytes(struct.pack("<HH", 0x3f80, 0x4000) + b"\0" * 4092)
        return {name: (src / name).read_bytes() for name in ("trunk.json", "trunk.bin")}

    def run_tool(self, src, dst):
        return subprocess.run([sys.executable, str(ROOT / "tools/int8_trunk.py"), str(src), str(dst)],
                              capture_output=True, text=True, check=False)

    def test_aliases_preserve_input(self):
        kinds = ["same-directory", "dot-directory", "linked-binary", "linked-manifest",
                 "cross-binary", "cross-manifest"]
        if os.name != "nt":
            kinds += ["directory-symlink", "cross-symlink-binary", "cross-symlink-manifest"]
        for kind in kinds:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                src = root / "src"
                original = self.fixture(src)
                dst = src if kind == "same-directory" else root / "out"
                if kind == "dot-directory":
                    (src / "child").mkdir()
                    dst = src / "child" / ".."
                if kind == "directory-symlink":
                    dst.symlink_to(src, target_is_directory=True)
                if kind.startswith(("linked", "cross")):
                    dst.mkdir()
                    name = "trunk.bin" if kind.endswith("binary") else "trunk.json"
                    target = name
                    if kind.startswith("cross"):
                        target = "trunk.json" if name == "trunk.bin" else "trunk.bin"
                    if "symlink" in kind:
                        (dst / target).symlink_to(src / name)
                    else:
                        os.link(src / name, dst / target)
                result = self.run_tool(src, dst)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("source", result.stderr)
                for name, raw in original.items():
                    self.assertEqual((src / name).read_bytes(), raw)

    def test_separate_output_converts_and_keeps_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, dst = root / "src", root / "out"
            original = self.fixture(src)
            result = self.run_tool(src, dst)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = json.loads((dst / "trunk.json").read_text())
            self.assertEqual(output["layers"][0]["tensors"]["matrix"]["dtype"], "I8R")
            for name, raw in original.items():
                self.assertEqual((src / name).read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
