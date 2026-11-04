"""A failed generation/copy must leave the prior packed artifact untouched."""
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class PackedOutput(unittest.TestCase):
    def write_shard(self, src, value=1.0, truncated=False):
        header = {"language_model.model.layers.0.self_attn.q_proj.weight":
                  {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
        raw = json.dumps(header).encode()
        payload = struct.pack("<f", value)
        (src / "model.safetensors").write_bytes(
            struct.pack("<Q", len(raw)) + raw + (payload[:2] if truncated else payload)
        )

    def run_tool(self, src, out, layers):
        return subprocess.run([sys.executable, str(ROOT / "tools/pack_trunk.py"),
                               str(src), str(out), str(layers)], capture_output=True, text=True, check=False)

    def test_copy_and_layer_failures_preserve_prior_valid_pair(self):
        for mode in ("missing-layer", "truncated", "zero-layers", "negative-layers"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                src, out = root / "src", root / "out"
                src.mkdir()
                self.write_shard(src)
                initial = self.run_tool(src, out, 1)
                self.assertEqual(initial.returncode, 0, initial.stdout + initial.stderr)
                old = {name: (out / name).read_bytes() for name in ("trunk.bin", "trunk.json")}
                self.write_shard(src, 2.0, truncated=mode == "truncated")
                layers = 2 if mode == "missing-layer" else 0 if mode == "zero-layers" else -1 if mode == "negative-layers" else 1
                result = self.run_tool(src, out, layers)
                self.assertEqual(result.returncode, 2 if layers <= 0 else 1,
                                 result.stdout + result.stderr)
                for name, raw in old.items():
                    self.assertEqual((out / name).read_bytes(), raw)
                self.assertEqual(sorted(path.name for path in out.iterdir()), ["trunk.bin", "trunk.json"])

    def test_successful_staged_replacement_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, out = root / "src", root / "out"
            src.mkdir()
            self.write_shard(src, 2.0)
            result = self.run_tool(src, out, 1)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((out / "trunk.json").read_text())
            self.assertEqual(manifest["layers"][0]["nbytes"], 4096)
            self.assertEqual((out / "trunk.bin").stat().st_size, 4096)
            self.assertEqual((out / "trunk.bin").read_bytes()[:4], struct.pack("<f", 2.0))


if __name__ == "__main__":
    unittest.main()
