"""Test benchmark receipt validation and stale-output rejection without a model."""
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("read_result", ROOT / "benchmarks/read_result.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


def receipt():
    return {"prompt_ids": [1008, 10484, 318, 15383, 387], "generated_ids": list(range(8)),
            "interrupted": False, "expert_drops": 0, "layers_requested": 2, "layers_completed": 2,
            "seconds_per_token": 1.5, "peak_rss_bytes": 2_000_000_000,
            "expert_bytes_read": 1_000_000_000, "trunk_bytes_read": 2_000_000_000,
            "embedding_bytes_read": 3_000_000_000, "lm_head_bytes_read": 4_000_000_000}


class ReceiptFields(unittest.TestCase):
    def test_exact_metrics_include_all_io_sources(self):
        value = receipt()
        self.assertEqual(reader.fields(value, value["prompt_ids"], 8),
                         "1.500000\t2.000000\t10.000000\t0,1,2,3,4,5,6,7")

    def test_invalid_results_cannot_enter_table(self):
        changes = [{"seconds_per_token": float("nan")}, {"peak_rss_bytes": 0},
                   {"generated_ids": [1]}, {"generated_ids": [True] * 8},
                   {"prompt_ids": [1]}, {"expert_drops": 1}, {"interrupted": True},
                   {"layers_completed": 1}, {"expert_bytes_read": -1}]
        for change in changes:
            value = copy.deepcopy(receipt())
            value.update(change)
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                reader.fields(value, receipt()["prompt_ids"], 8)
        value = receipt()
        value["prompt_ids"] = [True]
        with self.assertRaises(ValueError):
            reader.fields(value, [1], 8)
        value = receipt()
        del value["seconds_per_token"]
        with self.assertRaises(KeyError):
            reader.fields(value, value["prompt_ids"], 8)


@unittest.skipUnless(os.name != "nt" and shutil.which("bash"), "POSIX bash required")
class FreshResults(unittest.TestCase):
    def test_fresh_receipt_required_and_valid_new_output_accepted(self):
        for script in ("memory-ladder.sh", "split-sweep.sh"):
            for produces in (False, True, "oom"):
                with self.subTest(script=script, produces=produces), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    for folder in ("commands", "bin", "model", "trunk", "out"):
                        (root / folder).mkdir()
                    new = receipt()
                    new["seconds_per_token"] = 2.5
                    engine = (f"#!{sys.executable}\nimport json,sys\n"
                              "if '--version' in sys.argv: raise SystemExit(0)\n"
                              f"value = {new!r}\n"
                              "with open(sys.argv[sys.argv.index('--out')+1], 'w') as f: json.dump(value,f)\n")
                    commands = {
                        "commands/systemd-run": "#!/bin/sh\nwhile [ \"$#\" -gt 0 ]; do\n"
                                                "if [ \"$1\" = ./bin/k3 ]; then exec \"$@\"; fi\n"
                                                "shift\ndone\nexit 0\n",
                        "commands/column": "#!/bin/sh\nexit 0\n",
                        "bin/k3": ("#!/bin/sh\nexit 137\n" if produces == "oom" else
                                   engine if produces else "#!/bin/sh\nexit 0\n"),
                    }
                    for name, text in commands.items():
                        path = root / name
                        path.write_text(text)
                        path.chmod(0o755)
                    tags = ([f"{gb}gb_r1" for gb in (8, 16, 32, 64, 96, 128)]
                            if script == "memory-ladder.sh" else
                            [f"tr{tr}_r1" for tr in (12.8, 32.0, 76.8, 110.1)])
                    old = json.dumps(receipt())
                    for tag in tags:
                        (root / "out" / (tag + ".json")).write_text(old)
                    args = ["bash", str(ROOT / "benchmarks" / script), str(root / "model"),
                            str(root / "trunk"), str(root / "out")]
                    if script == "split-sweep.sh":
                        args.append("128")
                    args.append("1")
                    env = dict(os.environ, PATH=str(root / "commands") + os.pathsep + os.environ["PATH"])
                    result = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, check=False)
                    self.assertEqual(result.returncode, 0 if produces is True else 1,
                                     result.stdout + result.stderr)
                    if produces is True:
                        self.assertIn("Output identical", result.stdout)
                        for tag in tags:
                            value = json.loads((root / "out" / (tag + ".json")).read_text())
                            self.assertEqual(value["seconds_per_token"], 2.5)
                    else:
                        if produces is False:
                            self.assertIn("fresh result", result.stdout)
                        self.assertNotIn("Output identical", result.stdout)
                        for tag in tags:
                            self.assertEqual((root / "out" / (tag + ".json")).read_text(), old)


if __name__ == "__main__":
    unittest.main()
