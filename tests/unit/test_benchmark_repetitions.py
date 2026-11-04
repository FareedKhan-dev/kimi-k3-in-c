"""Benchmark determinism cannot pass without any requested repetitions."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.name != "nt" and shutil.which("bash"), "POSIX bash required")
class Repetitions(unittest.TestCase):
    def test_repetition_contract_before_model_execution(self):
        for script in ("memory-ladder.sh", "split-sweep.sh"):
            for reps in ("0", "00", "-1", "junk", "1.5", "1", "03"):
                with self.subTest(script=script, reps=reps), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    commands = root / "commands"
                    commands.mkdir()
                    valid = reps in ("1", "03")
                    probe = commands / "systemd-run"
                    probe.write_text("#!/bin/sh\necho PREFLIGHT_RAN\n"
                                     + ("exit 99\n" if valid else "exit 0\n"))
                    probe.chmod(0o755)
                    (root / "bin").mkdir()
                    engine = root / "bin/k3"
                    engine.write_text("#!/bin/sh\nexit 0\n")
                    engine.chmod(0o755)
                    column = commands / "column"
                    column.write_text("#!/bin/sh\nexit 0\n")
                    column.chmod(0o755)
                    (root / "model").mkdir()
                    (root / "trunk").mkdir()
                    args = ["bash", str(ROOT / "benchmarks" / script),
                            str(root / "model"), str(root / "trunk"), str(root / "out")]
                    if script == "split-sweep.sh":
                        args.append("128")
                    args.append(reps)
                    env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"])
                    result = subprocess.run(args, env=env, cwd=root, capture_output=True, text=True, check=False)
                    self.assertEqual(result.returncode, 1 if valid else 2,
                                     result.stdout + result.stderr)
                    if valid:
                        self.assertIn("does not work", result.stdout)
                        self.assertIn("PREFLIGHT_RAN", result.stdout)
                    else:
                        self.assertIn("repetitions", result.stdout + result.stderr)
                        self.assertNotIn("PREFLIGHT_RAN", result.stdout + result.stderr)
                    self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
