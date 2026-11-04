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
    def test_invalid_repetitions_are_refused_before_preflight(self):
        for script in ("memory-ladder.sh", "split-sweep.sh"):
            for reps in ("0", "00", "-1", "junk", "1.5"):
                with self.subTest(script=script, reps=reps), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    commands = root / "commands"
                    commands.mkdir()
                    probe = commands / "systemd-run"
                    probe.write_text("#!/bin/sh\necho PREFLIGHT_RAN >&2\nexit 99\n")
                    probe.chmod(0o755)
                    args = ["bash", str(ROOT / "benchmarks" / script),
                            "unused-model", "unused-trunk", str(root / "out")]
                    if script == "split-sweep.sh":
                        args.append("128")
                    args.append(reps)
                    env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"])
                    result = subprocess.run(args, env=env, capture_output=True, text=True, check=False)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("repetitions", result.stdout + result.stderr)
                    self.assertNotIn("PREFLIGHT_RAN", result.stdout + result.stderr)
                    self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
