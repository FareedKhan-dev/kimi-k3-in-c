"""Doctor's storage probe must never overwrite a caller's existing file."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.name != "nt" and shutil.which("bash"), "POSIX bash required")
class DoctorProbe(unittest.TestCase):
    def test_probe_preserves_existing_file_and_cleans_its_own(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                target, commands = root / "model", root / "commands"
                target.mkdir()
                commands.mkdir()
                sentinel = target / ".k3_doctor_probe"
                sentinel.write_text("caller data")
                scripts = {
                    "uname": "#!/bin/sh\necho Linux\n",
                    "sync": "#!/bin/sh\nexit 0\n",
                    "dd": "#!/bin/sh\nfor arg in \"$@\"; do\n"
                          "case \"$arg\" in of=*) output=${arg#of=};; esac\ndone\n"
                          "if [ \"$output\" != /dev/null ]; then printf probe > \"$output\"; fi\n"
                          + ("exit 1\n" if fail else
                             "echo '1 bytes copied, 1 s, 1 MB/s' >&2\nexit 0\n"),
                }
                for name, text in scripts.items():
                    path = commands / name
                    path.write_text(text)
                    path.chmod(0o755)
                env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"])
                result = subprocess.run(["bash", str(ROOT / "scripts/k3-doctor.sh"), str(target)],
                                        env=env, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(sentinel.read_text(), "caller data")
                self.assertEqual(list(target.iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()
