"""Build-mode regression with a recording compiler, no real compilation."""
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("make") and os.name != "nt",
                     "POSIX make and executable recording compiler are required")
class BuildModes(unittest.TestCase):
    def test_mode_changes_rebuild_all_engine_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            workspace = tmp / "source"
            workspace.mkdir()
            shutil.copy2(ROOT / "Makefile", workspace / "Makefile")
            for folder in ("src", "include", "third_party", "tests/unit"):
                shutil.copytree(ROOT / folder, workspace / folder)
            log = tmp / "compile.jsonl"
            driver = tmp / "cc.py"
            driver.write_text(
                f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n"
                f"with open({str(log)!r}, 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
                "Path(sys.argv[sys.argv.index('-o')+1]).touch()\n"
            )
            driver.chmod(0o755)
            compiler = shlex.quote(str(driver))
            common = ["make", f"CC={compiler}", f"SANITIZER_CC={compiler}",
                      f"BUILD={tmp / 'build'}", f"BIN={tmp / 'bin'}",
                      "OMP_CFLAGS=", "OMP_LDFLAGS=", "SANITIZER_DLLS="]
            # A parent make exports command-line variables such as CI's portable
            # ARCH. Start the recording experiment from its own default mode while
            # preserving PATH and the host's tool/interpreter locations.
            build_inputs = {
                "MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEOVERRIDES", "GNUMAKEFLAGS", "MAKEFILES", "OPT",
                "CC", "CFLAGS", "LDFLAGS", "ARCH", "INCLUDES", "WARN", "BUILD", "BIN",
                "OMP_PREFIX", "OMP_CFLAGS", "OMP_LDFLAGS", "SANITIZER_CC",
                "SANITIZER_LDFLAGS", "SANITIZER_DLLS", "UNAME_S", "UNAME_M",
            }
            env = {k: v for k, v in os.environ.items() if k not in build_inputs}

            def build(*targets, extra=()):
                log.write_text("")
                result = subprocess.run(common + list(extra) + list(targets), cwd=workspace, env=env,
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return [json.loads(line) for line in log.read_text().splitlines()]

            initial = build()
            self.assertEqual(sum("-c" in cmd for cmd in initial), 6)
            self.assertEqual(len(initial), 7)  # six objects and the CLI
            portable = build("portable")
            self.assertEqual(sum("-c" in cmd for cmd in portable), 6)
            self.assertFalse(any("-mcpu=native" in cmd or "-march=native" in cmd
                                 for cmd in portable))
            self.assertEqual(build("portable"), [])
            sanitized = build("asan")
            self.assertEqual(sum("-c" in cmd for cmd in sanitized), 6)
            self.assertTrue(all("-fsanitize=address,undefined" in cmd
                                for cmd in sanitized))
            restored = build("all")
            self.assertEqual(sum("-c" in cmd for cmd in restored), 6)
            self.assertFalse(any("-fsanitize=address,undefined" in cmd
                                 for cmd in restored))
            self.assertEqual(len(restored), 7)

            direct = [str(tmp / "bin" / "test_sampler"), str(tmp / "bin" / "test_tok")]
            self.assertEqual(len(build(*direct)), 2)
            changed = build(*direct, extra=("CFLAGS=-O0 -DK3_TEST_BUILD_MODE=1",))
            self.assertEqual(len(changed), 2)
            self.assertTrue(any("-DK3_TEST_BUILD_MODE=1" in cmd for cmd in changed))
            self.assertFalse(any(arg.endswith(".configuration") or arg == "FORCE"
                                 for cmd in changed for arg in cmd))
            build("all", *direct)  # restore the default configuration
            self.assertEqual(build("all", *direct), [])

            # macOS make3.81 compares second-resolution timestamps. Touch a
            # copied header in a later second, never the caller's source tree.
            time.sleep(1.1)
            (workspace / "include/k3/k3.h").touch()
            header_changed = build("all", *direct)
            self.assertEqual(sum("-c" in cmd for cmd in header_changed), 6)
            self.assertEqual(len(header_changed), 9)
            self.assertFalse(any(arg.endswith(".h") for cmd in header_changed for arg in cmd))


if __name__ == "__main__":
    unittest.main()
