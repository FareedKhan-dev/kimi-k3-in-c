"""Numeric argument refusals must happen before loading any model files."""
import os
import subprocess
import unittest


class NumericArguments(unittest.TestCase):
    def test_invalid_numbers(self):
        cases = {
            "--gen": ["", "1junk", "4294967296", "-1"],
            "--spec": ["", "4x", "4294967296", "-1"],
            "--trunk-ring": ["", "2x", "4294967296", "-1"],
            "--stop-id": ["", "4294967296", "999999999999999999999"],
            "--seed": ["", "-1", "123junk", "18446744073709551616"],
        }
        for flag in ("--cache-gb", "--trunk-gb", "--draft-trunk-gb"):
            cases[flag] = ["", "1junk", "nan", "inf", "-1", "1e300"]
        for flag in ("--temperature", "--top-p"):
            cases[flag] = ["", "0.7junk", "inf", "nan", "1e999", "1e-999"]
        for flag, values in cases.items():
            for value in values:
                with self.subTest(flag=flag, value=value):
                    result = subprocess.run(
                        [os.environ.get("K3_TEST_BIN", "./bin/k3"), "nonexistent-model", "--ids", "1", flag, value],
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(flag, result.stderr)
                    self.assertNotIn("cannot open", result.stderr)

    def test_valid_syntax_reaches_later_validation(self):
        cases = {
            "--gen": ["0", "1"], "--spec": ["0", "4"],
            "--trunk-ring": ["0", "2"], "--stop-id": ["0", "1"],
            "--seed": ["0", "18446744073709551615"],
            "--cache-gb": ["0", "1.5", "1e-3"],
            "--trunk-gb": ["0", "1.5"], "--draft-trunk-gb": ["0", "1.5"],
            "--temperature": ["0.7"], "--top-p": ["0.95"],
        }
        for flag, values in cases.items():
            for value in values:
                with self.subTest(flag=flag, value=value):
                    result = subprocess.run(
                        [os.environ.get("K3_TEST_BIN", "./bin/k3"),
                         "nonexistent-model", "--ids", "1", flag, value],
                        capture_output=True, text=True, check=False,
                    )
                    self.assertNotIn("expected a", result.stderr)
                    self.assertNotIn("needs an unsigned", result.stderr)


if __name__ == "__main__":
    unittest.main()
