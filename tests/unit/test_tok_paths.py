"""Tokenizer discovery must honor repository paths and explicit selection."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "tools/tok.py"


class TokenizerPaths(unittest.TestCase):
    def load_at(self, root, env):
        namespace = {"__file__": str(root / "tools/tok.py"), "__name__": "tok_path_test"}
        with patch.dict(os.environ, env, clear=True), \
             patch.object(os.path, "expanduser", return_value=str(root / "absent-home")):
            exec(compile(SOURCE.read_text(), str(SOURCE), "exec"), namespace)
        return namespace

    def test_in_repository_location_is_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            files = root / "kimi_k3_hf/files"
            files.mkdir(parents=True)
            (files / "tiktoken.model").touch()
            module = self.load_at(root, {})
            self.assertEqual(Path(module["ROOT"]), root)
            self.assertEqual(Path(module["FILES"]), files)

    def test_explicit_missing_selection_does_not_use_other_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            fallback = Path(tmp) / "kimi_k3_hf/files"
            fallback.mkdir(parents=True)
            (fallback / "tiktoken.model").touch()
            selected = str(Path(tmp) / "missing-selected")
            with self.assertRaisesRegex(SystemExit, "K3_TOK_FILES"):
                self.load_at(root, {"K3_TOK_FILES": selected})

    def test_explicit_valid_selection_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            chosen = Path(tmp) / "selected"
            chosen.mkdir()
            (chosen / "tiktoken.model").touch()
            module = self.load_at(root, {"K3_TOK_FILES": str(chosen)})
            self.assertEqual(Path(module["FILES"]), chosen)


if __name__ == "__main__":
    unittest.main()
