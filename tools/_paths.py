"""_paths.py - one definition of where things live, shared by every tool.

WHY THIS EXISTS
    Five generators write fixtures and four readers consume them. When each computes its
    own path from __file__, one of them eventually disagrees with the rest, and the
    failure is silent in the worst direction: os.makedirs(..., exist_ok=True) happily
    creates the wrong directory, the generator reports success, and the fixtures the test
    suite actually reads go stale without anything saying so.

    Every path in the project is therefore derived here, once.

LAYOUT

    <repo>/
      tools/            this file
      tests/fixtures/   everything the C test binaries read
        ops/            per-kernel fixtures      (emit_fixtures.py)
        cache/          expert-cache fixtures    (make_cache_fixture.py)
        st/             safetensors fixtures     (make_st_fixture.py)
        ref_k3.json     full-model oracle        (make_k3_oracle.py)
        tiny_k3.bin     the oracle's weights     (make_k3_oracle.py)

MODEL FILES
    hf_dir() locates the released tokenizer and config files. There is no single correct
    answer, a checkout may sit beside them, or they may live under a download directory
    on a remote machine, so it searches a list of candidates and lets K3_HF_DIR override
    the search entirely.
"""
from __future__ import annotations

import os

# tools/ -> repo root. Everything below is relative to this and nothing else.
TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
FIX_OPS = os.path.join(FIXTURES, "ops")
FIX_CACHE = os.path.join(FIXTURES, "cache")
FIX_ST = os.path.join(FIXTURES, "st")
FIX_TOK = os.path.join(FIXTURES, "tok")
FIX_GOLDEN = os.path.join(FIXTURES, "golden")

REF_K3_JSON = os.path.join(FIXTURES, "ref_k3.json")
TINY_K3_BIN = os.path.join(FIXTURES, "tiny_k3.bin")


def ensure(path: str) -> str:
    """Create a fixture directory and return it.

    Only ever call this on a path from this module. Calling makedirs on a hand-built
    path is what lets a generator silently populate a tree nothing reads.
    """
    os.makedirs(path, exist_ok=True)
    return path


def hf_dir() -> str | None:
    """Locate the released Kimi K3 files (tiktoken.model, config.json, ...).

    Returns the first directory that contains tiktoken.model, or None. Set K3_HF_DIR to
    skip the search. Returning None rather than raising is deliberate: several callers
    can do useful work without the model files and should say so specifically.
    """
    override = os.environ.get("K3_HF_DIR")
    if override:
        return override if os.path.isfile(os.path.join(override, "tiktoken.model")) else None

    home = os.path.expanduser("~")
    candidates = [
        os.path.join(ROOT, "kimi_k3_hf", "files"),
        os.path.join(os.path.dirname(ROOT), "kimi_k3_hf", "files"),
        os.path.join(home, "k3model"),
        os.path.join(home, "k3", "hf"),
        os.path.join(home, "k3c", "hf_files"),
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "tiktoken.model")):
            return c
    return None
