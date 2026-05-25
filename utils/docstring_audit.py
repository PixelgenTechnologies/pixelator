#!/usr/bin/env python3
"""Read-only audit of docstring styles and coverage gaps.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPHINX_FIELD_RE = re.compile(
    r":(?:param(?:s)?|returns?|rtype|raises|ivar|vartype|attr|var|yield|meta|cvar|type)\b",
    re.IGNORECASE,
)

SKIP_DIRS = {".venv", ".git", "__pycache__"}


def iter_py_files(base: Path) -> list[Path]:
    """Yield Python file paths under base, skipping common cache directories."""
    files: list[Path] = []
    for path in base.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def iter_docstrings(tree: ast.AST) -> list[str]:
    """Collect docstrings attached to module, class, and function nodes."""
    docs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docs.append(doc)
    return docs


def audit_file(path: Path) -> dict[str, int]:
    """Return sphinx-marker and missing-docstring counts for one file."""
    text = path.read_text(encoding="utf-8")
    missing_public = 0
    module_doc = False
    sphinx_hits = 0
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {"sphinx": 0, "missing": 0, "module_doc": False}

    docs = iter_docstrings(tree)
    sphinx_hits = sum(len(SPHINX_FIELD_RE.findall(doc)) for doc in docs)

    module_doc = ast.get_docstring(tree) is not None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") and node.name != "__init__":
                continue
            if ast.get_docstring(node) is None:
                missing_public += 1
        elif isinstance(node, ast.ClassDef):
            if ast.get_docstring(node) is None:
                missing_public += 1

    return {"sphinx": sphinx_hits, "missing": missing_public, "module_doc": module_doc}


def main(paths: list[str]) -> int:
    """Print docstring audit statistics for the given paths."""
    totals = {"files": 0, "sphinx": 0, "missing": 0, "no_module_doc": 0}
    for arg in paths:
        base = ROOT / arg
        for path in iter_py_files(base):
            stats = audit_file(path)
            totals["files"] += 1
            totals["sphinx"] += stats["sphinx"]
            totals["missing"] += stats["missing"]
            if not stats["module_doc"]:
                totals["no_module_doc"] += 1
    print(f"paths={paths}")
    print(f"files={totals['files']}")
    print(f"sphinx_markers={totals['sphinx']}")
    print(f"missing_public_docstrings={totals['missing']}")
    print(f"files_without_module_docstring={totals['no_module_doc']}")
    return 0


if __name__ == "__main__":
    targets = sys.argv[1:] or ["src/pixelator", "tests", "utils"]
    raise SystemExit(main(targets))
