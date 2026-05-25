#!/usr/bin/env python3
"""Read-only audit of docstring styles and coverage gaps."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPHINX_MARKERS = re.compile(r":(?:param|returns?|rtype|raises|ivar|vartype)\b")

SKIP_DIRS = {".venv", ".git", "__pycache__"}


def iter_py_files(base: Path) -> list[Path]:
    files: list[Path] = []
    for path in base.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def audit_file(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    sphinx_hits = len(SPHINX_MARKERS.findall(text))
    missing_public = 0
    module_doc = False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {"sphinx": sphinx_hits, "missing": 0, "module_doc": False}

    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        module_doc = True

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
