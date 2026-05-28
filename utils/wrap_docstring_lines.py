#!/usr/bin/env python3
"""Wrap long lines in Python docstrings.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import ast
import re
import sys
import textwrap
from pathlib import Path

from fix_docstring_sections import normalize_docstring, render_docstring

ROOT = Path(__file__).resolve().parents[1]
MAX_LINE_LENGTH = 100
SKIP_DIRS = {".git", ".venv", "__pycache__"}
SECTION_RE = re.compile(r"^[A-Z][A-Za-z ]+:$")
ARG_ENTRY_RE = re.compile(r"^(\s+\*?\*?\w[\w*]*(?:\s*\([^)]*\))?:\s+)(.+)$")


def _wrap_line(
    line: str, width: int, subsequent_indent: str | None = None
) -> list[str]:
    """Wrap one docstring content line."""
    if len(line) <= width:
        return [line]
    if "://" in line or line.lstrip().startswith((">>>", "...", "|")):
        return [line]
    if subsequent_indent is None:
        indent_match = re.match(r"^\s*", line)
        subsequent_indent = indent_match.group(0) if indent_match else ""
    return textwrap.wrap(
        line,
        width=width,
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def wrap_docstring(doc: str, source_indent: str) -> str:
    """Wrap long lines in a normalized docstring body."""
    width = MAX_LINE_LENGTH - len(source_indent)
    lines = normalize_docstring(doc).splitlines()
    wrapped: list[str] = []
    current_section: str | None = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped in {
            "Args:",
            "Returns:",
            "Raises:",
            "Yields:",
            "Attributes:",
            "Examples:",
            "References:",
            "Note:",
            "Notes:",
        }:
            current_section = stripped[:-1]
            wrapped.append(line)
            continue
        if SECTION_RE.match(stripped) and not line.startswith(" "):
            current_section = None
            wrapped.append(line)
            continue
        if not stripped:
            wrapped.append("")
            continue

        entry_match = ARG_ENTRY_RE.match(line)
        if current_section in {"Args", "Attributes", "Raises"} and entry_match:
            wrapped.extend(_wrap_line(line, width, subsequent_indent=" " * 8))
        elif line.startswith("        "):
            wrapped.extend(_wrap_line(line, width, subsequent_indent=" " * 8))
        elif line.startswith("    "):
            wrapped.extend(_wrap_line(line, width, subsequent_indent=" " * 4))
        elif index == 0:
            # Keep summaries as one logical line. Splitting the first line turns the
            # continuation into a description for pydocstyle (D205).
            wrapped.append(line)
        else:
            wrapped.extend(_wrap_line(line, width, subsequent_indent=""))

    return "\n".join(wrapped)


def _docstring_expr(node: ast.AST, module: ast.Module) -> ast.Expr | None:
    if isinstance(node, ast.Module):
        if not module.body or not isinstance(module.body[0], ast.Expr):
            return None
        expr = module.body[0]
    else:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return None
        if not node.body or not isinstance(node.body[0], ast.Expr):
            return None
        expr = node.body[0]

    if isinstance(expr.value, ast.Constant) and isinstance(expr.value.value, str):
        return expr
    return None


def process_file(path: Path) -> bool:
    """Wrap docstrings in one Python file."""
    source = path.read_text(encoding="utf-8")
    try:
        module = ast.parse(source)
    except SyntaxError:
        return False

    DocNode = ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    nodes: list[DocNode] = [module]
    nodes.extend(
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )

    lines = source.splitlines(keepends=True)
    changed = False
    for node in sorted(nodes, key=lambda n: getattr(n, "lineno", 0), reverse=True):
        doc = ast.get_docstring(node, clean=False)
        if not doc:
            continue
        expr = _docstring_expr(node, module)
        if expr is None or expr.end_lineno is None:
            continue
        start = expr.lineno - 1
        end = expr.end_lineno
        source_slice = "".join(lines[start:end])
        quote = "'''" if "'''" in source_slice and '"""' not in source_slice else '"""'
        indent = " " * expr.col_offset

        wrapped_doc = wrap_docstring(doc, indent)
        if wrapped_doc == normalize_docstring(doc):
            continue
        lines[start:end] = render_docstring(wrapped_doc, indent, quote)
        changed = True

    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return changed


def iter_targets(targets: list[str]) -> list[Path]:
    """Return Python files below the given targets."""
    paths: list[Path] = []
    for target in targets:
        path = ROOT / target
        if path.is_file() and path.suffix == ".py":
            paths.append(path)
        elif path.is_dir():
            paths.extend(
                child
                for child in path.rglob("*.py")
                if not any(part in SKIP_DIRS for part in child.parts)
            )
    return sorted(set(paths))


def main(targets: list[str]) -> int:
    """Wrap long docstring lines under the given paths."""
    changed = 0
    for path in iter_targets(targets):
        if process_file(path):
            changed += 1
    print(f"wrapped_files={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["src/pixelator", "utils"]))
