#!/usr/bin/env python3
"""Normalize Google docstring section indentation inside docstrings.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = (
    "Args",
    "Returns",
    "Raises",
    "Yields",
    "Attributes",
    "Examples",
    "References",
    "Note",
    "Notes",
)


def normalize_docstring(doc: str) -> str:
    """Normalize Google-style section indentation in a docstring body.

    Args:
        doc: Raw docstring body from an AST node.

    Returns:
        Docstring body with section entries indented relative to section headers.
    """
    lines = [line.rstrip() for line in inspect.cleandoc(doc).splitlines()]
    if not lines:
        return doc

    normalized: list[str] = []
    current: str | None = None
    previous_was_entry = False

    for line in lines:
        stripped = line.strip()
        if stripped in {f"{name}:" for name in SECTIONS}:
            normalized.append(stripped)
            current = stripped[:-1]
            previous_was_entry = False
            continue

        if current is None:
            # Previous bulk conversion preserved some code indentation inside the
            # docstring body. Narrative text should be flush with the docstring
            # content, not indented like source code.
            normalized.append(stripped if stripped else "")
            continue

        if not stripped:
            normalized.append("")
            previous_was_entry = False
            continue

        if current in {"Args", "Attributes", "Raises"}:
            match = re.match(
                r"^(\*?\*?\w[\w*]*(?:\s*\([^)]*\))?):\s*(.*)$",
                stripped,
            )
            if match:
                name, desc = match.groups()
                normalized.append(f"    {name}: {desc.strip()}")
                previous_was_entry = True
            else:
                indent = "        " if previous_was_entry else "    "
                normalized.append(f"{indent}{stripped}")
        else:
            normalized.append(f"    {stripped}")
            previous_was_entry = False

    while normalized and not normalized[-1].strip():
        normalized.pop()

    return "\n".join(normalized)


def render_docstring(doc: str, indent: str, quote: str = '"""') -> list[str]:
    """Render a normalized docstring body as source lines.

    Args:
        doc: Normalized docstring body without delimiters.
        indent: Source indentation for the docstring delimiters.
        quote: Triple-quote delimiter to use.

    Returns:
        Source lines for the complete docstring.
    """
    doc_lines = doc.splitlines()
    if len(doc_lines) == 1 and len(f"{indent}{quote}{doc_lines[0]}{quote}") <= 88:
        return [f"{indent}{quote}{doc_lines[0]}{quote}\n"]

    rendered = [f"{indent}{quote}{doc_lines[0]}\n"]
    rendered.extend(
        f"{indent}{line}\n" if line else f"{indent}\n" for line in doc_lines[1:]
    )
    rendered.append(f"{indent}{quote}\n")
    return rendered


def process_file(path: Path) -> bool:
    """Process file."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    lines = source.splitlines(keepends=True)
    changed = False
    nodes: list[ast.AST] = []
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
    ):
        nodes.append(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nodes.append(node)

    for node in sorted(nodes, key=lambda n: getattr(n, "lineno", 0), reverse=True):
        doc = ast.get_docstring(node, clean=False)
        if not doc or not any(f"{section}:" in doc for section in SECTIONS):
            continue
        new_doc = normalize_docstring(doc)
        if new_doc == doc:
            continue
        if isinstance(node, ast.Module):
            expr = tree.body[0]
        else:
            if not node.body or not isinstance(node.body[0], ast.Expr):
                continue
            expr = node.body[0]
        start = expr.lineno - 1
        end_line = expr.end_lineno
        indent_width = expr.col_offset or 0
        indent = " " * indent_width
        source = "".join(lines[start:end_line])
        quote = "'''" if "'''" in source and '"""' not in source else '"""'
        lines[start:end_line] = render_docstring(new_doc, indent, quote)
        changed = True

    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return changed


def main(targets: list[str]) -> int:
    """Normalize Google docstring sections for the given paths."""
    from fill_missing_docstrings import iter_targets

    count = 0
    for path in iter_targets(targets):
        if process_file(path):
            count += 1
    print(f"fixed_files={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
