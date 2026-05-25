#!/usr/bin/env python3
"""Normalize Google docstring section indentation inside docstrings."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ("Args", "Returns", "Raises", "Yields", "Attributes", "Examples", "References", "Note")


def normalize_docstring(doc: str) -> str:
    lines = [line.rstrip() for line in doc.splitlines()]
    if not lines:
        return doc

    narrative: list[str] = []
    sections: dict[str, list[str]] = {name: [] for name in SECTIONS}
    current: str | None = None

    for line in lines:
        header = line.strip()
        if header in {f"{name}:" for name in SECTIONS}:
            current = header[:-1]
            continue
        if current:
            if not line.strip():
                continue
            match = re.match(r"^\s+(\*?\*?\w[\w*]*)\s*:\s*(.*)$", line)
            if match:
                name, desc = match.groups()
                sections[current].append(f"{name}: {desc.strip()}")
            else:
                if sections[current]:
                    sections[current][-1] = f"{sections[current][-1]} {line.strip()}"
        else:
            narrative.append(line)

    while narrative and not narrative[-1].strip():
        narrative.pop()

    out = list(narrative)
    if narrative and any(sections.values()):
        out.append("")

    for name in SECTIONS:
        entries = sections[name]
        if not entries:
            continue
        out.append(f"{name}:")
        for entry in entries:
            out.append(f"    {entry}")
        out.append("")

    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def process_file(path: Path) -> bool:
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
        if not doc or "Args:" not in doc and ":param" not in doc:
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
        quote = '"""'
        new_block = [f"{indent}{quote}{new_doc}\n{indent}{quote}\n"]
        lines[start:end_line] = new_block
        changed = True

    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return changed


def main(targets: list[str]) -> int:
    from fill_missing_docstrings import iter_targets

    count = 0
    for path in iter_targets(targets):
        if process_file(path):
            count += 1
    print(f"fixed_files={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
