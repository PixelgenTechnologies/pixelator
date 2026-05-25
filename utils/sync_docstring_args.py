#!/usr/bin/env python3
"""Add missing Args entries to Google-style docstrings from function signatures."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _signature_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    names: list[str] = []
    args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
    for arg in args:
        if arg.arg in {"self", "cls"}:
            continue
        names.append(arg.arg)
    if node.args.vararg:
        names.append(node.args.vararg.arg)
    for arg in node.args.kwonlyargs:
        if arg.arg not in names:
            names.append(arg.arg)
    if node.args.kwarg:
        names.append(node.args.kwarg.arg)
    return names


def _existing_arg_names(doc: str) -> set[str]:
    names: set[str] = set()
    in_args = False
    for line in doc.splitlines():
        if line.strip() == "Args:":
            in_args = True
            continue
        if in_args:
            if re.match(r"^(Returns|Raises|Yields|Note|Examples|References|Attributes):", line.strip()):
                break
            match = re.match(r"^\s+(\*?\*?\w+)", line)
            if match:
                names.add(match.group(1).lstrip("*"))
    return names


def _humanize(name: str) -> str:
    name = name.strip("_")
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ")
    return words.lower()


def _add_missing_args(doc: str, missing: list[str], indent: str) -> str:
    if not missing:
        return doc
    lines = doc.splitlines()
    args_index = next((i for i, line in enumerate(lines) if line.strip() == "Args:"), None)
    arg_indent = indent + "    "
    entries = [
        f"{arg_indent}{name}: {_humanize(name).capitalize()}."
        for name in missing
    ]
    if args_index is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("Args:")
        lines.extend(entries)
        if lines and lines[-1] != "":
            lines.append("")
        return "\n".join(lines)

    insert_at = args_index + 1
    while insert_at < len(lines) and lines[insert_at].strip():
        insert_at += 1
    new_lines = lines[:insert_at] + entries + lines[insert_at:]
    return "\n".join(new_lines)


def process_file(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    lines = source.splitlines(keepends=True)
    changed = False
    for node in sorted(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        key=lambda n: n.lineno,
        reverse=True,
    ):
        doc = ast.get_docstring(node, clean=False)
        if not doc:
            continue
        required = _signature_args(node)
        existing = _existing_arg_names(doc)
        missing = [name for name in required if name not in existing]
        if not missing:
            continue
        indent = " " * ((node.body[0].col_offset if node.body else node.col_offset + 4) or 4)
        new_doc = _add_missing_args(doc, missing, indent)
        if new_doc == doc:
            continue
        # Replace docstring in source via line-based approach
        if not node.body:
            continue
        start = node.body[0].lineno - 1
        end_line = node.body[0].end_lineno
        if not isinstance(node.body[0], ast.Expr):
            continue
        old_lines = lines[start:end_line]
        quote = '"""' if '"""' in "".join(old_lines) else "'''"
        new_block = [f'{indent}{quote}{new_doc}\n{indent}{quote}\n']
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
            print(path.relative_to(ROOT))
    print(f"synced_files={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
