#!/usr/bin/env python3
"""Add Google-style docstrings to public symbols missing documentation."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _humanize(name: str) -> str:
    name = name.strip("_")
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ")
    return words.lower()


def _summary_for(name: str, kind: str) -> str:
    if name == "__init__":
        return "Initialize the instance."
    if kind == "class":
        return f"Represent {_humanize(name)}."
    if name.startswith("test_"):
        return f"Verify {_humanize(name.removeprefix('test_'))}."
    if name.startswith("fixture_") or name in {"setup", "teardown"}:
        return f"Provide {_humanize(name)} for tests."
    return f"{_humanize(name).capitalize()}."


def _annotation_name(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Attribute):
        return f"{_annotation_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return ast.unparse(node)
    return ast.unparse(node)


def _build_args_section(args: list[ast.arg], defaults_offset: int) -> list[str]:
    lines: list[str] = []
    for index, arg in enumerate(args):
        if arg.arg in {"self", "cls"}:
            continue
        ann = _annotation_name(arg.annotation)
        desc = f"{_humanize(arg.arg)}."
        if ann:
            lines.append(f"        {arg.arg} ({ann}): {desc}")
        else:
            lines.append(f"        {arg.arg}: {desc}")
    return lines


def _build_docstring(node: ast.AST, kind: str) -> str:
    name = getattr(node, "name", "object")
    summary = _summary_for(name, kind)
    sections: list[str] = [summary, ""]

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = list(node.args.args)
        if node.args.posonlyargs:
            args = list(node.args.posonlyargs) + args
        arg_lines = _build_args_section(args, 0)
        if arg_lines:
            sections.extend(["Args:", *arg_lines, ""])

        if node.returns:
            ret = _annotation_name(node.returns)
            sections.extend(["Returns:", f"        Result ({ret})." if ret else "        Result.", ""])

    while sections and sections[-1] == "":
        sections.pop()

    body = "\n".join(sections)
    return f'"""{body}\n"""\n'


def _insert_docstring(source: str, node: ast.AST, kind: str) -> str | None:
    if ast.get_docstring(node, clean=False):
        return None
    if isinstance(node, ast.FunctionDef) and node.name.startswith("_") and node.name != "__init__":
        return None
    if isinstance(node, ast.ClassDef) and node.name.startswith("_"):
        return None

    lines = source.splitlines(keepends=True)
    if node.body:
        insert_at = node.body[0].lineno - 1
        indent_width = getattr(node.body[0], "col_offset", 0) or (
            (getattr(node, "col_offset", 0) or 0) + 4
        )
    else:
        insert_at = node.lineno
        indent_width = (getattr(node, "col_offset", 0) or 0) + 4
    indent = " " * indent_width

    doc = _build_docstring(node, kind)
    doc_lines = []
    for line in doc.splitlines():
        if line:
            doc_lines.append(f"{indent}{line}\n")
        else:
            doc_lines.append("\n")
    new_lines = lines[:insert_at] + doc_lines + lines[insert_at:]
    return "".join(new_lines)


def process_file(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    updated = source
    changed = False
    nodes: list[tuple[ast.AST, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            nodes.append((node, "class"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") and node.name != "__init__":
                continue
            nodes.append((node, "function"))

    for node, kind in sorted(nodes, key=lambda item: item[0].lineno, reverse=True):
        new_source = _insert_docstring(updated, node, kind)
        if new_source is not None:
            updated = new_source
            changed = True

    if changed and updated != source:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def iter_targets(targets: list[str]) -> list[Path]:
    paths: list[Path] = []
    for target in targets:
        base = ROOT / target
        if base.is_file() and base.suffix == ".py":
            paths.append(base)
        elif base.is_dir():
            paths.extend(
                p
                for p in sorted(base.rglob("*.py"))
                if ".venv" not in p.parts
            )
    return paths


def main(targets: list[str]) -> int:
    changed = 0
    for path in iter_targets(targets):
        if process_file(path):
            changed += 1
            print(path.relative_to(ROOT))
    print(f"filled_files={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
