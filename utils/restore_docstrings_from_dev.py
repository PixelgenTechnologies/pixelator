#!/usr/bin/env python3
"""Merge missing Google docstring sections from dev Sphinx docstrings.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import ast
import inspect
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
from convert_sphinx_to_google import convert_docstring_body  # noqa: E402

SECTION_ORDER = (
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
SECTION_SET = {f"{name}:" for name in SECTION_ORDER}


def _dev_blob(path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "show", f"dev:{path}"],
            text=True,
            stderr=subprocess.DEVNULL,
            cwd=ROOT,
        )
    except subprocess.CalledProcessError:
        return None


def _parse_sections(doc: str) -> tuple[list[str], dict[str, list[str]]]:
    lines = inspect.cleandoc(doc).splitlines()
    narrative: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for line in lines:
        stripped = line.strip()
        if stripped in SECTION_SET:
            current = stripped[:-1]
            sections[current] = []
            continue
        if current is None:
            narrative.append(line)
            continue
        if re.match(r"^[A-Z][A-Za-z ]+:$", stripped) and stripped not in SECTION_SET:
            current = None
            narrative.append(line)
            continue
        sections[current].append(line)

    while narrative and not narrative[-1].strip():
        narrative.pop()
    return narrative, sections


def _format_sections(sections: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for name in SECTION_ORDER:
        if name not in sections or not sections[name]:
            continue
        out.append(f"{name}:")
        for entry in sections[name]:
            stripped = entry.strip()
            if not stripped:
                continue
            if name in {"Args", "Attributes", "Raises"}:
                out.append(f"    {stripped}")
            else:
                out.append(f"    {stripped}")
    return out


def _arg_map(section_lines: list[str]) -> dict[str, str]:
    args: dict[str, str] = {}
    for line in section_lines:
        match = re.match(r"^\s+(\*?\*?\w[\w*]*)(?:\s*\([^)]*\))?\s*:\s*(.*)$", line)
        if match:
            args[match.group(1).lstrip("*")] = match.group(2).strip()
    return args


def _is_placeholder(name: str, desc: str) -> bool:
    if not desc or not desc.endswith("."):
        return False
    humanized = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ").strip()
    return desc in {
        humanized.capitalize() + ".",
        humanized.title() + ".",
        name.capitalize() + ".",
        name + ".",
    }


def _merge_docs(head_doc: str, dev_doc: str) -> tuple[str, bool]:
    dev_google = convert_docstring_body(inspect.cleandoc(dev_doc))
    head_narrative, head_sections = _parse_sections(head_doc)
    _, dev_sections = _parse_sections(dev_google)
    changed = False

    for section in ("Returns", "Raises", "Yields", "Attributes"):
        if (
            section in dev_sections
            and dev_sections[section]
            and section not in head_sections
        ):
            head_sections[section] = dev_sections[section]
            changed = True
        elif (
            section in dev_sections
            and dev_sections[section]
            and section in head_sections
            and not any(line.strip() for line in head_sections[section])
        ):
            head_sections[section] = dev_sections[section]
            changed = True

    head_args = _arg_map(head_sections.get("Args", []))
    dev_args = _arg_map(dev_sections.get("Args", []))
    new_arg_lines: list[str] = []
    for name, desc in head_args.items():
        if (
            name in dev_args
            and _is_placeholder(name, desc)
            and not _is_placeholder(name, dev_args[name])
        ):
            new_arg_lines.append(f"    {name}: {dev_args[name]}")
            changed = True
        else:
            new_arg_lines.append(f"    {name}: {desc}")
    for name, desc in dev_args.items():
        if name not in head_args:
            new_arg_lines.append(f"    {name}: {desc}")
            changed = True
    if new_arg_lines:
        head_sections["Args"] = new_arg_lines

    if not changed:
        return head_doc, False

    out_lines = list(head_narrative)
    if out_lines and _format_sections(head_sections):
        out_lines.append("")
    out_lines.extend(_format_sections(head_sections))
    return "\n".join(out_lines), True


def _collect_docs(tree: ast.AST) -> dict[str, str | None]:
    docs: dict[str, str | None] = {}
    stack: list[str] = []

    class Walker(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            q = ".".join(stack + [node.name]) if stack else node.name
            docs[q] = ast.get_docstring(node, clean=False)
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            q = ".".join(stack + [node.name]) if stack else node.name
            docs[q] = ast.get_docstring(node, clean=False)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node)

    Walker().visit(tree)
    return docs


def _replace_docstring(source: str, node: ast.AST, new_doc: str) -> str:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return source
    if not node.body:
        return source
    first = node.body[0]
    if not (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return source

    lines = source.splitlines(keepends=True)
    start = first.lineno - 1
    end = first.end_lineno
    indent = lines[start][: first.col_offset]
    inner = indent + "    "
    doc_lines = new_doc.splitlines()
    if len(doc_lines) == 1:
        block = [f'{indent}"""{new_doc}"""\n']
    else:
        block = [f'{indent}"""\n']
        for dl in doc_lines:
            block.append(f"{inner}{dl}\n")
        block.append(f'{indent}"""\n')
    lines[start:end] = block
    return "".join(lines)


def _iter_nodes(
    tree: ast.AST,
) -> list[tuple[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, str]]:
    nodes: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, str]] = []
    stack: list[str] = []

    class Walker(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            q = ".".join(stack + [node.name]) if stack else node.name
            nodes.append((node, q))
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            q = ".".join(stack + [node.name]) if stack else node.name
            nodes.append((node, q))

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node)

    Walker().visit(tree)
    return nodes


def process_file(path: Path) -> bool:
    """Restore missing Google docstring sections from dev for one file."""
    rel = path.relative_to(ROOT).as_posix()
    dev_source = _dev_blob(rel)
    if dev_source is None:
        return False

    source = path.read_text(encoding="utf-8")
    try:
        dev_docs = _collect_docs(ast.parse(dev_source))
        ast.parse(source)
    except SyntaxError:
        return False

    file_changed = False
    while True:
        tree = ast.parse(source)
        pending: list[
            tuple[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, str]
        ] = []
        for node, qual in _iter_nodes(tree):
            head_doc = ast.get_docstring(node, clean=False)
            dev_doc = dev_docs.get(qual)
            if not head_doc or not dev_doc or head_doc == dev_doc:
                continue
            merged, changed = _merge_docs(head_doc, dev_doc)
            if changed:
                pending.append((node, merged))
        if not pending:
            break
        pending.sort(key=lambda item: item[0].lineno, reverse=True)
        node, merged = pending[0]
        source = _replace_docstring(source, node, merged)
        file_changed = True

    if file_changed:
        path.write_text(source, encoding="utf-8")
    return file_changed


def main(targets: list[str]) -> int:
    """Run dev docstring restoration for the given paths."""
    changed = 0
    for target in targets:
        base = ROOT / target
        paths = (
            [base]
            if base.is_file()
            else sorted(
                p
                for p in base.rglob("*.py")
                if ".venv" not in p.parts and "__pycache__" not in p.parts
            )
        )
        for path in paths:
            if process_file(path):
                print(path.relative_to(ROOT))
                changed += 1
    print(f"restored_files={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["src/pixelator"]))
