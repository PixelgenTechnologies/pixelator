#!/usr/bin/env python3
"""Replace tautological Google Args with richer text when available.

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PLACEHOLDER_RE = re.compile(r"^(\s+)(\w+):\s*([A-Z][a-z_ ]*|\w+)\.\s*$")
CLICK_OPTION_RE = re.compile(
    r'@click\.option\([^)]*?["\']--([\w-]+)["\'][^)]*?help\s*=\s*["\'](.+?)["\']',
    re.DOTALL,
)


def _arg_map_from_doc(doc: str) -> dict[str, str]:
    args: dict[str, str] = {}
    in_args = False
    for line in doc.splitlines():
        if line.strip() == "Args:":
            in_args = True
            continue
        if in_args:
            if re.match(
                r"^(Returns|Raises|Yields|Note|Examples|References|Attributes):",
                line.strip(),
            ):
                break
            match = re.match(r"^\s+(\*?\*?\w[\w*]*)(?:\s*\([^)]*\))?\s*:\s*(.*)$", line)
            if match:
                args[match.group(1).lstrip("*")] = match.group(2).strip()
    return args


def _is_placeholder(name: str, desc: str) -> bool:
    if not desc.endswith("."):
        return False
    humanized = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ").strip()
    return desc in {
        humanized.capitalize() + ".",
        humanized.title() + ".",
        name.capitalize() + ".",
        name + ".",
    }


def _collect_file_arg_hints(tree: ast.AST) -> dict[str, str]:
    """Best non-placeholder Arg descriptions seen anywhere in the file."""
    hints: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node, clean=False)
        if not doc:
            continue
        for name, desc in _arg_map_from_doc(doc).items():
            if _is_placeholder(name, desc):
                continue
            if name not in hints or len(desc) > len(hints[name]):
                hints[name] = desc
    return hints


def _click_help_by_param(source: str, func_name: str) -> dict[str, str]:
    match = re.search(rf"def {func_name}\s*\(", source)
    if not match:
        return {}
    header = source[: match.start()]
    options = list(CLICK_OPTION_RE.finditer(header))
    mapping: dict[str, str] = {}
    for opt in options[-30:]:
        flag, help_text = opt.group(1), re.sub(r"\s+", " ", opt.group(2)).strip()
        param = flag.replace("-", "_")
        mapping[param] = help_text.rstrip(".") + "."
    return mapping


def _replace_docstring(
    source: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    new_doc: str,
) -> str:
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


def _enrich_doc(doc: str, hints: dict[str, str]) -> tuple[str, bool]:
    lines = doc.splitlines()
    changed = False
    new_lines: list[str] = []
    in_args = False
    for line in lines:
        stripped = line.strip()
        if stripped == "Args:":
            in_args = True
            new_lines.append(line)
            continue
        if in_args:
            if re.match(
                r"^(Returns|Raises|Yields|Note|Examples|References|Attributes):",
                stripped,
            ):
                in_args = False
                new_lines.append(line)
                continue
            match = re.match(r"^(\s+)(\w+):\s*(.*)$", line)
            if match:
                indent, name, desc = match.groups()
                if name in hints and _is_placeholder(name, desc.strip()):
                    new_lines.append(f"{indent}{name}: {hints[name]}")
                    changed = True
                    continue
        new_lines.append(line)
    return "\n".join(new_lines), changed


def process_file(path: Path) -> bool:
    """Enrich placeholder Google Args in one Python file."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    file_changed = False
    while True:
        tree = ast.parse(source)
        file_hints = _collect_file_arg_hints(tree)
        pending: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, str]] = []

        class Walker(ast.NodeVisitor):
            def _visit_function(
                self, node: ast.FunctionDef | ast.AsyncFunctionDef
            ) -> None:
                doc = ast.get_docstring(node, clean=False)
                if not doc:
                    return
                hints = dict(file_hints)
                hints.update(_click_help_by_param(source, node.name))
                new_doc, changed = _enrich_doc(doc, hints)
                if changed:
                    pending.append((node, new_doc))

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._visit_function(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._visit_function(node)

        Walker().visit(tree)
        if not pending:
            break
        pending.sort(key=lambda item: item[0].lineno, reverse=True)
        node, new_doc = pending[0]
        source = _replace_docstring(source, node, new_doc)
        file_changed = True

    if file_changed:
        path.write_text(source, encoding="utf-8")
    return file_changed


def main(targets: list[str]) -> int:
    """Run placeholder enrichment for the given paths."""
    changed = 0
    for target in targets:
        base = ROOT / target
        paths = (
            [base]
            if base.is_file()
            else sorted(p for p in base.rglob("*.py") if ".venv" not in p.parts)
        )
        for path in paths:
            if process_file(path):
                print(path.relative_to(ROOT))
                changed += 1
    print(f"enriched_files={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["src/pixelator"]))
