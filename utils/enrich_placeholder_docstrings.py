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

COMMON_PARAM_HINTS: dict[str, str] = {
    "output": (
        "The path where the results will be placed (it is created if it does not exist)."
    ),
    "pxl_file": "Path to the input PXL (PixelDataset) file.",
    "input_pxl_file": "Path to the input PXL (PixelDataset) file.",
    "ctx": "Click context from the command decorator.",
}

STRING_LITERAL_RE = re.compile(r"""(['"])(?:\\.|(?!\1).)*\1""", re.DOTALL)


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


def _balanced_paren_end(text: str, open_paren: int) -> int:
    depth = 0
    in_string: str | None = None
    escape = False
    for index in range(open_paren, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == in_string:
                in_string = None
            continue
        if char in {"'", '"'}:
            in_string = char
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _join_string_literals(expr: str) -> str:
    parts = [match.group(0)[1:-1] for match in STRING_LITERAL_RE.finditer(expr)]
    if not parts:
        return ""
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _extract_help_from_decorator(block: str) -> str | None:
    help_match = re.search(r"\bhelp\s*=", block)
    if not help_match:
        return None
    expr_start = help_match.end()
    while expr_start < len(block) and block[expr_start].isspace():
        expr_start += 1
    if expr_start >= len(block):
        return None
    if block[expr_start] == "(":
        close = _balanced_paren_end(block, expr_start)
        if close == -1:
            return None
        help_text = _join_string_literals(block[expr_start + 1 : close])
    else:
        literal = STRING_LITERAL_RE.match(block, expr_start)
        if not literal:
            return None
        help_text = literal.group(0)[1:-1]
    if not help_text:
        return None
    help_text = help_text.replace("\\n", "\n").replace("\\t", "\t")
    help_text = re.sub(r"\s+", " ", help_text).strip()
    return help_text.rstrip(".") + "."


def _parse_click_decorators(header: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    search_at = 0
    while True:
        option_idx = header.find("@click.option", search_at)
        argument_idx = header.find("@click.argument", search_at)
        if option_idx == -1 and argument_idx == -1:
            break
        if option_idx == -1 or (argument_idx != -1 and argument_idx < option_idx):
            decorator = "@click.argument"
            start = argument_idx
        else:
            decorator = "@click.option"
            start = option_idx
        open_paren = header.find("(", start)
        if open_paren == -1:
            break
        close_paren = _balanced_paren_end(header, open_paren)
        if close_paren == -1:
            break
        block = header[start : close_paren + 1]
        search_at = close_paren + 1

        first_literal = STRING_LITERAL_RE.search(block)
        if not first_literal:
            continue
        name = _join_string_literals(first_literal.group(0))
        if not name:
            continue
        if decorator == "@click.option":
            if not name.startswith("--"):
                continue
            param = name.lstrip("-").replace("-", "_")
        else:
            param = name

        help_text = _extract_help_from_decorator(block)
        if help_text:
            mapping[param] = help_text
        elif decorator == "@click.argument" and param in COMMON_PARAM_HINTS:
            mapping[param] = COMMON_PARAM_HINTS[param]
    return mapping


def _click_help_by_param(source: str, func_name: str) -> dict[str, str]:
    match = re.search(rf"def {func_name}\s*\(", source)
    if not match:
        return {}
    header = source[: match.start()]
    mapping = dict(COMMON_PARAM_HINTS)
    mapping.update(_parse_click_decorators(header))
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
