#!/usr/bin/env python3
"""Convert Sphinx/reST docstring sections to Google style (format-only pass).

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PARAM_RE = re.compile(
    r"^(\s*):param\s+(\S+)(?:\s+([^:]+))?:\s*(.*)$",
)
RETURN_RE = re.compile(r"^(\s*):returns?:\s*(.*)$", re.IGNORECASE)
RTYPE_RE = re.compile(r"^(\s*):rtype:\s*(.*)$", re.IGNORECASE)
RAISES_RE = re.compile(r"^(\s*):raises\s+([^:]+):\s*(.*)$")
RAISES_BARE_RE = re.compile(r"^(\s*):raises:\s*(.*)$", re.IGNORECASE)
IVAR_RE = re.compile(r"^(\s*):ivar\s+(\S+):\s*(.*)$")
YIELD_RE = re.compile(r"^(\s*):yields:\s*(.*)$", re.IGNORECASE)
YTYPE_RE = re.compile(r"^(\s*):ytype:\s*(.*)$", re.IGNORECASE)

SECTION_HEADERS = {
    "args": "Args:",
    "returns": "Returns:",
    "raises": "Raises:",
    "attributes": "Attributes:",
    "yields": "Yields:",
}


def _indent_for_section(line: str) -> str:
    match = re.match(r"^(\s+)", line)
    return match.group(1) if match else "    "


def convert_docstring_body(body: str) -> str:
    """Convert docstring body."""
    if not any(
        marker in body
        for marker in (
            ":param",
            ":return",
            ":returns",
            ":rtype",
            ":raises",
            ":ivar",
            ":yields",
        )
    ):
        return body

    lines = body.split("\n")
    narrative: list[str] = []
    sections: dict[str, list[str]] = {
        "args": [],
        "returns": [],
        "raises": [],
        "attributes": [],
        "yields": [],
    }
    pending_return_type: str | None = None
    base_indent = "    "

    def append_continuation(start_index: int, target: str) -> int:
        j = start_index
        while j < len(lines):
            cont = lines[j]
            if not cont.strip():
                break
            if re.match(r"^\s*:", cont):
                break
            target_list = sections[target]
            if target_list:
                target_list[-1] = f"{target_list[-1]} {cont.strip()}"
            j += 1
        return j

    i = 0
    while i < len(lines):
        line = lines[i]
        param_match = PARAM_RE.match(line)
        if param_match:
            indent, name, type_hint, desc = param_match.groups()
            base_indent = indent or base_indent
            entry = f"{name}"
            if type_hint:
                entry += f" ({type_hint.strip()})"
            entry += f": {desc.strip()}"
            sections["args"].append(entry)
            i = append_continuation(i + 1, "args")
            continue

        return_match = RETURN_RE.match(line)
        if return_match:
            _, desc = return_match.groups()
            sections["returns"].append(desc.strip())
            i = append_continuation(i + 1, "returns")
            continue

        rtype_match = RTYPE_RE.match(line)
        if rtype_match:
            pending_return_type = rtype_match.group(2).strip()
            i += 1
            continue

        raises_match = RAISES_RE.match(line)
        if raises_match:
            _, exc, desc = raises_match.groups()
            sections["raises"].append(f"{exc}: {desc.strip()}")
            i += 1
            continue

        raises_bare_match = RAISES_BARE_RE.match(line)
        if raises_bare_match:
            sections["raises"].append(raises_bare_match.group(2).strip())
            i += 1
            continue

        ivar_match = IVAR_RE.match(line)
        if ivar_match:
            _, name, desc = ivar_match.groups()
            sections["attributes"].append(f"{name}: {desc.strip()}")
            i += 1
            continue

        yield_match = YIELD_RE.match(line)
        if yield_match:
            sections["yields"].append(yield_match.group(2).strip())
            i += 1
            continue

        ytype_match = YTYPE_RE.match(line)
        if ytype_match:
            i += 1
            continue

        narrative.append(line)
        i += 1

    if pending_return_type and sections["returns"]:
        sections["returns"][0] = f"{sections['returns'][0]} ({pending_return_type})"
    elif pending_return_type and not sections["returns"]:
        sections["returns"].append(pending_return_type)

    # Trim trailing blank lines from narrative
    while narrative and narrative[-1] == "":
        narrative.pop()

    out: list[str] = list(narrative)
    if narrative and any(sections.values()):
        out.append("")

    for key in ("args", "returns", "raises", "yields", "attributes"):
        entries = sections[key]
        if not entries:
            continue
        out.append(SECTION_HEADERS[key])
        for entry in entries:
            out.append(f"{base_indent}{entry}")
        out.append("")

    if out and out[-1] == "":
        out.pop()

    return "\n".join(out)


def convert_file(path: Path) -> bool:
    """Convert file."""
    original = path.read_text(encoding="utf-8")
    if (
        ":param" not in original
        and ":return" not in original
        and ":raises" not in original
    ):
        if ":ivar" not in original and ":rtype" not in original:
            return False

    # Process triple-quoted docstrings with a simple state machine
    result: list[str] = []
    i = 0
    changed = False
    n = len(original)

    while i < n:
        if original.startswith('"""', i) or original.startswith("'''", i):
            quote = original[i : i + 3]
            j = i + 3
            # Skip module docstring at file start if needed - still convert all
            body_start = j
            while j < n:
                if original.startswith(quote, j):
                    if j > 0 and original[j - 1] == "\\":
                        j += 1
                        continue
                    body = original[body_start:j]
                    converted = convert_docstring_body(body)
                    if converted != body:
                        changed = True
                    result.append(quote)
                    result.append(converted)
                    result.append(quote)
                    i = j + 3
                    break
                j += 1
            else:
                result.append(original[i])
                i += 1
            continue
        result.append(original[i])
        i += 1

    new_text = "".join(result)
    if changed and new_text != original:
        path.write_text(new_text, encoding="utf-8")
        return True
    return False


def iter_targets(targets: list[str]) -> list[Path]:
    """Iter targets."""
    paths: list[Path] = []
    for target in targets:
        base = ROOT / target
        if base.is_file() and base.suffix == ".py":
            paths.append(base)
        elif base.is_dir():
            paths.extend(
                p for p in sorted(base.rglob("*.py")) if ".venv" not in p.parts
            )
    return paths


def main(targets: list[str]) -> int:
    """Convert Sphinx-style docstrings under the given paths."""
    """Main."""
    changed_files = 0
    for path in iter_targets(targets):
        if convert_file(path):
            changed_files += 1
            print(path.relative_to(ROOT))
    print(f"converted_files={changed_files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
