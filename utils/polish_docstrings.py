#!/usr/bin/env python3
"""Polish Google docstrings (deprecated helper; prefer fix_docstring_sections).

Copyright © 2025 Pixelgen Technologies AB.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _humanize(name: str) -> str:
    name = name.strip("_")
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ")
    return words.lower()


def polish(text: str) -> str:
    """Polish."""
    text = re.sub(
        r"^(\s+):raise\s+",
        r"\1Raises:\n\1    ",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    text = re.sub(
        r"^(\s+)(\w[\w*]*):\s*TODO\.\s*$",
        lambda m: f"{m.group(1)}{m.group(2)}: {_humanize(m.group(2)).capitalize()}.",
        text,
        flags=re.MULTILINE,
    )

    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "Args:":
            base_indent = line[: len(line) - len(line.lstrip())]
            arg_indent = base_indent + "    "
            out.append(line)
            i += 1
            while i < len(lines):
                next_line = lines[i]
                stripped = next_line.strip()
                if not stripped:
                    out.append(next_line)
                    i += 1
                    break
                if stripped.endswith(":") and not re.match(r"^\s+\w", next_line):
                    break
                if re.match(r"^\s+\w", next_line) and not next_line.startswith(
                    arg_indent
                ):
                    name_match = re.match(r"^(\s+)(\*?\*?\w+)(.*)$", next_line)
                    if name_match:
                        _, name, rest = name_match.groups()
                        out.append(f"{arg_indent}{name}{rest}")
                    else:
                        out.append(next_line)
                else:
                    out.append(next_line)
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def main(targets: list[str]) -> int:
    """Polish docstrings under the given paths (whole-file pass; use with care)."""
    """Main."""
    from fill_missing_docstrings import iter_targets

    changed = 0
    for path in iter_targets(targets):
        original = path.read_text(encoding="utf-8")
        updated = polish(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    print(f"polished_files={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
