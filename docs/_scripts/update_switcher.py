#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path

VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def main() -> None:
    base = os.environ["DOCS_BASE_URL"].rstrip("/")
    docs_dir = Path("gh-pages/docs")

    versions = []
    if docs_dir.is_dir():
        for child in docs_dir.iterdir():
            match = VERSION_RE.match(child.name)
            if child.is_dir() and match:
                versions.append((tuple(int(g) for g in match.groups()), child.name))

    # Newest first, sorted numerically (so 0.30.0 > 0.9.0, unlike string sort)
    versions.sort(reverse=True)

    data = [
        {"version": name, "url": f"{base}/{name}/"}
        for _, name in versions
    ]
    if data:
        data[0]["preferred"] = True

    out_dir = Path("switcher-out")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "switcher.json").write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
