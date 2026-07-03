#!/usr/bin/env python3
import json
import os
from pathlib import Path


def main() -> None:
    version = os.environ["VERSION"]
    base = os.environ["DOCS_BASE_URL"].rstrip("/")

    entry = {
        "version": version,
        "url": f"{base}/{version}/",
    }

    src = Path("gh-pages/docs/switcher.json")
    if src.exists():
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    else:
        data = []

    exists = any(
        isinstance(item, dict) and str(item.get("version", "")) == version
        for item in data
    )
    if not exists:
        data.append(entry)

    out_dir = Path("switcher-out")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "switcher.json").write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
