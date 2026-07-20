#!/usr/bin/env python3
"""Generate the version switcher JSON for the docs site."""

import json
import os
import re
from pathlib import Path

VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

REDIRECT_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Redirecting...</title>
  <meta http-equiv="refresh" content="0; url={version}/">
  <link rel="canonical" href="{version}/">
  <script>location.replace("{version}/");</script>
</head>
<body>
  <p>Redirecting to <a href="{version}/">{version}</a>…</p>
</body>
</html>
"""


def main() -> None:
    """Build switcher JSON from versioned docs directories."""
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

    data = [{"version": name, "url": f"{base}/{name}/"} for _, name in versions]
    if data:
        data[0]["preferred"] = True

    out_dir = Path("switcher-out")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "switcher.json").write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )

    if versions:
        latest = versions[0][1]
        (out_dir / "index.html").write_text(
            REDIRECT_HTML.format(version=latest),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
