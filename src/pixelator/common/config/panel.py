"""Marker panel management for different Molecular Pixelation assays.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import pydantic
import ruamel.yaml as yaml
from packaging.version import Version

from pixelator.common.utils import logger


class AntibodyPanelMetadata(pydantic.BaseModel):
    """Class representing the metadata of a Molecular Pixelation antibody panel."""

    model_config = pydantic.ConfigDict(extra="ignore")

    version: str
    name: str
    product: Optional[str] = None
    description: Optional[str] = None
    aliases: List[str] = []
    archived: Optional[bool] = False

    @pydantic.field_validator("version")
    @classmethod
    def check_version(cls, v: str) -> str:
        """Validate that the panel version string is parseable.

        Args:
            v: Version string from panel metadata.

        Returns:
            The input version string when validation succeeds.

        Raises:
            packaging.version.InvalidVersion: If the value is not a valid version.
        """
        Version(v)  # will raise if not a valid version string
        return v


def _strip_trailing_commas(metadata: str) -> tuple[str, bool]:
    """Remove line-end commas from header YAML.

    This keeps recovery narrow to the malformed pattern we want to tolerate.
    """
    normalized = re.sub(r",(\s*(?:\n|$))", r"\1", metadata)
    return normalized, normalized != metadata


def _load_header_frontmatter(metadata: str) -> AntibodyPanelMetadata:
    """Load and validate first YAML document from panel metadata text."""
    yaml_loader = yaml.YAML(typ="safe")
    raw_config = list(yaml_loader.load_all(metadata))

    if len(raw_config) == 0:
        raise ValueError("No header / metadata found in panel file")

    frontmatter = raw_config[0]
    return AntibodyPanelMetadata.model_validate(frontmatter)


def parse_panel_header_metadata(file: Path) -> AntibodyPanelMetadata:
    """Parse panel front-matter metadata and recover from trailing commas."""
    metadata_lines = []
    with open(str(file), "r") as handle:
        for line in handle:
            if line.startswith("# "):
                metadata_lines.append(line[2:])
            else:
                break

    metadata = "".join(metadata_lines)
    try:
        return _load_header_frontmatter(metadata)
    except (yaml.YAMLError, pydantic.ValidationError, ValueError):
        normalized_metadata, changed = _strip_trailing_commas(metadata)
        if not changed:
            if metadata.strip() == "":
                raise ValueError(f"No header / metadata found in panel file {file}")
            raise

        try:
            parsed = _load_header_frontmatter(normalized_metadata)
        except (yaml.YAMLError, pydantic.ValidationError, ValueError):
            if metadata.strip() == "":
                raise ValueError(f"No header / metadata found in panel file {file}")
            raise

        logger.warning(
            "Panel header in %s contains trailing comma(s); parsing with commas ignored.",
            file,
        )
        return parsed
