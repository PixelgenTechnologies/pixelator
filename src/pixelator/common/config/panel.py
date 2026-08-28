"""Marker panel management for different Molecular Pixelation assays.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import enum
import re
from pathlib import Path
from typing import List, Optional

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import pydantic
import ruamel.yaml as yaml
from anndata import AnnData
from packaging.version import Version

from pixelator.common.types import PathType
from pixelator.common.utils import logger


class PanelType(str, enum.Enum):
    """Type of antibody panel described by panel CSV metadata.

    Stored in :attr:`AntibodyPanelMetadata.panel_type` and used by PNA helpers
    such as
    :func:`pixelator.pna.config.panel.dispatch.get_panel_type_from_metadata` to
    select a concrete panel class:

    * ``partial`` — generic / legacy single panel
      (:class:`~pixelator.pna.config.panel.partial.PartialPNAAntibodyPanel`)
    * ``base`` — core marker panel
      (:class:`~pixelator.pna.config.panel.partial.PNABasePanel`)
    * ``addon`` — markers used together with a base panel
      (:class:`~pixelator.pna.config.panel.partial.PNAAddonPanel`)
    * ``sample_hashing`` — sample-hashing panel
      (:class:`~pixelator.pna.config.panel.partial.PNASampleHashingPanel`)

    ``None`` in metadata is treated as legacy ``partial``.
    """

    PARTIAL = "partial"
    BASE = "base"
    ADDON = "addon"
    SAMPLE_HASHING = "sample_hashing"


class AntibodyPanelMetadata(pydantic.BaseModel):
    """Metadata for a Molecular Pixelation antibody panel CSV.

    Parsed from the YAML front-matter of a panel file (comment lines that
    start with ``#`` followed by a space). Optional :attr:`panel_type` selects
    the concrete PNA panel class; ``None`` means legacy / untyped and is
    treated as
    :attr:`~pixelator.common.config.panel.PanelType.PARTIAL`.

    When several panels are combined in one sample, each member has its own
    metadata entry (see :meth:`from_adata`).
    """

    model_config = pydantic.ConfigDict(extra="ignore")

    version: str
    name: str
    product: Optional[str] = None
    description: Optional[str] = None
    aliases: List[str] = []
    archived: Optional[bool] = False
    panel_type: Optional[PanelType] = None

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

    def to_dict(self) -> dict:
        """Serialize metadata for storage in AnnData ``uns`` or HDF5.

        Returns:
            A plain dict from the pydantic model. ``panel_type`` is stored as
            its string value (or ``None``) so HDF5-backed AnnData can persist it.
        """
        serialized = self.model_dump()
        # panel type needs to be explicitly serialized as its value for hdf5 storage of anndata
        serialized["panel_type"] = (
            self.panel_type.value if self.panel_type is not None else None
        )
        return serialized

    @classmethod
    def _deserialize_from_adata_key(cls, adata: AnnData, key: str) -> Self:
        """Deserialize panel metadata from a key in ``adata.uns``.

        Args:
            adata: AnnData that stores panel metadata under ``uns``.
            key: ``uns`` key to read (for example ``panel_metadata`` or
                ``panel_metadata__0``).

        Returns:
            Validated metadata instance.

        Raises:
            KeyError: If ``key`` is missing from ``adata.uns``.
        """
        if key not in adata.uns:
            raise KeyError(
                f"Key {key!r} not found in adata.uns for panel metadata deserialization."
            )
        deserialized_metadata = adata.uns[key].copy()
        deserialized_metadata["panel_type"] = (
            PanelType(deserialized_metadata["panel_type"])
            if deserialized_metadata.get("panel_type") is not None
            else None
        )
        return cls.model_validate(deserialized_metadata)

    @classmethod
    def from_panel_csv(cls, panel_file: PathType) -> AntibodyPanelMetadata:
        """Parse panel metadata from a CSV file's YAML front-matter.

        Args:
            panel_file: Path to a panel ``.csv`` whose YAML header lines start
                with ``#`` followed by a space.

        Returns:
            Parsed :class:`AntibodyPanelMetadata`.

        Raises:
            ValueError: If the file has no metadata header or YAML is invalid.
        """
        return parse_panel_header_metadata(Path(panel_file))

    @classmethod
    def from_adata(cls, adata: AnnData) -> list[Self]:
        """Load panel metadata entries stored on an AnnData object.

        Prefers multi-panel keys (``num_partial_panels`` and
        ``panel_metadata__{i}``). Falls back to a single ``panel_metadata``
        entry, or a placeholder ``unknown`` / ``0.0.0`` metadata object when
        none is present.

        Args:
            adata: AnnData that may contain panel metadata in ``uns``.

        Returns:
            One metadata object per partial panel (length 1 for legacy data).

        Raises:
            KeyError: If multi-panel metadata indexes are incomplete.
            ValueError: If the number of metadata entries does not match
                ``num_partial_panels``.
        """
        if "num_partial_panels" in adata.uns:
            logger.debug(
                "Found metadata for %s partial panels in adata.uns",
                adata.uns["num_partial_panels"],
            )
            panel_metadatas = []
            for idx in range(adata.uns["num_partial_panels"]):
                if f"panel_metadata__{idx}" not in adata.uns:
                    raise KeyError(
                        "The provided AnnData object contains partial panel information but is "
                        + f"missing the metadata for panel at index {idx}."
                    )

                panel_metadatas.append(
                    cls._deserialize_from_adata_key(adata, f"panel_metadata__{idx}")
                )
            if len(panel_metadatas) != adata.uns["num_partial_panels"]:
                raise ValueError(
                    "The provided AnnData object contains partial panel information but the number "
                    + f"of panel metadata entries ({len(panel_metadatas)}) does not match the "
                    + f"expected number of partial panels ({adata.uns['num_partial_panels']})."
                )
            return panel_metadatas
        elif "panel_metadata" in adata.uns:
            logger.debug(
                'Found "panel_metadata" in adata.uns, loading panel metadata from there.'
            )
            return [cls._deserialize_from_adata_key(adata, "panel_metadata")]
        else:
            logger.warning(
                "The provided AnnData object does not contains panel metadata information."
                + "panel name and version will be set to 'unknown' and '0.0.0' respectively."
            )
            return [
                cls(
                    name="unknown",
                    version="0.0.0",
                    description="No panel metadata found in adata.uns",
                )
            ]


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
    """Parse YAML front-matter from a panel CSV comment header.

    Reads leading lines that start with ``#`` followed by a space. If the YAML
    fails to parse, retries after stripping trailing commas (a common
    spreadsheet export artifact) and emits a warning when that recovery
    succeeds.

    Args:
        file: Path to the panel CSV file.

    Returns:
        Validated :class:`AntibodyPanelMetadata`.

    Raises:
        ValueError: If no header is found or metadata cannot be parsed or
            validated (including invalid YAML / pydantic validation failures).
    """
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
