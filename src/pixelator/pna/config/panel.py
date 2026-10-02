"""Marker panel management for different PNA assays.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Set

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import re

import pandas as pd
import polars as pl
from anndata import AnnData
from packaging.version import Version

from pixelator.common.config.panel import (
    AntibodyPanelMetadata,
    parse_panel_header_metadata,
)
from pixelator.common.types import PathType
from pixelator.common.utils import logger

if TYPE_CHECKING:
    from pixelator.pna.config.config_class import PNAConfig

# Trailing ``-<digits>`` is the hash group (``B2M-1`` → ``B2M``). The same
# pattern matches ordinary names such as ``PD-1``, so it is only applied to
# rows already flagged by ``sample_hashing``.
_HASHING_MARKER_ID_RE = re.compile(r"^(?P<base>.+)-(?P<index>\d+)$")


@dataclass(frozen=True)
class PanelSource:
    """One panel file that contributed markers to a ``PNAAntibodyPanel``."""

    metadata: AntibodyPanelMetadata
    file_name: str | None = None
    filepath: str | None = None


def sample_hashing_mask(sample_hashing: pd.Series) -> pd.Series:
    """Return a boolean mask for values that flag a hashing marker."""
    if pd.api.types.is_bool_dtype(sample_hashing):
        return sample_hashing.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(sample_hashing):
        return sample_hashing.fillna(0).astype(bool)
    normalized = sample_hashing.astype(str).str.strip().str.lower()
    return normalized.isin(["yes", "true"])


def split_hashing_marker_id(marker_id: str) -> tuple[str, str] | None:
    """Return ``(base, index)`` for a hashing id such as ``B2M-1``."""
    match = _HASHING_MARKER_ID_RE.fullmatch(str(marker_id))
    if match is None:
        return None
    return match.group("base"), match.group("index")


def collapsed_hashing_marker_id(marker_id: str) -> str:
    """Return the marker id sample calling stores for a hashing antibody."""
    parts = split_hashing_marker_id(marker_id)
    return parts[0] if parts is not None else str(marker_id)


def _hashing_marker_ids(panel_df: pd.DataFrame) -> set[str]:
    """Return hashing marker ids, or an empty set when the column is absent."""
    if "sample_hashing" not in panel_df.columns:
        return set()
    mask = sample_hashing_mask(panel_df["sample_hashing"])
    return {str(marker_id) for marker_id in panel_df.index[mask]}


class PNAAntibodyPanel:
    """Class representing a PNA antibody panel."""

    # required columns
    _INDEX_COLUMN = "marker_id"
    _INDEX_COLUMN_TYPE = str
    _REQUIRED_COLUMNS = {
        "control": bool,
        "sequence_1": str,
        "sequence_2": str,
    }

    # and these should have unique values
    _UNIQUE_COLUMNS = ["sequence_1", "sequence_2"]

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: AntibodyPanelMetadata | None = None,
        file_name: Optional[str] = None,
        filepath: Optional[PathType] = None,
        *,
        sources: list[PanelSource] | None = None,
        marker_source_ids: pd.Series | None = None,
    ) -> None:
        """Build a panel from a marker table.

        Pass ``metadata`` or ``sources``, not both. ``metadata`` becomes one
        source, using ``file_name`` and ``filepath``. Pass ``sources`` for a
        panel that already names its files. The ``metadata`` property returns
        that entry only when the panel has one source.

        Args:
            df: Marker table, indexed by marker id.
            metadata: Metadata for the single source. Omit when passing
                ``sources``.
            file_name: Basename of the file this panel was loaded from.
            filepath: Full path of the file this panel was loaded from.
            sources: Panel files that contributed markers. Omit when passing
                ``metadata``.
            marker_source_ids: Source index for each marker. Required when
                there is more than one source.

        Raises:
            ValueError: If both ``metadata`` and ``sources`` are omitted or
                both are given, or if several sources are given without
                ``marker_source_ids``.
            AssertionError: If the marker table fails panel validation.
        """
        self._filename = file_name
        self._filepath: Optional[Path] = Path(filepath).resolve() if filepath else None
        self._df = df
        if sources is not None and metadata is not None:
            raise ValueError("Pass metadata or sources, not both.")
        if sources is None:
            if metadata is None:
                raise ValueError("Pass metadata or sources.")
            sources = [
                PanelSource(
                    metadata=metadata,
                    file_name=file_name,
                    filepath=str(self._filepath) if self._filepath else None,
                )
            ]
        self.sources: list[PanelSource] = list(sources)
        if marker_source_ids is None:
            if len(self.sources) > 1:
                raise ValueError(
                    "marker_source_ids is required when a panel has multiple sources."
                )
            marker_source_ids = pd.Series(0, index=df.index, dtype="int64")
        self._marker_source_ids = marker_source_ids

        # validate the panel
        errors = self.validate_antibody_panel(df)
        if len(errors) > 0:
            msg_str = "\n".join(errors)
            raise AssertionError(
                f"The following errors were found validating the panel: {msg_str}"
            )

    @classmethod
    def from_csv(cls, filename: PathType) -> Self:
        """Create an AntibodyPanel from a csv panel file.

        Args:
            filename: The path to the panel file.

        Returns:
            The AntibodyPanel object. (AntibodyPanel)

        Raises:
            AssertionError: exception if panel file is missing,
        """
        panel_file = Path(filename)

        if not panel_file.is_file() or panel_file.suffix != ".csv":
            raise AssertionError(
                f"Panel file {filename} not found or has an incorrect format"
            )

        logger.debug("Creating Antibody panel from file %s", filename)

        df = cls._parse_panel(panel_file)
        metadata = cls._parse_header(panel_file)

        logger.debug("Antibody panel from file %s created", filename)

        return cls(df, metadata, file_name=panel_file.name, filepath=panel_file)

    @classmethod
    def from_legacy_var(
        cls,
        var: pd.DataFrame,
        panel_metadata: dict,
        *,
        file_name: str | None = None,
    ) -> Self:
        """Build a panel from a pixelator 0.22.0 through 0.30.0 ``var`` table.

        ``panel_metadata`` is the ``uns['panel_metadata']`` entry, and ``var``
        is indexed by marker id. Those releases wrote ``panel_columns`` and
        the columns it names together.
        """
        df = var[list(panel_metadata["panel_columns"])]
        metadata = AntibodyPanelMetadata.model_validate(panel_metadata)
        return cls(df, metadata, file_name=file_name)

    @classmethod
    def _from_panel_tables(cls, markers: pd.DataFrame, sources: pd.DataFrame) -> Self:
        """Build a panel from the stored marker and source tables."""
        panel_sources: list[PanelSource] = []
        for row in sources.sort_values("source_id").itertuples(index=False):
            aliases = json.loads(row.aliases) if isinstance(row.aliases, str) else []
            metadata = AntibodyPanelMetadata(
                name=row.name,
                version=row.version,
                product=None if pd.isna(row.product) else row.product,
                description=None if pd.isna(row.description) else row.description,
                aliases=aliases or [],
                archived=bool(row.archived) if not pd.isna(row.archived) else False,
            )
            file_name = None if pd.isna(row.file_name) else row.file_name
            filepath = None if pd.isna(row.filepath) else row.filepath
            panel_sources.append(
                PanelSource(metadata=metadata, file_name=file_name, filepath=filepath)
            )

        markers = markers.sort_values("row_nr")
        source_ids = markers["source_id"].astype(int)
        drop_cols = ["row_nr", "source_id"]
        df = markers.drop(columns=[col for col in drop_cols if col in markers.columns])
        df = df.set_index("marker_id")
        df.index.name = "marker_id"
        source_ids.index = df.index
        file_name = panel_sources[0].file_name if len(panel_sources) == 1 else None
        filepath = panel_sources[0].filepath if len(panel_sources) == 1 else None
        return cls(
            df,
            file_name=file_name,
            filepath=filepath,
            sources=panel_sources,
            marker_source_ids=source_ids.astype("int64"),
        )

    @classmethod
    def concatenate(cls, panels: Sequence[PNAAntibodyPanel]) -> PNAAntibodyPanel:
        """Concatenate panels into one panel.

        One panel is returned unchanged. Several panels are stacked in the
        given order. The result keeps every input source.
        ``marker_id``, ``sequence_1``, and ``sequence_2`` must be unique
        across the concatenation. An optional column present on only some
        sources is blank on the others, the same as an empty cell in a panel CSV.
        """
        if not panels:
            raise ValueError("At least one panel is required to concatenate.")
        if len(panels) == 1:
            return panels[0]

        frames: list[pd.DataFrame] = []
        sources: list[PanelSource] = []
        source_id_frames: list[pd.Series] = []
        for panel in panels:
            if not panel.sources:
                raise ValueError("Cannot concatenate a panel that has no sources.")
            for source_index, source in enumerate(panel.sources):
                new_source_id = len(sources)
                sources.append(source)
                marker_index = panel.marker_source_ids.index[
                    panel.marker_source_ids == source_index
                ]
                part = panel.df.loc[list(marker_index)]
                frames.append(part)
                source_id_frames.append(
                    pd.Series(new_source_id, index=part.index, dtype="int64")
                )

        df = pd.concat(cls._align_optional_columns(frames))
        df.index.name = cls._INDEX_COLUMN
        if "control" in df.columns:
            df["control"] = df["control"].map(
                lambda value: bool(value) if pd.notna(value) else False
            )
        marker_source_ids = pd.concat(source_id_frames)
        marker_source_ids.index = df.index
        return cls(
            df,
            sources=sources,
            marker_source_ids=marker_source_ids.astype("int64"),
        )

    @staticmethod
    def _align_optional_columns(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """Give every frame the same columns before they are stacked.

        ``pd.concat`` inserts NaN where a column exists on only some frames.
        A blank UniProt id is an empty string, and a missing hashing flag is
        false. Filling after the stack mixes those types and breaks validation.
        """
        columns = list(
            dict.fromkeys(column for frame in frames for column in frame.columns)
        )
        missing_on_some = [
            column
            for column in columns
            if any(column not in frame.columns for frame in frames)
        ]
        if not missing_on_some:
            return frames
        fills = {
            column: PNAAntibodyPanel._missing_optional_value(frames, column)
            for column in missing_on_some
        }
        aligned: list[pd.DataFrame] = []
        for frame in frames:
            missing = {
                column: fill
                for column, fill in fills.items()
                if column not in frame.columns
            }
            if not missing:
                aligned.append(frame)
                continue
            extra = pd.DataFrame(missing, index=frame.index)
            aligned.append(pd.concat([frame, extra], axis=1))
        return aligned

    @staticmethod
    def _missing_optional_value(frames: list[pd.DataFrame], column: str):
        """Return the blank value for a column some sources do not have."""
        if column == "control":
            return False
        present = [
            frame[column].dropna() for frame in frames if column in frame.columns
        ]
        values = pd.concat(present) if present else pd.Series(dtype=object)
        if values.empty:
            return ""
        if pd.api.types.is_bool_dtype(values) or all(
            isinstance(value, bool) for value in values
        ):
            return False
        if pd.api.types.is_numeric_dtype(values):
            return 0
        return ""

    def _single_metadata(self) -> AntibodyPanelMetadata | None:
        """Return metadata when this panel has exactly one source."""
        if len(self.sources) != 1:
            return None
        return self.sources[0].metadata

    def _require_single_metadata(self, field: str) -> AntibodyPanelMetadata:
        """Return the only source metadata, refusing a concatenated panel."""
        metadata = self._single_metadata()
        if metadata is None:
            raise ValueError(
                f"Panel {field} is only available for a single source. "
                "Read it from each entry in sources."
            )
        return metadata

    @property
    def metadata(self) -> AntibodyPanelMetadata:
        """Metadata for the only source in this panel.

        Raises:
            ValueError: When this panel does not have exactly one source.
                Read ``sources`` instead.
        """
        return self._require_single_metadata("metadata")

    @property
    def name(self) -> str:
        """Panel name.

        Raises:
            ValueError: When this panel does not have exactly one source.
                Read ``sources`` instead.
        """
        return self._require_single_metadata("name").name

    @property
    def product(self) -> Optional[str]:
        """Product identifier from metadata, if present.

        Returns:
            Product name, or None when the single source does not set one.

        Raises:
            ValueError: When this panel does not have exactly one source.
                Read ``sources`` instead.
        """
        return self._require_single_metadata("product").product

    @property
    def version(self) -> str:
        """Panel version.

        Raises:
            ValueError: When this panel does not have exactly one source.
                Read ``sources`` instead.
        """
        return self._require_single_metadata("version").version

    @property
    def description(self) -> Optional[str]:
        """Return the panel file description.

        Raises:
            ValueError: When this panel does not have exactly one source.
                Read ``sources`` instead.
        """
        return self._require_single_metadata("description").description

    @property
    def aliases(self) -> list[str]:
        """Return the (optional) list of panel file aliases.

        Raises:
            ValueError: When this panel does not have exactly one source.
                Read ``sources`` instead.
        """
        return self._require_single_metadata("aliases").aliases

    @property
    def archived(self) -> Optional[bool]:
        """Return whether the panel is marked as archived.

        Raises:
            ValueError: When this panel does not have exactly one source.
                Read ``sources`` instead.
        """
        return self._require_single_metadata("archived").archived

    @property
    def marker_source_ids(self) -> pd.Series:
        """Return the panel-source index of each marker."""
        return self._marker_source_ids

    @property
    def hashing_marker_ids(self) -> set[str]:
        """Return marker ids flagged by the ``sample_hashing`` column."""
        return _hashing_marker_ids(self.df)

    @classmethod
    def _parse_header(cls, file: Path) -> AntibodyPanelMetadata:
        """Parse front-matter YAML metadata from a panel file.

        Args:
            file: Panel CSV file whose leading comment block contains YAML metadata.

        Returns:
            Parsed panel metadata.

        Raises:
            ValueError: If no metadata header is present in the file.
        """
        return parse_panel_header_metadata(file)

    @classmethod
    def _parse_panel(cls, panel_file: Path) -> pd.DataFrame:
        """Read the marker table from a panel CSV and convert ``control`` to bool."""
        panel = pd.read_csv(str(panel_file), comment="#", index_col="marker_id").fillna(
            ""
        )

        panel["control"] = panel["control"].map(lambda s: s.lower() == "yes")

        return panel.copy()

    @cached_property
    def markers_control(self) -> List[str]:
        """Return a list of marker control (names)."""
        return list(self._df[self._df["control"]].index)

    @cached_property
    def markers(self) -> List[str]:
        """Return the list of unique markers in the panel."""
        return list(self._df.index.unique())

    @property
    def df(self) -> pd.DataFrame:
        """Return the panel dataframe."""
        return self._df

    @property
    def filename(self) -> Optional[str]:
        """Return the filename of the marker panel."""
        return self._filename

    @property
    def filepath(self) -> Optional[Path]:
        """Return the full path of the marker panel file, if any."""
        return self._filepath

    def copy(self) -> Self:
        """Return a shallow copy with its own marker table and source list."""
        return type(self)(
            self.df.copy(),
            file_name=self.filename,
            filepath=self.filepath,
            sources=list(self.sources),
            marker_source_ids=self.marker_source_ids.copy(),
        )

    def source_as_panel(self, source_index: int) -> Self:
        """Return one source as its own single-source panel."""
        source = self.sources[source_index]
        marker_index = self.marker_source_ids.index[
            self.marker_source_ids == source_index
        ]
        df = self.df.loc[list(marker_index)].copy()
        return type(self)(
            df,
            source.metadata,
            file_name=source.file_name,
            filepath=source.filepath,
        )

    def replace_source(self, source_index: int, replacement: PNAAntibodyPanel) -> Self:
        """Return a copy with one source replaced by a single-source panel.

        An optional column present on only one side is blank on the other,
        the same as an empty cell in a panel CSV.
        """
        if len(replacement.sources) != 1:
            raise ValueError("Replacement panel must come from a single source.")
        keep = self.marker_source_ids.index[self.marker_source_ids != source_index]
        kept = self.df.loc[list(keep)]
        if kept.empty:
            df = replacement.df.copy()
        else:
            kept, incoming = self._align_optional_columns([kept, replacement.df])
            df = pd.concat([kept, incoming])
        df.index.name = self._INDEX_COLUMN
        source_ids = pd.concat(
            [
                self.marker_source_ids.loc[list(keep)],
                pd.Series(source_index, index=replacement.df.index, dtype="int64"),
            ]
        )
        sources = list(self.sources)
        sources[source_index] = replacement.sources[0]
        single_source = len(sources) == 1
        return type(self)(
            df,
            file_name=self.filename if single_source else None,
            filepath=self.filepath if single_source else None,
            sources=sources,
            marker_source_ids=source_ids.astype("int64"),
        )

    @cached_property
    def size(self) -> int:
        """Return the size of the marker panel."""
        return self._df.shape[0]

    @staticmethod
    def _validate_sequences(panel_df, sequence_col):
        """Return errors when sequences differ in length or are not ATCG."""
        errors = []
        sequences = panel_df[sequence_col]
        ref_length = len(sequences.iloc[0])
        if not sequences.apply(lambda x: len(x) == ref_length).all():
            errors.append(f"All {sequence_col} values must have the same length.")

        if not sequences.str.match("^[ATCG]*$").all():
            errors.append(
                f"All {sequence_col} values must only contain ATCG characters. Offending values: "
                f"{sequences[~sequences.str.match('^[ATCG]*$')].tolist()}"
            )

        return errors

    @staticmethod
    def _validate_marker_names(panel_df):
        """Return errors when marker ids contain underscores or whitespace."""
        errors = []
        if any(panel_df.index.str.contains("_")):
            # Markers should not contain underscores since this messes
            # things up with Seurat on the R side
            errors.append(
                "The marker_id column should not contain underscores. "
                "Please use dashes instead. Offending values: "
                f"{panel_df.index[panel_df.index.str.contains('_')]}"
            )
        if any(panel_df.index.str.contains(r"\s")):
            # Markers should not contain white-spaces since this causes
            # issues in the demultiplexing step (and other places that
            # might assume that marker names are single tokens)
            problematic_lines = panel_df.index[panel_df.index.str.contains(r"\s")]
            errors.append(
                "The marker_id column should not contain white-spaces. "
                "Please use dashes instead or remove the white-spaces. Offending values: "
                f"{problematic_lines}"
            )
        return errors

    @classmethod
    def validate_antibody_panel(
        cls, panel_df: pd.DataFrame, validate_types: bool = True
    ) -> list[str]:
        """Validate antibody panel schema and content.

        Args:
            panel_df: Dataframe containing panel markers and sequences.
            validate_types: If True, validate dataframe column types.

        Returns:
            A list of validation error messages. Empty means valid input.
        """
        errors = []

        # some basic sanity check on the panel size and columns
        if not set(cls._REQUIRED_COLUMNS).issubset(set(panel_df.columns)):
            missing_columns = set(cls._REQUIRED_COLUMNS) - set(panel_df.columns)
            errors.append(f"Panel has missing required columns: {missing_columns}")
            return errors

        if validate_types:
            panel_pl_df = pl.from_pandas(panel_df, include_index=True)
            for col, expected_type in (
                cls._REQUIRED_COLUMNS | {cls._INDEX_COLUMN: cls._INDEX_COLUMN_TYPE}
            ).items():
                found_type = panel_pl_df[col].dtype.to_python()
                if not found_type == expected_type:
                    errors.append(
                        f"Column {col} has incorrect type. Expected {expected_type}, got {found_type}"
                    )

        if panel_df.shape[0] == 0:
            errors.append("Panel file is empty")
            return errors

        # sanity check on the unique columns
        for col in cls._UNIQUE_COLUMNS:
            if not len(panel_df[col].unique()) == len(panel_df[col]):
                errors.append(f"All values in column: {col} were not unique")

        if panel_df.index.name != cls._INDEX_COLUMN:
            errors.append(f"`{cls._INDEX_COLUMN}` is missing or is not set as index")
            return errors

        if panel_df.index.duplicated().any():
            duplicated = panel_df.index[panel_df.index.duplicated()].unique().tolist()
            errors.append(
                "All values in column: marker_id were not unique. "
                f"Offending values: {duplicated}"
            )

        errors += cls._validate_marker_names(panel_df)

        if panel_df["control"].dtype != bool:
            errors.append("`control` column is not boolean")

        # Check UniProt IDs format conforming to the UniProt naming convention. Empty IDs are allowed.
        if "uniprot_id" in panel_df.columns:
            # Pattern for valid UniProt IDs
            pattern = r"^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}|$"

            def check_id(id_str):
                """Check id.

                Args:
                    id_str: id str.
                """
                return all(
                    bool(re.match(pattern, id_)) for id_ in str(id_str).split(";")
                )

            bad_ids = panel_df[~panel_df["uniprot_id"].apply(check_id)]["uniprot_id"]

            if len(bad_ids) > 0:
                errors.append(
                    "Invalid UniProt IDs found."
                    "Please conform to the naming convention or remove the following IDs:"
                    f"{bad_ids.tolist()}"
                )

        errors += cls._validate_sequences(panel_df, "sequence_1")
        errors += cls._validate_sequences(panel_df, "sequence_2")
        errors += cls._validate_hashing_marker_ids(panel_df)

        return errors

    @staticmethod
    def _validate_hashing_marker_ids(panel_df: pd.DataFrame) -> list[str]:
        """Return errors when hashing ids lack a ``-<digits>`` suffix or nest."""
        hashing_ids = _hashing_marker_ids(panel_df)
        if not hashing_ids:
            return []
        errors: list[str] = []
        missing_suffix = [
            marker_id
            for marker_id in sorted(hashing_ids)
            if split_hashing_marker_id(marker_id) is None
        ]
        if missing_suffix:
            errors.append(
                "Hashing marker ids must end with -<digits> (e.g. B2M-1). "
                f"Offending values: {missing_suffix}"
            )
        nested = sorted(
            marker_id
            for marker_id in hashing_ids
            if (base := collapsed_hashing_marker_id(marker_id)) != marker_id
            and base in hashing_ids
        )
        if nested:
            errors.append(
                "Hashing marker ids must not collapse to another hashing id "
                f"(e.g. B2M-1-1 next to B2M-1). Offending values: {nested}"
            )
        return errors

    def to_polars(self) -> pl.DataFrame:
        """Convert the panel to a Polars DataFrame."""
        return pl.from_pandas(self.df, include_index=True)

    def _source_marker_groups(
        self,
    ) -> list[tuple[AntibodyPanelMetadata, frozenset[str]]]:
        """Return each source with its marker ids, independent of source order."""
        groups = []
        for source_index, source in enumerate(self.sources):
            marker_ids = frozenset(
                str(marker_id)
                for marker_id in self.marker_source_ids.index[
                    self.marker_source_ids == source_index
                ]
            )
            groups.append((source.metadata, marker_ids))
        return sorted(
            groups,
            key=lambda group: (group[0].model_dump_json(), tuple(sorted(group[1]))),
        )

    def __eq__(self, other: object) -> bool:
        """Return whether two panels describe the same sources and markers.

        Row order, column order, source order, and the file a source was
        loaded from are ignored. Each marker must still belong to the same
        source.

        Args:
            other: Panel to compare for equality.
        """
        if not isinstance(other, PNAAntibodyPanel):
            raise ValueError("Can only compare with another PNAAntibodyPanel")
        if self._source_marker_groups() != other._source_marker_groups():
            return False
        left = self.df.sort_index()
        right = other.df.sort_index()
        if set(left.columns) != set(right.columns):
            return False
        return left.equals(right[list(left.columns)])


def load_antibody_panel(config: PNAConfig, panel: PathType) -> PNAAntibodyPanel:
    """Load an antibody panel from a file or from the config file.

    Args:
        config: the config object
        panel: the path to the panel file or the name of the panel in the config file
    Returns:
        the antibody panel (PNAAntibodyPanel)
    """
    panel_str = str(panel)
    panel_from_config = config.get_panel(panel_str)

    if panel_from_config is not None:
        logger.info("Found panel in config file: %s", panel_from_config.name)
        return panel_from_config

    panel_obj = PNAAntibodyPanel.from_csv(panel)
    return panel_obj


def load_antibody_panels(
    config: PNAConfig, panels: PathType | Sequence[PathType]
) -> PNAAntibodyPanel:
    """Load one or more panels and concatenate them.

    A single path or name is returned as that panel. Several inputs are
    concatenated in order.
    """
    if isinstance(panels, (str, os.PathLike)):
        return load_antibody_panel(config, panels)
    loaded = [load_antibody_panel(config, panel) for panel in panels]
    return PNAAntibodyPanel.concatenate(loaded)


class PNAAntibodyPanelDiff:
    """Class representing the differences between two PNAAntibodyPanel objects."""

    join_on_columns: list[str] = ["sequence_1", "sequence_2"]

    def __init__(self, panel_1: PNAAntibodyPanel, panel_2: PNAAntibodyPanel) -> None:
        """Initialize the PNAAntibodyPanelDiff object.

        Args:
            panel_1: The first panel to compare.
            panel_2: The second panel to compare.

        Raises:
            ValueError: When either panel does not have exactly one source.
        """
        if len(panel_1.sources) != 1 or len(panel_2.sources) != 1:
            raise ValueError(
                "PNAAntibodyPanelDiff only compares panels with a single source. "
                "Split a concatenated panel with source_as_panel first."
            )
        self.panel_1 = panel_1
        self.panel_2 = panel_2

        logger.debug(
            "Comparing panels %s v%s and %s v%s",
            panel_1.name,
            panel_1.version,
            panel_2.name,
            panel_2.version,
        )

        self.joined = self.panel_1.to_polars().join(
            self.panel_2.to_polars(),
            on=self.join_on_columns,
            how="full",
            suffix="_panel_2",
        )

        self._identical_columns: List[str] | None = None
        self._changed_columns: List[str] | None = None
        self._removed_columns: Set[str] | None = None
        self._added_columns: Set[str] | None = None

    @property
    def col_names_in_both_panels(self) -> List[str]:
        """Return a list of column names that are present in both panels."""
        return list(
            set(self.panel_1.to_polars().columns).intersection(
                set(self.panel_2.to_polars().columns)
            )
        )

    @property
    def identical_columns(self) -> List[str]:
        """Return a list of columns that are identical between the two panels."""
        return [
            col_name
            for col_name in self.col_names_in_both_panels
            if self.joined[col_name]
            .eq_missing(self.joined[col_name + "_panel_2"])
            .all()
        ]

    @cached_property
    def changed_columns(self) -> List[str]:
        """Return a list of columns that are different between the two panels."""
        changed_columns = [
            col_name
            for col_name in set(self.col_names_in_both_panels).difference(
                set(self.join_on_columns)
            )
            if not self.joined[col_name]
            .eq_missing(self.joined[col_name + "_panel_2"])
            .all()
        ]
        for col_name in changed_columns:
            diff_count = self.joined.filter(
                pl.col(col_name).ne_missing(pl.col(col_name + "_panel_2"))
            ).shape[0]
            logger.debug(
                "Column %s is different between the two panels %s and %s (%d differing entries).",
                col_name,
                self.panel_1.name,
                self.panel_2.name,
                diff_count,
            )
        return changed_columns

    @cached_property
    def removed_columns(self) -> List[str]:
        """Return a list of columns that are present in panel 1 but not in panel 2."""
        removed_columns = set(self.panel_1.to_polars().columns).difference(
            set(self.panel_2.to_polars().columns)
        )
        for col_name in removed_columns:
            logger.debug(
                "Column %s is present in panel %s but not in panel %s.",
                col_name,
                self.panel_1.name,
                self.panel_2.name,
            )
        return sorted(removed_columns)

    @cached_property
    def added_columns(self) -> List[str]:
        """Return a list of columns that are present in panel 2 but not in panel 1."""
        added_columns = set(self.panel_2.to_polars().columns).difference(
            set(self.panel_1.to_polars().columns)
        )
        for col_name in added_columns:
            logger.debug(
                "Column %s is present in panel %s but not in panel %s.",
                col_name,
                self.panel_2.name,
                self.panel_1.name,
            )
        return sorted(added_columns)

    @property
    def added_clones(self) -> pl.DataFrame:
        """Return a dataframe with the clones that are present in panel 2 but not in panel 1."""
        return (
            self.joined.filter(
                pl.any_horizontal(
                    pl.col(col_name).is_null()
                    & pl.col(col_name + "_panel_2").is_not_null()
                    for col_name in self.join_on_columns
                )
            )
            .drop([col_name for col_name in self.panel_1.to_polars().columns])
            .rename(
                {
                    col_name + "_panel_2": col_name
                    for col_name in self.panel_2.to_polars().columns
                    if col_name + "_panel_2" in self.joined.columns
                }
            )
        )

    @property
    def removed_clones(self) -> pl.DataFrame:
        """Return a dataframe with the clones that are present in panel 1 but not in panel 2."""
        return self.joined.filter(
            pl.any_horizontal(
                pl.col(col_name).is_not_null() & pl.col(col_name + "_panel_2").is_null()
                for col_name in self.join_on_columns
            )
        ).drop(
            [
                col_name + "_panel_2"
                if col_name in self.joined.columns
                and col_name not in self.added_columns
                else col_name
                for col_name in self.panel_2.to_polars().columns
            ]
        )

    def changed_marker_ids(self) -> dict[str, str]:
        """Return ``old marker_id -> new marker_id`` for markers whose id changed."""
        if (
            "marker_id" not in self.joined.columns
            or "marker_id_panel_2" not in self.joined.columns
        ):
            return {}
        both = self.joined.filter(
            pl.col("marker_id").is_not_null()
            & pl.col("marker_id_panel_2").is_not_null()
        )
        mapping: dict[str, str] = {}
        for old, new in zip(
            both["marker_id"].to_list(), both["marker_id_panel_2"].to_list()
        ):
            if str(old) != str(new):
                mapping[str(old)] = str(new)
        return mapping


def sample_calling_hashing_collapsed(
    hashing_ids: set[str],
    *,
    adata: AnnData | None = None,
    pxl_file_metadata: dict | None = None,
) -> bool:
    """Return whether sample calling has already collapsed hashing clones.

    Explicit ``hashing_collapsed`` on the pixel file metadata wins. Older
    files are inferred from ``original_hash_counts_*`` columns or from hashing
    clones that are absent from ``var``. Without an AnnData and without that
    key, there is nothing to infer from, so this returns False.
    """
    if pxl_file_metadata is not None and "hashing_collapsed" in pxl_file_metadata:
        return bool(pxl_file_metadata["hashing_collapsed"])
    if adata is None:
        return False
    if any(str(col).startswith("original_hash_counts_") for col in adata.obs.columns):
        return True
    if not hashing_ids:
        return False
    var_names = {str(name) for name in adata.var_names}
    return hashing_ids.isdisjoint(var_names)


def align_panel_patches(
    panels: list[PNAAntibodyPanel],
    adatas: list[AnnData] | None = None,
    *,
    pxl_file_metadata: list[dict] | None = None,
) -> tuple[list[PNAAntibodyPanel], list[dict[str, str]], list[dict[str, str]]]:
    """Bump each source to the newest patch carried by another panel.

    Sources match on name and product, and only when major and minor versions
    agree. A source that is not already on a panel is left alone.
    ``pxl_file_metadata`` is the pixel file metadata for each panel, in the same
    order, and carries ``hashing_collapsed`` when sample calling wrote the file.

    Returns:
        Updated panel copies, marker renames to apply to stored data (var,
        edgelist, proximity, layouts), and hashing-clone renames for
        ``original_hash_counts_*`` columns. Renames are keyed by input order.
    """
    updated = [panel.copy() for panel in panels]
    data_renames: list[dict[str, str]] = [{} for _ in panels]
    hash_renames: list[dict[str, str]] = [{} for _ in panels]

    families: dict[tuple[str, str, tuple[int, ...]], list[tuple[int, int, Version]]] = (
        defaultdict(list)
    )
    for panel_index, panel in enumerate(updated):
        for source_index, source in enumerate(panel.sources):
            identity = _source_identity(source)
            if identity is None:
                continue
            version = Version(source.metadata.version)
            minor = version.release[:2]
            families[(*identity, minor)].append((panel_index, source_index, version))

    for members in families.values():
        latest_idx, latest_source_idx, latest_version = max(
            members, key=lambda member: member[2]
        )
        latest_panel = updated[latest_idx].source_as_panel(latest_source_idx)
        for panel_index, source_index, version in members:
            if version == latest_version:
                continue
            current = updated[panel_index].source_as_panel(source_index)
            logger.info(
                "Upgrading panel source %s %s from %s to %s.",
                current.name,
                current.product,
                current.version,
                latest_panel.version,
            )
            diff = PNAAntibodyPanelDiff(current, latest_panel)
            clone_map = diff.changed_marker_ids()
            _validate_hashing_renames(current, latest_panel, clone_map)
            hashing_ids = current.hashing_marker_ids
            pxl_metadata = (
                None if pxl_file_metadata is None else pxl_file_metadata[panel_index]
            )
            collapsed = sample_calling_hashing_collapsed(
                hashing_ids,
                adata=None if adatas is None else adatas[panel_index],
                pxl_file_metadata=pxl_metadata,
            )
            if adatas is not None:
                _require_expected_markers(
                    adata=adatas[panel_index],
                    old_panel=current,
                    new_panel=latest_panel,
                    clone_map=clone_map,
                    collapsed=collapsed,
                )
            data_map = _data_rename_map(clone_map, hashing_ids, collapsed=collapsed)
            _merge_renames(data_renames[panel_index], data_map)
            _merge_renames(
                hash_renames[panel_index],
                {old: new for old, new in clone_map.items() if old in hashing_ids},
            )
            updated[panel_index] = updated[panel_index].replace_source(
                source_index, latest_panel
            )

    return updated, data_renames, hash_renames


def aligned_dataset_panel(panels: list[PNAAntibodyPanel]) -> PNAAntibodyPanel:
    """Return one panel after per-source patch alignment across files.

    Several files that describe the same sources collapse to a single panel.
    The order of ``--panel`` inputs does not have to match.
    """
    if len(panels) == 1:
        return panels[0]
    updated, _, _ = align_panel_patches(panels)
    first = updated[0]
    for other in updated[1:]:
        if first != other:
            raise ValueError(
                "Samples do not share the same panel sources after patch alignment."
            )
    return first


def _source_identity(source: PanelSource) -> tuple[str, str] | None:
    """Return ``(name, product)`` for a source, or None when product is unset."""
    product = source.metadata.product
    if not product:
        return None
    return (source.metadata.name, product)


def _data_rename_map(
    clone_map: dict[str, str], hashing_ids: set[str], *, collapsed: bool
) -> dict[str, str]:
    """Return marker renames, collapsing hashing clones when ``collapsed`` is set."""
    if not collapsed:
        return dict(clone_map)
    data_map = {old: new for old, new in clone_map.items() if old not in hashing_ids}
    families: dict[str, str] = {}
    for old, new in clone_map.items():
        if old not in hashing_ids:
            continue
        old_base = collapsed_hashing_marker_id(old)
        new_base = collapsed_hashing_marker_id(new)
        if old_base in families and families[old_base] != new_base:
            raise ValueError(
                f"Hashing markers with collapsed name {old_base!r} do not share "
                "a single new base name."
            )
        families[old_base] = new_base
    for old_base, new_base in families.items():
        if old_base != new_base:
            data_map[old_base] = new_base
    return data_map


def _validate_hashing_renames(
    old_panel: PNAAntibodyPanel,
    new_panel: PNAAntibodyPanel,
    clone_map: dict[str, str],
) -> None:
    """Reject a hashing rename that breaks the collapsed-name rules.

    Each of these fails:

    * ``B2M-1`` → ``C2M``: a hashing id must end with ``-<digits>``.
    * ``B2M-1`` → ``C2M-2``: the numeric suffix stays.
    * ``B2M-1`` → ``C2M-1`` and ``B2M-2`` → ``D2M-2``: clones in one family
      share one new base.
    * ``B2M`` stays while ``B2M-1`` → ``C2M-1``, or ``B2M`` → ``C2M`` while
      ``B2M-1`` stays: a non-hashing marker that already has the collapsed
      name renames with the family.
    * ``CD19`` → ``C2M`` while ``B2M-1`` → ``C2M-1``: the family must not
      land on a different non-hashing marker.
    """
    old_hashing = old_panel.hashing_marker_ids
    new_hashing = new_panel.hashing_marker_ids
    if not old_hashing:
        return

    families: dict[str, str] = {}
    for old in sorted(old_hashing):
        new = clone_map.get(old, old)
        if new not in new_hashing and old not in clone_map:
            continue
        old_parts = split_hashing_marker_id(old)
        new_parts = split_hashing_marker_id(new)
        if old_parts is None or new_parts is None:
            raise ValueError(
                "Hashing marker ids must end with -<digits> to be renamed "
                f"({old!r} -> {new!r})."
            )
        if old_parts[1] != new_parts[1]:
            raise ValueError(
                "Hashing marker rename may only change the base name, not the "
                f"hash group suffix {old_parts[1]}. Got {old!r} -> {new!r}."
            )
        old_base = collapsed_hashing_marker_id(old)
        new_base = collapsed_hashing_marker_id(new)
        if old_base in families and families[old_base] != new_base:
            raise ValueError(
                f"Hashing markers with collapsed name {old_base!r} must keep a "
                "single base name per hash group family."
            )
        families[old_base] = new_base

    non_hashing = {
        str(marker_id)
        for marker_id in old_panel.markers
        if str(marker_id) not in old_hashing
    }
    for old_base, new_base in families.items():
        if old_base in non_hashing:
            moved = clone_map.get(old_base, old_base)
            if moved != new_base:
                # TODO: we should think about if we want this check or not in the long run.
                # adding it here now we can always remove it later if we decide it's too strict.
                raise ValueError(
                    f"Hashing collapsed base {old_base!r} and non-hashing marker "
                    f"{old_base!r} must be renamed together "
                    f"(hashing maps to {new_base!r}, non-hashing maps to {moved!r})."
                )
        for other in non_hashing:
            if other == old_base:
                continue
            if clone_map.get(other, other) == new_base:
                raise ValueError(
                    f"Hashing collapsed base {old_base!r} -> {new_base!r} collides "
                    f"with non-hashing marker {other!r}."
                )


def _require_expected_markers(
    *,
    adata: AnnData,
    old_panel: PNAAntibodyPanel,
    new_panel: PNAAntibodyPanel,
    clone_map: dict[str, str],
    collapsed: bool,
) -> None:
    """Raise when a patch bump is missing a marker the file should still contain.

    After sample calling, hashing clones may be absent when the file is
    collapsed. A missing non-hashing marker still fails.
    """
    var_names = {str(name) for name in adata.var_names}
    old_hashing = old_panel.hashing_marker_ids
    new_hashing = new_panel.hashing_marker_ids
    old_markers = {str(marker_id) for marker_id in old_panel.markers}
    missing: list[str] = []
    for marker_id in old_panel.markers:
        marker = str(marker_id)
        if marker in var_names or marker in clone_map:
            continue
        if collapsed and marker in old_hashing:
            continue
        missing.append(marker)
    for marker_id in new_panel.markers:
        marker = str(marker_id)
        if marker in var_names or marker in old_markers:
            continue
        old_ids = [old for old, new in clone_map.items() if new == marker]
        if any(old in var_names for old in old_ids):
            continue
        if collapsed and (
            marker in new_hashing or any(old in old_hashing for old in old_ids)
        ):
            continue
        missing.append(marker)
    if missing:
        raise ValueError(
            "Row count mismatch in automatic panel patch version bump. "
            f"Missing markers: {sorted(set(missing))[:5]}"
        )


def _merge_renames(target: dict[str, str], extra: dict[str, str]) -> None:
    """Copy renames into ``target``, rejecting two destinations for one marker."""
    for old, new in extra.items():
        if old in target and target[old] != new:
            raise ValueError(
                f"Marker {old!r} is renamed to both {target[old]!r} and {new!r}."
            )
        if old != new:
            target[old] = new
