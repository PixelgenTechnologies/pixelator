"""PNA antibody panel loading.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import pandas as pd
import polars as pl

from pixelator.common.config.panel import (
    AntibodyPanelMetadata,
    parse_panel_header_metadata,
)
from pixelator.common.types import PathType
from pixelator.common.utils import logger
from pixelator.pna.config.panel.hashing import _hashing_marker_ids
from pixelator.pna.config.panel.validation import (
    _INDEX_COLUMN,
    _INDEX_COLUMN_TYPE,
    _REQUIRED_COLUMNS,
    _UNIQUE_COLUMNS,
    validate_antibody_panel,
)

if TYPE_CHECKING:
    from pixelator.pna.config.config_class import PNAConfig


@dataclass(frozen=True)
class PanelSource:
    """One panel file that contributed markers to a ``PNAAntibodyPanel``.

    ``columns`` is the marker columns this source was loaded with, in that
    order. It is None when the file did not store them.
    """

    metadata: AntibodyPanelMetadata
    file_name: str | None = None
    filepath: str | None = None
    columns: Sequence[str] | None = None


class PNAAntibodyPanel:
    """Class representing a PNA antibody panel."""

    _INDEX_COLUMN = _INDEX_COLUMN
    _INDEX_COLUMN_TYPE = _INDEX_COLUMN_TYPE
    _REQUIRED_COLUMNS = _REQUIRED_COLUMNS
    _UNIQUE_COLUMNS = _UNIQUE_COLUMNS

    def __init__(
        self,
        df: pd.DataFrame,
        sources: list[PanelSource],
        marker_source_ids: pd.Series,
        file_name: Optional[str] = None,
        filepath: Optional[PathType] = None,
    ) -> None:
        """Build a panel from a marker table and the sources that contributed it.

        Args:
            df: Marker table, indexed by marker id.
            sources: Panel files that contributed markers, in source-index order.
            marker_source_ids: Source index for each marker.
            file_name: Basename of the file this panel was loaded from.
            filepath: Full path of the file this panel was loaded from.

        Raises:
            AssertionError: If the marker table fails panel validation.
        """
        self._filename = file_name
        self._filepath = Path(filepath).resolve() if filepath else None
        self._df = df
        self.sources = list(sources)
        self._marker_source_ids = marker_source_ids
        errors = self.validate_antibody_panel(df)
        if len(errors) > 0:
            msg_str = "\n".join(errors)
            raise AssertionError(
                f"The following errors were found validating the panel: {msg_str}"
            )

    @classmethod
    def from_metadata(
        cls,
        df: pd.DataFrame,
        metadata: AntibodyPanelMetadata,
        file_name: Optional[str] = None,
        filepath: Optional[PathType] = None,
    ) -> Self:
        """Build a single-source panel from a marker table and its metadata.

        Args:
            df: Marker table, indexed by marker id.
            metadata: Metadata for this panel file.
            file_name: Basename of the file this panel was loaded from.
            filepath: Full path of the file this panel was loaded from.

        Raises:
            AssertionError: If the marker table fails panel validation.
        """
        resolved = Path(filepath).resolve() if filepath else None
        return cls(
            df,
            [
                PanelSource(
                    metadata=metadata,
                    file_name=file_name,
                    filepath=str(resolved) if resolved else None,
                    columns=tuple(map(str, df.columns)),
                )
            ],
            pd.Series(0, index=df.index, dtype="int64"),
            file_name=file_name,
            filepath=resolved,
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

        return cls.from_metadata(
            df, metadata, file_name=panel_file.name, filepath=panel_file
        )

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
        # Sample calling can leave a collapsed hashing id in var. That row is
        # not a panel marker, so its panel columns are empty.
        if "sequence_1" in df.columns:
            df = df[df["sequence_1"].notna()].copy()
        if "control" in df.columns and df["control"].dtype != bool:
            df["control"] = df["control"].fillna(False).astype(bool)
        metadata = AntibodyPanelMetadata.model_validate(panel_metadata)
        return cls.from_metadata(df, metadata, file_name=file_name)

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
            panel_sources,
            source_ids.astype("int64"),
            file_name=file_name,
            filepath=filepath,
        )

    @classmethod
    def concatenate(cls, panels: Sequence[PNAAntibodyPanel]) -> PNAAntibodyPanel:
        """Concatenate panels into one panel.

        One panel is returned unchanged. Several panels are stacked in the
        given order. The result keeps every input source.
        ``marker_id``, ``sequence_1``, and ``sequence_2`` must be unique
        across the concatenation. An optional column present on only some
        sources is blank on the others, the same as an empty cell in a panel CSV.
        Each source still remembers the columns it was loaded with.
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
        return cls(df, sources, marker_source_ids.astype("int64"))

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
            list(self.sources),
            self.marker_source_ids.copy(),
            file_name=self.filename,
            filepath=self.filepath,
        )

    def source_as_panel(self, source_index: int) -> Self:
        """Return one source as its own single-source panel.

        Columns added because another source had them are left behind. A
        source that was stored without its column list keeps every column.
        """
        if source_index < 0:
            raise ValueError("Source index must be zero or greater.")
        source = self.sources[source_index]
        marker_index = self.marker_source_ids.index[
            self.marker_source_ids == source_index
        ]
        df = self.df.loc[list(marker_index)].copy()
        if source.columns is not None:
            df = df.loc[:, list(source.columns)]
        return type(self).from_metadata(
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
        if source_index < 0:
            raise ValueError("Source index must be zero or greater.")
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
            sources,
            source_ids.astype("int64"),
            file_name=self.filename if single_source else None,
            filepath=self.filepath if single_source else None,
        )

    @cached_property
    def size(self) -> int:
        """Return the size of the marker panel."""
        return self._df.shape[0]

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
        return validate_antibody_panel(panel_df, validate_types=validate_types)

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
