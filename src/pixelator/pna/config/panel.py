"""Marker panel management for different PNA assays.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Set

from anndata import AnnData

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import re
from io import StringIO

import numpy as np
import pandas as pd
import polars as pl

from pixelator.common.config.panel import AntibodyPanelMetadata, PanelType
from pixelator.common.types import PathType
from pixelator.common.utils import logger

if TYPE_CHECKING:
    from pixelator.pna.config.config_class import PNAConfig
    from pixelator.pna.pixeldataset.dataset import PNAPixelDataset


class PNAPanel(ABC):
    """Shared read interface for single PNA panels and panel combinations.

    Subclasses must provide :attr:`df` (and typically :meth:`from_adata`).
    Marker helpers, size, Polars conversion, and schema validation are
    implemented here in terms of that dataframe.
    """

    _INDEX_COLUMN = "marker_id"
    _INDEX_COLUMN_TYPE = str
    _REQUIRED_COLUMNS = {
        "control": bool,
        "sequence_1": str,
        "sequence_2": str,
    }
    _UNIQUE_COLUMNS = ["sequence_1", "sequence_2"]

    @property
    @abstractmethod
    def df(self) -> pd.DataFrame:
        """Return the panel marker dataframe indexed by ``marker_id``."""

    @property
    @abstractmethod
    def metadata(self) -> AntibodyPanelMetadata | list[AntibodyPanelMetadata]:
        """Return panel metadata (one entry, or one per member for combinations)."""

    @classmethod
    @abstractmethod
    def from_adata(
        cls,
        adata: AnnData,
        file_name: Optional[str] = None,
        filepath: Optional[PathType] = None,
    ) -> Self:
        """Create a panel instance from AnnData-embedded panel data.

        Args:
            adata: AnnData with embedded panel information.
            file_name: Optional basename of the source file.
            filepath: Optional full path of the source file.
        """

    @classmethod
    def from_pxl_dataset(
        cls,
        pxl_data: PNAPixelDataset,
        file_name: Optional[str] = None,
        filepath: Optional[PathType] = None,
    ) -> Self:
        """Create a panel from a PNA pixel dataset via :meth:`from_adata`.

        When ``file_name`` / ``filepath`` are omitted and ``pxl_data`` wraps a
        single ``.pxl`` file, those values are taken from that file path.

        Args:
            pxl_data: Dataset whose ``adata()`` holds panel information.
            file_name: Optional source file basename to attach to the panel.
            filepath: Optional full source path to attach to the panel.

        Returns:
            A panel of this class.
        """
        logger.debug("Creating panel from PNAPixelDataset object")
        file_name, filepath = _resolve_panel_source_from_pxl(
            pxl_data, file_name=file_name, filepath=filepath
        )
        panel = cls.from_adata(pxl_data.adata(), file_name=file_name, filepath=filepath)
        logger.debug("Panel from PNAPixelDataset created")
        return panel

    @cached_property
    def markers_control(self) -> List[str]:
        """Return a list of marker control (names)."""
        return list(self.df[self.df["control"]].index)

    @cached_property
    def markers(self) -> List[str]:
        """Return the list of unique markers in the panel."""
        return list(self.df.index.unique())

    @cached_property
    def size(self) -> int:
        """Return the size of the marker panel."""
        return self.df.shape[0]

    def to_polars(self) -> pl.DataFrame:
        """Convert the panel to a Polars DataFrame."""
        return pl.from_pandas(self.df, include_index=True)

    @staticmethod
    def _validate_sequences(panel_df, sequence_col):
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
                        f"Column {col} has incorrect type. "
                        + f"Expected {expected_type}, got {found_type}"
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

        errors += cls._validate_marker_names(panel_df)

        if panel_df["control"].dtype != bool:
            errors.append("`control` column is not boolean")

        # Check UniProt IDs format conforming to the UniProt naming convention.
        # Empty IDs are allowed.
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

        return errors


class PartialPNAAntibodyPanel(PNAPanel):
    """A single PNA antibody panel loaded from a CSV (or equivalent source).

    This is the base type for typed panels. Concrete subclasses bind a fixed
    :class:`~pixelator.common.config.panel.PanelType`:

    * :class:`~pixelator.pna.config.panel.PNABasePanel` — core markers
    * :class:`~pixelator.pna.config.panel.PNAAddonPanel` — addon markers used
      with a base panel
    * :class:`~pixelator.pna.config.panel.PNASampleHashingPanel` —
      sample-hashing markers

    Prefer the module-level helpers
    :func:`~pixelator.pna.config.panel.panel_from_csv`,
    :func:`~pixelator.pna.config.panel.panel_from_adata`, and
    :func:`~pixelator.pna.config.panel.load_antibody_panel` when the concrete
    type should follow metadata. Use
    :class:`~pixelator.pna.config.panel.PNAAntibodyPanelCombination` when
    several panels are used together in one sample.
    """

    _panel_type: PanelType = PanelType.PARTIAL

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: AntibodyPanelMetadata,
        file_name: Optional[str] = None,
        filepath: Optional[PathType] = None,
    ) -> None:
        """Load a panel from a dataframe and metadata.

        Args:
            df: Panel rows indexed by ``marker_id`` with required columns.
            metadata: Panel YAML metadata. Missing ``panel_type`` is defaulted
                to :attr:`~pixelator.common.config.panel.PanelType.PARTIAL` for
                legacy files when constructing this base class.
            file_name: Optional basename of the source file.
            filepath: Optional full path of the source file.

        Raises:
            ValueError: If ``metadata`` is ``None`` or its ``panel_type`` does
                not match this class.
            AssertionError: If the panel dataframe fails validation.
        """
        self._filename = file_name

        if metadata is None:
            raise ValueError("Panel metadata cannot be None")
        if (
            self.__class__._panel_type is PanelType.PARTIAL
            and metadata.panel_type is None
        ):
            # if panel type is missing from metadata use partial as default legacy behavior
            metadata.panel_type = PanelType.PARTIAL
        elif metadata.panel_type != self.__class__._panel_type:
            raise ValueError(
                f"Panel metadata panel_type {metadata.panel_type!r} does not match "
                + f"{self.__class__.__name__} (expected {self.__class__._panel_type.value})."
            )
        self._filepath: Optional[Path] = Path(filepath).resolve() if filepath else None
        self._metadata = metadata

        self._df = df

        # validate the panel
        errors = self.validate_antibody_panel(df)
        if len(errors) > 0:
            msg_str = "\n".join(errors)
            raise AssertionError(
                f"The following errors were found validating the panel: {msg_str}"
            )

    @property
    def metadata(self) -> AntibodyPanelMetadata:
        """Return the panel metadata."""
        return self._metadata

    @classmethod
    def from_csv(cls, filename: PathType) -> Self:
        """Create a panel instance from a CSV panel file.

        Does not dispatch on ``panel_type``; the caller must use the matching
        class (or :func:`~pixelator.pna.config.panel.panel_from_csv` for
        automatic dispatch).

        Args:
            filename: Path to a ``.csv`` panel file with YAML front-matter.

        Returns:
            A panel of this class.

        Raises:
            AssertionError: If the file is missing, not a ``.csv``, or fails
                validation.
            ValueError: If metadata ``panel_type`` is incompatible with this
                class.
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
    def from_adata(
        cls,
        adata: AnnData,
        file_name: Optional[str] = None,
        filepath: Optional[PathType] = None,
    ) -> Self:
        """Create a panel from legacy single-panel AnnData ``uns`` / ``var`` data.

        Expects ``adata.uns["panel_metadata"]`` with ``panel_columns`` naming
        the ``adata.var`` columns that form the panel table. For multi-panel
        AnnData, use
        :meth:`~pixelator.pna.config.panel.PNAAntibodyPanelCombination.from_adata`
        or :func:`~pixelator.pna.config.panel.panel_from_adata`.

        Args:
            adata: AnnData with embedded panel metadata and marker columns.
            file_name: Optional basename of the source file.
            filepath: Optional full path of the source file.

        Returns:
            A panel of this class.

        Raises:
            KeyError: If panel metadata or ``panel_columns`` are missing.
        """
        logger.debug("Creating Antibody panel from AnnData object")
        try:
            panel_metadata = adata.uns["panel_metadata"]
        except KeyError as err:
            logger.error(  # pylint: disable=logging-not-lazy
                f"The provided AnnData object does not contain {err}. "
                + "Please, regenerate your data with the most recent version of pixelator."
            )
            raise
        panel_columns = panel_metadata.get("panel_columns")
        if not panel_columns:
            raise KeyError(
                "The provided AnnData object does not contain panel columns information in the "
                + "metadata. Please, regenerate your data with the most recent version of "
                + "pixelator."
            )
        df = adata.var[panel_columns]
        metadata = AntibodyPanelMetadata.model_validate(panel_metadata)

        logger.debug("Antibody panel from AnnData object created")
        return cls(df, metadata, file_name=file_name, filepath=filepath)

    @property
    def name(self) -> str:
        """Panel name from metadata.

        Returns:
            The panel name.
        """
        return self.metadata.name

    @property
    def product(self) -> Optional[str]:
        """Product identifier from metadata, if present.

        Returns:
            Product name, or None when not provided in panel metadata.
        """
        return self.metadata.product

    @property
    def version(self) -> str:
        """Panel version from metadata.

        Returns:
            Semantic version string for this panel.
        """
        return self.metadata.version

    @property
    def description(self) -> Optional[str]:
        """Return the panel file description."""
        return self.metadata.description

    @property
    def aliases(self) -> list[str]:
        """Return the (optional) list of panel file aliases."""
        return self.metadata.aliases

    @property
    def archived(self) -> Optional[bool]:
        """Return whether the panel is marked as archived."""
        return self.metadata.archived

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
        return AntibodyPanelMetadata.from_panel_csv(file)

    @classmethod
    def _parse_panel(cls, panel_file: Path) -> pd.DataFrame:
        panel = pd.read_csv(str(panel_file), comment="#", index_col="marker_id").fillna(
            ""
        )

        panel["control"] = panel["control"].map(lambda s: s.lower() == "yes")
        if "sample_hashing" in panel.columns:
            panel["sample_hashing"] = panel["sample_hashing"].map(
                lambda s: s.lower() == "yes"
            )

        return panel.copy()

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

    def __eq__(self, other: object) -> bool:
        """Check if two panels are equal based on dataframe and metadata.

        Args:
            other: Panel to compare for equality.
        """
        if not isinstance(other, PNAPanel):
            raise ValueError("Can only compare with another PNAPanel")
        return self.df.equals(other.df) and self.metadata == other.metadata


def sample_hashing_mask(sample_hashing: pd.Series) -> pd.Series:
    """Normalize a ``sample_hashing`` column to a boolean mask.

    Panel CSV parsing converts ``yes``/``no`` to bool; AnnData / Polars
    round-trips may still expose strings (``yes``/``no`` or ``True``/``False``).

    Args:
        sample_hashing: Column values from a panel dataframe.

    Returns:
        Boolean series aligned to ``sample_hashing`` (``True`` = hashing marker).
    """
    if pd.api.types.is_bool_dtype(sample_hashing):
        return sample_hashing.fillna(False)
    normalized = sample_hashing.astype(str).str.strip().str.lower()
    return normalized.isin(["yes", "true", "1"])


def get_panel_type_from_metadata(
    metadata: AntibodyPanelMetadata,
) -> type[PartialPNAAntibodyPanel]:
    """Map panel metadata to the concrete panel class to instantiate.

    Args:
        metadata: Panel metadata, typically from CSV front-matter.

    Returns:
        :class:`~pixelator.pna.config.panel.PartialPNAAntibodyPanel`,
        :class:`~pixelator.pna.config.panel.PNABasePanel`,
        :class:`~pixelator.pna.config.panel.PNAAddonPanel`, or
        :class:`~pixelator.pna.config.panel.PNASampleHashingPanel`. Missing
        or unknown ``panel_type`` values fall back to
        :class:`~pixelator.pna.config.panel.PartialPNAAntibodyPanel` (with a
        warning).
    """
    match metadata.panel_type:
        case PanelType.PARTIAL:
            return PartialPNAAntibodyPanel
        case PanelType.BASE:
            return PNABasePanel
        case PanelType.ADDON:
            return PNAAddonPanel
        case PanelType.SAMPLE_HASHING:
            return PNASampleHashingPanel
        case None:
            logger.debug(
                "Panel metadata has no panel_type. "
                + "Falling back to generic PartialPNAAntibodyPanel.",
            )
            return PartialPNAAntibodyPanel
        case _:
            logger.debug(
                "Unknown panel type %s in panel metadata. "
                + "Falling back to generic PartialPNAAntibodyPanel.",
                metadata.panel_type,
            )
            return PartialPNAAntibodyPanel


def _resolve_panel_source_from_pxl(
    pxl_data: PNAPixelDataset,
    file_name: Optional[str] = None,
    filepath: Optional[PathType] = None,
) -> tuple[Optional[str], Optional[PathType]]:
    """Fill missing file_name/filepath from a single-file PNAPixelDataset."""
    if file_name is not None and filepath is not None:
        return file_name, filepath

    file_mapping = getattr(pxl_data.view, "_db_to_file_mapping", None) or {}
    if len(file_mapping) == 1:
        source_path = Path(next(iter(file_mapping.values())).path)
        if filepath is None:
            filepath = source_path
        if file_name is None:
            file_name = source_path.name
    return file_name, filepath


def panel_from_adata(
    adata: AnnData,
    file_name: Optional[str] = None,
    filepath: Optional[PathType] = None,
) -> PNAPanel:
    """Create a panel (or combination) from AnnData panel metadata.

    If ``adata.uns`` contains ``num_partial_panels``, returns a
    :class:`~pixelator.pna.config.panel.PNAAntibodyPanelCombination`.
    Otherwise returns a single typed panel whose class follows the stored
    ``panel_type``.

    Args:
        adata: AnnData with panel metadata in ``uns`` (and marker columns in
            ``var`` for single-panel data).
        file_name: Optional basename of the source file.
        filepath: Optional full path of the source file.

    Returns:
        A :class:`~pixelator.pna.config.panel.PNAPanel` — typically a typed
        :class:`~pixelator.pna.config.panel.PartialPNAAntibodyPanel`
        subclass, or a combination when multiple partial panels are stored.
    """
    if "num_partial_panels" in adata.uns:
        return PNAAntibodyPanelCombination.from_adata(
            adata, file_name=file_name, filepath=filepath
        )
    panel_type = get_panel_type_from_metadata(
        AntibodyPanelMetadata.from_adata(adata)[0]
    )
    return panel_type.from_adata(adata, file_name=file_name, filepath=filepath)


def panel_from_pxl_dataset(
    pxl_data: PNAPixelDataset,
    file_name: Optional[str] = None,
    filepath: Optional[PathType] = None,
) -> PNAPanel:
    """Create a panel (or combination) from a PNA pixel dataset.

    Equivalent to :func:`~pixelator.pna.config.panel.panel_from_adata` on
    ``pxl_data.adata()``. When ``file_name`` / ``filepath`` are omitted and
    ``pxl_data`` wraps a single ``.pxl`` file, those values are taken from
    that file path.

    Args:
        pxl_data: Dataset that embeds panel information in its AnnData.
        file_name: Optional basename of the source file.
        filepath: Optional full path of the source file.

    Returns:
        A typed single panel or
        :class:`~pixelator.pna.config.panel.PNAAntibodyPanelCombination`.
    """
    file_name, filepath = _resolve_panel_source_from_pxl(
        pxl_data, file_name=file_name, filepath=filepath
    )
    return panel_from_adata(pxl_data.adata(), file_name=file_name, filepath=filepath)


def panel_from_csv(panel_file: PathType) -> PartialPNAAntibodyPanel:
    """Create a typed panel from a CSV file, dispatching on metadata ``panel_type``.

    Prefer this over
    :meth:`~pixelator.pna.config.panel.PartialPNAAntibodyPanel.from_csv` when
    the concrete class should follow the file header.

    Args:
        panel_file: Path to a ``.csv`` panel file with YAML front-matter.

    Returns:
        An instance of
        :class:`~pixelator.pna.config.panel.PartialPNAAntibodyPanel` or a
        typed subclass.

    Raises:
        AssertionError: If the file is missing or not a ``.csv``.
    """
    panel_file = Path(panel_file)
    if not panel_file.is_file() or panel_file.suffix != ".csv":
        raise AssertionError(
            f"Panel file {panel_file} not found or has an incorrect format"
        )

    metadata = AntibodyPanelMetadata.from_panel_csv(panel_file)
    panel_type = get_panel_type_from_metadata(metadata)
    return panel_type.from_csv(panel_file)


def load_antibody_panel(
    config: PNAConfig, requested_panels: PathType | list[PathType] | list[str]
) -> PNAAntibodyPanelCombination:
    """Load one or more panels from config names and/or CSV paths.

    Each entry is resolved from ``config`` when possible, otherwise loaded with
    :func:`~pixelator.pna.config.panel.panel_from_csv`. All resolved panels
    are returned as a
    :class:`~pixelator.pna.config.panel.PNAAntibodyPanelCombination`
    (including when only one panel is requested).

    Args:
        config: PNA configuration that may already contain named panels.
        requested_panels: Panel path(s) and/or config panel name(s).

    Returns:
        Combination containing the loaded panel(s).

    Raises:
        ValueError: If the resulting panel list is empty or panels conflict.
        AssertionError: If a CSV path cannot be loaded.
    """
    return_panels = []
    for panel in (
        requested_panels if isinstance(requested_panels, list) else [requested_panels]
    ):
        panel_str = str(panel)
        logger.debug("Loading panel %s", panel_str)
        panel_from_config = config.get_panel(panel_str)

        if panel_from_config is not None:
            logger.info("Found panel in config file: %s", panel_from_config.name)
            return_panels.append(panel_from_config)
            continue

        panel_obj = panel_from_csv(panel)
        logger.info(
            "Loaded %s %s from CSV file: %s",
            panel_obj.__class__.__name__,
            panel_obj.name,
            panel_obj.filename,
        )
        return_panels.append(panel_obj)
    return PNAAntibodyPanelCombination(return_panels)


class PNAAntibodyPanelDiff:
    """Compare two :class:`~pixelator.pna.config.panel.PartialPNAAntibodyPanel` instances by clone sequences.

    Used to describe column-level differences between panel versions and to
    support automatic AnnData panel patch upgrades via :meth:`upgrade_adata`.
    """

    join_on_columns: list[str] = ["sequence_1", "sequence_2"]

    def __init__(
        self, panel_1: PartialPNAAntibodyPanel, panel_2: PartialPNAAntibodyPanel
    ) -> None:
        """Initialize the PNAAntibodyPanelDiff object.

        Args:
            panel_1: The first panel to compare.
            panel_2: The second panel to compare.
        """
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

    def upgrade_adata(self, adata: AnnData) -> AnnData:
        """Apply a patch-level panel upgrade to AnnData marker annotations.

        Updates overlapping clone rows from ``panel_2`` when the two panels
        share the same product and only patch-compatible differences exist.

        Args:
            adata: AnnData whose ``var`` table embeds panel columns.

        Returns:
            The same AnnData instance after in-place panel column updates.

        Raises:
            ValueError: If clones were added/removed, products differ, panels
                do not match the AnnData contents, or row counts diverge.
        """
        if len(self.added_clones) > 0:
            raise ValueError(
                "Cannot automatically upgrade panel if there are added clones. "
                + "Please check the differences between the panels and upgrade manually."
            )
        if len(self.removed_clones) > 0:
            raise ValueError(
                "Cannot automatically upgrade panel if there are removed clones. "
                + "Please check the differences between the panels and upgrade manually."
            )

        adata_panel = (
            [
                pp
                for pp in PNAAntibodyPanelCombination.from_adata(adata).partial_panels()
                if pp.metadata.name == self.panel_1.name
                and pp.metadata.version == self.panel_1.version
            ]
            or [None]
        ).pop()
        if adata_panel is None:
            raise ValueError(
                "The provided AnnData object does not contain the panel. Cannot upgrade."
            )
        if self.panel_1 != adata_panel:
            raise ValueError(
                "The provided AnnData object does not match the panel. Cannot upgrade."
                + f" Expected panel {self.panel_2.name} v{self.panel_2.version}, "
                + f"but got panel {adata_panel.name} v{adata_panel.version}."
            )

        # first update the uns variables
        for idx in range(adata.uns.get("num_partial_panels", 0)):
            metadata_key = f"panel_metadata__{idx}"
            metadata = AntibodyPanelMetadata.model_validate(adata.uns[metadata_key])
            if (
                metadata.name == self.panel_1.name
                and metadata.version == self.panel_1.version
            ):
                adata.uns[metadata_key] = self.panel_2.metadata.to_dict()
                adata.uns[f"panel_df__{idx}"] = self.panel_2.df.to_csv()
                break

        # update the andata var table
        org_var_shape = adata.var.shape
        org_index = adata.var.index.name
        adata.var.reset_index(inplace=True)
        panel1_row_identifiers = list(
            self.panel_1.to_polars().select(self.join_on_columns).iter_rows()
        )
        var_identifiers = list(
            adata.var[self.join_on_columns].itertuples(index=False, name=None)
        )
        is_panel_row = np.array(
            [row in panel1_row_identifiers for row in var_identifiers]
        )
        row_indexes_in_var = np.array(
            [var_identifiers.index(row) for row in panel1_row_identifiers]
        )
        for col in self.removed_columns:
            if (adata.var.loc[~is_panel_row, col].fillna("") == "").all():
                adata.var.drop(col, inplace=True, errors="ignore", axis=1)
            else:
                adata.var.loc[is_panel_row, col] = pd.NA
        for col in self.added_columns:
            if col not in adata.var.columns:
                adata.var[col] = pd.NA
            else:
                assert (adata.var.loc[is_panel_row, col].fillna("") == "").all(), (
                    "added column already exists in adata.var with non-empty values for some of the"
                    + " panel rows. Cannot automatically upgrade."
                )
            adata.var.iloc[row_indexes_in_var, adata.var.columns.get_loc(col)] = (
                self.joined[col]
            )
        for col in self.changed_columns:
            adata.var.iloc[row_indexes_in_var, adata.var.columns.get_loc(col)] = (
                self.joined[col + "_panel_2"]
            )
        adata.var.set_index(org_index, inplace=True)

        # shape check
        if adata.var.shape[0] != org_var_shape[0]:
            raise ValueError(
                "Row count mismatch in automatic patch panel patch version bump."
            )
        return adata


class PNAAntibodyPanelCombination(PNAPanel):
    """Concatenation of the antibody panels used together in one sample.

    Combines base, addon, and sample-hashing panels into a single view of all
    antibodies present in the same tube. Subpanels are kept separately in
    :attr:`base_panels`, :attr:`addon_panels`, and :attr:`hashing_panels`, while
    :attr:`df` exposes the concatenated marker table (with
    ``partial_panel_name`` / ``partial_panel_type`` columns).

    Shares the :class:`~pixelator.pna.config.panel.PNAPanel` read interface for
    callers that only need the combined marker table, but several properties
    differ: :attr:`metadata` is a list, and display fields such as :attr:`name`
    / :attr:`version` join member values with ``" + "``.

    Raises if panels are incompatible (conflicting ``marker_id`` rows or
    duplicate clone sequences).
    """

    _REQUIRED_COLUMNS = {
        **PNAPanel._REQUIRED_COLUMNS,
        "partial_panel_name": str,
        "partial_panel_type": str,
    }

    base_panels: list[PNABasePanel | PartialPNAAntibodyPanel]
    hashing_panels: Optional[list[PNASampleHashingPanel]]
    addon_panels: Optional[list[PNAAddonPanel]]

    def __init__(
        self,
        panels: PartialPNAAntibodyPanel | Sequence[PartialPNAAntibodyPanel],
    ) -> None:
        """Initialize a combination from one panel or a sequence of panels.

        Args:
            panels: A single
                :class:`~pixelator.pna.config.panel.PartialPNAAntibodyPanel`
                (including typed subclasses), or a non-empty sequence of them.

        Raises:
            ValueError: If ``panels`` is empty or members conflict.
        """
        self.base_panels = []
        self.hashing_panels = None
        self.addon_panels = None

        # a combination never has a filename itself but the partial panel member might have
        self._filename = None

        if isinstance(panels, PartialPNAAntibodyPanel):
            panel_list: list[PartialPNAAntibodyPanel] = [panels]
        else:
            panel_list = list(panels)

        if not panel_list:
            raise ValueError("At least one panel is required to build a combination.")

        for panel in panel_list:
            self.add_panel(panel)

    @classmethod
    def from_csv(cls, filename: PathType) -> Self:
        """Create a one-member combination from a CSV panel file.

        Loads the panel with :func:`~pixelator.pna.config.panel.panel_from_csv`
        and wraps it as a one-member combination.

        Args:
            filename: Path to a ``.csv`` panel file with YAML front-matter.

        Returns:
            Combination containing the loaded panel.
        """
        return cls(panel_from_csv(filename))

    def __eq__(self, other: object) -> bool:
        """Check if two panels are equal based on dataframe and metadata.

        Args:
            other: Panel to compare for equality.
        """
        if not isinstance(other, PNAPanel):
            raise ValueError("Can only compare with another PNAPanel")
        return self.df.equals(other.df) and self.metadata == other.metadata

    @property
    def metadata(self) -> list[AntibodyPanelMetadata]:
        """Metadata for each member panel, in :meth:`partial_panels` order.

        Returns:
            One :class:`~pixelator.common.config.panel.AntibodyPanelMetadata`
            per subpanel. Read-only; assignment raises :class:`AttributeError`.
        """
        return [p.metadata for p in self.partial_panels()]

    @metadata.setter
    def metadata(self, _value: list[AntibodyPanelMetadata]):
        """Reject writes; combination metadata is derived from member panels.

        Raises:
            AttributeError: Always, because metadata is read-only.
        """
        raise AttributeError("Metadata for combination panels is read-only.")

    def partial_panels(self):
        """Return member panels in base, hashing, then addon order.

        Returns:
            Flat list of all subpanels in the combination.
        """
        return (
            self.base_panels + (self.hashing_panels or []) + (self.addon_panels or [])
        )

    @property
    def num_partial_panels(self):
        """Number of member panels in the combination.

        Returns:
            Count of base, hashing, and addon panels.
        """
        return sum(1 for _ in self.partial_panels())

    @staticmethod
    def _validate_no_conflicting_duplicate_markers(df_list: list[pd.DataFrame]) -> None:
        """Reject the same marker_id with incompatible rows across partial panels."""
        seen_markers: dict[str, pd.Series] = {}
        for partial_df in df_list:
            for marker_id, row in partial_df.iterrows():
                if marker_id in seen_markers:
                    if not seen_markers[marker_id].equals(row):
                        raise ValueError(
                            "Conflicting duplicate marker_id "
                            + f"'{marker_id}' across panels in combination."
                        )
                else:
                    seen_markers[marker_id] = row

    @property
    def df(self) -> pd.DataFrame:
        """Concatenated marker table for all member panels.

        Adds ``partial_panel_name`` and ``partial_panel_type`` columns and
        rejects conflicting duplicate ``marker_id`` rows or duplicate clone
        sequences across members.

        Returns:
            Combined panel dataframe indexed by ``marker_id``.

        Raises:
            ValueError: On conflicting markers or duplicate sequences.
        """
        partial_dfs = [panel.df for panel in self.partial_panels()]
        if len(partial_dfs) > 1:
            self._validate_no_conflicting_duplicate_markers(partial_dfs)

        df_list = [
            panel.to_polars()
            .with_columns(
                pl.lit(panel.name).alias("partial_panel_name"),
                pl.lit(panel.metadata.panel_type or PanelType.PARTIAL).alias(
                    "partial_panel_type"
                ),
            )
            .to_pandas()
            .set_index(self._INDEX_COLUMN)
            for panel in self.partial_panels()
        ]
        df = (
            pd.concat(
                df_list,
                axis=0,
            )
            if len(df_list) > 1
            else df_list[0]
        )
        # make sure clone sequences are unique! Otherwise raise an error
        if df.duplicated(subset=["sequence_1", "sequence_2"]).any():
            raise ValueError("Duplicate sequences found in the panel combination.")
        return df

    def add_base_panel(self, base_panel: PNABasePanel | PartialPNAAntibodyPanel):
        """Add a base (or legacy untyped) panel to the combination.

        Args:
            base_panel: A :class:`~pixelator.pna.config.panel.PNABasePanel`,
                or a legacy
                :class:`~pixelator.pna.config.panel.PartialPNAAntibodyPanel`
                without a defined type (emits a warning).

        Raises:
            ValueError: If adding the panel creates marker/sequence conflicts.
        """
        if type(base_panel) is PartialPNAAntibodyPanel:
            logger.warning(
                "Adding a PartialPNAAntibodyPanel as a base panel. "
                + "this is expected legacy behavior for panels without a defined type."
                + " Consider updating the panel metadata to set the panel type."
            )
        self.base_panels.append(base_panel)
        self._df = self.df

    def add_addon_panel(self, addon_panel: PNAAddonPanel):
        """Add an addon panel to the combination.

        Args:
            addon_panel: Addon marker panel to append.

        Raises:
            ValueError: If adding the panel creates marker/sequence conflicts.
        """
        if self.addon_panels is None:
            self.addon_panels = []
        self.addon_panels.append(addon_panel)
        self._df = self.df

    def add_hashing_panel(self, hashing_panel: PNASampleHashingPanel):
        """Add a sample-hashing panel to the combination.

        Args:
            hashing_panel: Sample-hashing panel to append.

        Raises:
            ValueError: If adding the panel creates marker/sequence conflicts.
        """
        if self.hashing_panels is None:
            self.hashing_panels = []
        self.hashing_panels.append(hashing_panel)
        self._df = self.df

    def add_panel(
        self,
        panel: PartialPNAAntibodyPanel
        | PNABasePanel
        | PNASampleHashingPanel
        | PNAAddonPanel,
    ):
        """Add a panel, routing it to the matching member list by type.

        Typed subclasses go to hashing / addon / base lists. A plain
        :class:`~pixelator.pna.config.panel.PartialPNAAntibodyPanel` is
        treated as a legacy base panel.

        Args:
            panel: Panel instance to add.

        Raises:
            ValueError: If the panel type is unsupported or conflicts with
                existing members.
        """
        # Match subclasses before PartialPNAAntibodyPanel; class patterns also
        # match instances of subclasses.
        match panel:
            case PNASampleHashingPanel():
                self.add_hashing_panel(panel)
            case PNAAddonPanel():
                self.add_addon_panel(panel)
            case PNABasePanel():
                self.add_base_panel(panel)
            case PartialPNAAntibodyPanel():
                self.add_base_panel(panel)
            case _:
                raise ValueError(f"Unknown panel type: {panel.__class__.__name__}")
        self._df = self.df

    @property
    def name(self) -> str:
        """Joined display name of all member panels.

        Returns:
            Member ``name`` values joined with ``" + "``.
        """
        return " + ".join(p.metadata.name for p in self.partial_panels())

    @property
    def product(self) -> Optional[str]:
        """Joined display product string for member panels.

        Returns:
            Member ``product`` values joined with ``" + "``, or ``None`` if no
            member defines a product. Intended for display; version-bump logic
            uses the individual member panels.
        """
        # ok to drop None values here since we anyway use the partial panels when checking this
        # property for e.g. bumping panel versions, i.e. this string is only for display
        products = [
            p.metadata.product
            for p in self.partial_panels()
            if p.metadata.product is not None
        ]
        if not products:
            return None
        return " + ".join(products)

    @property
    def version(self) -> str:
        """Joined display version of all member panels.

        Returns:
            Member ``version`` values joined with ``" + "``.
        """
        return " + ".join(p.metadata.version for p in self.partial_panels())

    @property
    def description(self) -> Optional[str]:
        """Joined display description of all member panels.

        Returns:
            Member descriptions joined with ``" + "``.
        """
        return " + ".join(str(p.metadata.description) for p in self.partial_panels())

    @property
    def aliases(self) -> list[str]:
        """Aliases of the sole member panel.

        Returns:
            Alias list when the combination has exactly one member.

        Raises:
            AttributeError: If more than one member panel is present.
        """
        if self.num_partial_panels == 1:
            return self.partial_panels()[0].aliases
        else:
            raise AttributeError(
                "Cannot get aliases for a combination of panels. "
                + "Aliases are only defined for individual panels."
            )

    @property
    def archived(self) -> Optional[bool]:
        """Whether any member panel is marked archived.

        Returns:
            ``True`` if any member has ``archived`` set.
        """
        return any(p.metadata.archived for p in self.partial_panels())

    @property
    def filename(self) -> Optional[str]:
        """Filename of the sole member panel, if available.

        Returns:
            Member filename when the combination has exactly one panel.

        Raises:
            AttributeError: If more than one member panel is present.
        """
        if self.num_partial_panels == 1:
            return self.partial_panels()[0].filename
        else:
            raise AttributeError(
                "Cannot get filename for a combination of panels. "
                + "Filename is only defined for individual panels."
            )

    @property
    def filepath(self) -> Optional[Path]:
        """Shared source path when all members come from one file.

        Returns:
            The common filepath, or ``None`` (with a warning) when members
            come from different paths.
        """
        unique_filepaths = set(p.filepath for p in self.partial_panels())
        if len(unique_filepaths) == 1:
            return unique_filepaths.pop()
        else:
            logger.warning(
                "Cannot get filepath for multiple sources of panels. "
                + "Filepath is only defined if all panels have a single source e.g. the same pxl file."
            )
            return None

    @classmethod
    def from_adata(
        cls,
        adata: AnnData,
        file_name: Optional[str] = None,
        filepath: Optional[PathType] = None,
    ) -> Self:
        """Create a combination from AnnData multi-panel ``uns`` entries.

        Uses ``num_partial_panels``, ``panel_metadata__{i}``, and
        ``panel_df__{i}`` when present. If those keys are missing, wraps a
        single panel from the appropriate typed ``from_adata`` constructor.

        Args:
            adata: AnnData with embedded panel metadata and dataframes.
            file_name: Optional basename of the source file; applied to each
                restored member panel.
            filepath: Optional full path of the source file; applied to each
                restored member panel.

        Returns:
            Combination of all panels stored on ``adata``.

        Raises:
            KeyError: If a multi-panel index is missing its dataframe.
            ValueError: If the restored panels conflict.
        """
        if "num_partial_panels" not in adata.uns:
            panel_type = get_panel_type_from_metadata(
                AntibodyPanelMetadata.from_adata(adata)[0]
            )
            subpanels = [
                panel_type.from_adata(adata, file_name=file_name, filepath=filepath)
            ]
        else:
            list_of_metadatas = AntibodyPanelMetadata.from_adata(adata)
            logger.debug(
                "Found %d panel metadata entries in the AnnData object.",
                len(list_of_metadatas),
            )
            subpanels = []
            for idx, metadata in enumerate(list_of_metadatas):
                panel_df_key = f"panel_df__{idx}"
                if panel_df_key not in adata.uns:
                    raise KeyError(
                        "The provided AnnData object contains partial panel information but is "
                        + f"missing the panel dataframe for panel at index {idx}."
                    )
                df = (
                    pd.read_csv(StringIO(adata.uns[panel_df_key]))
                    .set_index(cls._INDEX_COLUMN)
                    .fillna("")
                )
                panel_type_class = get_panel_type_from_metadata(metadata)
                subpanels.append(
                    panel_type_class(
                        df, metadata, file_name=file_name, filepath=filepath
                    )
                )

        return cls(subpanels)


class PNABasePanel(PartialPNAAntibodyPanel):
    """Core / base marker panel for a PNA assay.

    Requires metadata ``panel_type``
    :attr:`~pixelator.common.config.panel.PanelType.BASE`. Typically the main
    marker set in a
    :class:`~pixelator.pna.config.panel.PNAAntibodyPanelCombination`.
    """

    _panel_type = PanelType.BASE


class PNAAddonPanel(PartialPNAAntibodyPanel):
    """Addon marker panel used together with a base panel.

    Requires metadata ``panel_type``
    :attr:`~pixelator.common.config.panel.PanelType.ADDON`. Addon markers are
    concatenated with base (and optional hashing) panels in a
    :class:`~pixelator.pna.config.panel.PNAAntibodyPanelCombination`.
    """

    _panel_type = PanelType.ADDON


class PNASampleHashingPanel(PartialPNAAntibodyPanel):
    """Sample-hashing antibody panel for PNA.

    Requires metadata ``panel_type``
    :attr:`~pixelator.common.config.panel.PanelType.SAMPLE_HASHING` and a
    ``sample_hashing`` column that is ``True`` / ``yes`` for every row.
    """

    _panel_type = PanelType.SAMPLE_HASHING

    _REQUIRED_COLUMNS = {
        **PNAPanel._REQUIRED_COLUMNS,
        "sample_hashing": bool,
    }

    @classmethod
    def validate_antibody_panel(cls, panel_df, validate_types=True):
        """Validate panel schema plus the sample-hashing column constraint.

        Args:
            panel_df: Panel dataframe to validate.
            validate_types: If True, also check column dtypes.

        Returns:
            Validation error messages; empty means the panel is valid. Includes
            parent checks and an error when any row has ``sample_hashing`` not
            set.
        """
        return super().validate_antibody_panel(panel_df, validate_types) + (
            []
            if "sample_hashing" in panel_df.columns
            and (panel_df["sample_hashing"]).all()
            else [
                "All entries in `sample_hashing` column must be 'yes' (True) for a sample hashing panel"
            ]
        )


if TYPE_CHECKING:
    # Deprecated alias of :class:`PartialPNAAntibodyPanel` (warns on runtime access).
    PNAAntibodyPanel = PartialPNAAntibodyPanel


def __getattr__(name: str):
    """Resolve the deprecated ``PNAAntibodyPanel`` alias with a warning."""
    if name == "PNAAntibodyPanel":
        warnings.warn(
            "PNAAntibodyPanel is deprecated and will be removed in a future release. "
            "Use PartialPNAAntibodyPanel (or a typed subclass), or "
            "PNAAntibodyPanelCombination for multi-panel samples.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Cache so repeated access does not re-warn.
        globals()["PNAAntibodyPanel"] = PartialPNAAntibodyPanel
        return PartialPNAAntibodyPanel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), "PNAAntibodyPanel"})
