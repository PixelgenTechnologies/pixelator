"""Single (partial) PNA antibody panels and their typed variants.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from anndata import AnnData

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from pixelator.common.config.panel import AntibodyPanelMetadata, PanelType
from pixelator.common.types import PathType
from pixelator.common.utils import logger
from pixelator.pna.config.panel.base import PNAPanel


class PartialPNAAntibodyPanel(PNAPanel):
    """A single PNA antibody panel loaded from a CSV (or equivalent source).

    This is the base type for typed panels. Concrete subclasses bind a fixed
    :class:`~pixelator.common.config.panel.PanelType`:

    * :class:`~pixelator.pna.config.panel.partial.PNABasePanel` — core markers
    * :class:`~pixelator.pna.config.panel.partial.PNAAddonPanel` — addon markers
      used with a base panel
    * :class:`~pixelator.pna.config.panel.partial.PNASampleHashingPanel` —
      sample-hashing markers

    Prefer the module-level helpers
    :func:`~pixelator.pna.config.panel.dispatch.panel_from_csv`,
    :func:`~pixelator.pna.config.panel.loaders.panel_from_adata`, and
    :func:`~pixelator.pna.config.panel.loaders.load_antibody_panel` when the
    concrete type should follow metadata. Use
    :class:`~pixelator.pna.config.panel.combination.PNAAntibodyPanelCombination`
    when several panels are used together in one sample.
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
                legacy files when constructing this base class. The default is
                applied on a copy; the caller's metadata instance is not
                modified.
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
            # Default missing panel_type to PARTIAL for legacy files without
            # mutating the caller's metadata instance (shared objects may be
            # reused for construction, registration, or later comparison).
            metadata = metadata.model_copy(update={"panel_type": PanelType.PARTIAL})
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
        class (or :func:`~pixelator.pna.config.panel.dispatch.panel_from_csv`
        for automatic dispatch).

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
        :meth:`~pixelator.pna.config.panel.combination.PNAAntibodyPanelCombination.from_adata`
        or :func:`~pixelator.pna.config.panel.loaders.panel_from_adata`.

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
            return NotImplemented
        return self.df.equals(other.df) and self.metadata == other.metadata


class PNABasePanel(PartialPNAAntibodyPanel):
    """Core / base marker panel for a PNA assay.

    Requires metadata ``panel_type``
    :attr:`~pixelator.common.config.panel.PanelType.BASE`. Typically the main
    marker set in a
    :class:`~pixelator.pna.config.panel.combination.PNAAntibodyPanelCombination`.
    """

    _panel_type = PanelType.BASE


class PNAAddonPanel(PartialPNAAntibodyPanel):
    """Addon marker panel used together with a base panel.

    Requires metadata ``panel_type``
    :attr:`~pixelator.common.config.panel.PanelType.ADDON`. Addon markers are
    concatenated with base (and optional hashing) panels in a
    :class:`~pixelator.pna.config.panel.combination.PNAAntibodyPanelCombination`.
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
