"""Combination of the PNA antibody panels used together in one sample.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl
from anndata import AnnData

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from pixelator.common.config.panel import AntibodyPanelMetadata, PanelType
from pixelator.common.types import PathType
from pixelator.common.utils import logger
from pixelator.pna.config.panel.base import PNAPanel
from pixelator.pna.config.panel.dispatch import (
    get_panel_type_from_metadata,
    panel_from_csv,
)
from pixelator.pna.config.panel.partial import (
    PartialPNAAntibodyPanel,
    PNAAddonPanel,
    PNABasePanel,
    PNASampleHashingPanel,
)
from pixelator.pna.config.panel.utils import sample_hashing_mask


class PNAAntibodyPanelCombination(PNAPanel):
    """Concatenation of the antibody panels used together in one sample.

    Combines base, addon, and sample-hashing panels into a single view of all
    antibodies present in the same tube. Subpanels are kept separately in
    :attr:`base_panels`, :attr:`addon_panels`, and :attr:`hashing_panels`, while
    :attr:`df` exposes the concatenated marker table (with
    ``partial_panel_name`` / ``partial_panel_type`` columns).

    Shares the :class:`~pixelator.pna.config.panel.base.PNAPanel` read interface
    for callers that only need the combined marker table, but several properties
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
                :class:`~pixelator.pna.config.panel.partial.PartialPNAAntibodyPanel`
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

        Loads the panel with
        :func:`~pixelator.pna.config.panel.dispatch.panel_from_csv` and wraps it
        as a one-member combination.

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
            return NotImplemented
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
        sequences across members. When any member has ``sample_hashing``,
        the combined column is normalized to bool (missing cells become
        ``False``) so concat upcasts such as ``True`` → ``1.0`` do not hide
        hashing markers.

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
        if "sample_hashing" in df.columns:
            df = df.copy()
            df["sample_hashing"] = sample_hashing_mask(df["sample_hashing"])
        # make sure clone sequences are unique! Otherwise raise an error
        if df.duplicated(subset=["sequence_1", "sequence_2"]).any():
            raise ValueError("Duplicate sequences found in the panel combination.")
        return df

    def _append_and_validate(
        self,
        attr: str,
        panel: PartialPNAAntibodyPanel
        | PNABasePanel
        | PNASampleHashingPanel
        | PNAAddonPanel,
    ) -> None:
        """Append ``panel`` to member list ``attr``, rolling back on conflict.

        Conflict checks run via :attr:`df` after the append. If they raise,
        the member list is restored so a failed add leaves the combination
        unchanged.
        """
        current = getattr(self, attr)
        created = current is None
        if created:
            current = []
            setattr(self, attr, current)
        current.append(panel)
        try:
            self._df = self.df
        except Exception:
            current.pop()
            if created:
                setattr(self, attr, None)
            raise

    def add_base_panel(self, base_panel: PNABasePanel | PartialPNAAntibodyPanel):
        """Add a base (or legacy untyped) panel to the combination.

        Args:
            base_panel: A
                :class:`~pixelator.pna.config.panel.partial.PNABasePanel`, or a
                legacy
                :class:`~pixelator.pna.config.panel.partial.PartialPNAAntibodyPanel`
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
        self._append_and_validate("base_panels", base_panel)

    def add_addon_panel(self, addon_panel: PNAAddonPanel):
        """Add an addon panel to the combination.

        Args:
            addon_panel: Addon marker panel to append.

        Raises:
            ValueError: If adding the panel creates marker/sequence conflicts.
        """
        self._append_and_validate("addon_panels", addon_panel)

    def add_hashing_panel(self, hashing_panel: PNASampleHashingPanel):
        """Add a sample-hashing panel to the combination.

        Args:
            hashing_panel: Sample-hashing panel to append.

        Raises:
            ValueError: If adding the panel creates marker/sequence conflicts.
        """
        self._append_and_validate("hashing_panels", hashing_panel)

    def add_panel(
        self,
        panel: PartialPNAAntibodyPanel
        | PNABasePanel
        | PNASampleHashingPanel
        | PNAAddonPanel,
    ):
        """Add a panel, routing it to the matching member list by type.

        Typed subclasses go to hashing / addon / base lists. A plain
        :class:`~pixelator.pna.config.panel.partial.PartialPNAAntibodyPanel` is
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
