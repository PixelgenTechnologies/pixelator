"""Schema checks for a PNA antibody panel dataframe.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import re

import pandas as pd
import polars as pl

from pixelator.pna.config.panel.hashing import (
    _hashing_marker_ids,
    collapsed_hashing_marker_id,
    split_hashing_marker_id,
)

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

# UniProt accession naming convention. The trailing alternative allows an empty id.
_UNIPROT_ID_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}|$"
)


def _uniprot_ids_are_valid(id_str: object) -> bool:
    """Return whether every semicolon-separated UniProt id matches the convention."""
    return all(bool(_UNIPROT_ID_RE.match(part)) for part in str(id_str).split(";"))


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


def _validate_structure(panel_df: pd.DataFrame) -> list[str]:
    """Return the first error that makes the remaining checks meaningless."""
    if not set(_REQUIRED_COLUMNS).issubset(set(panel_df.columns)):
        missing_columns = set(_REQUIRED_COLUMNS) - set(panel_df.columns)
        return [f"Panel has missing required columns: {missing_columns}"]

    if panel_df.shape[0] == 0:
        return ["Panel file is empty"]

    if panel_df.index.name != _INDEX_COLUMN:
        return [f"`{_INDEX_COLUMN}` is missing or is not set as index"]

    return []


def _validate_column_types(panel_df: pd.DataFrame) -> list[str]:
    """Return errors for required columns whose dtype does not match the schema."""
    errors = []
    panel_pl_df = pl.from_pandas(panel_df, include_index=True)
    for col, expected_type in (
        _REQUIRED_COLUMNS | {_INDEX_COLUMN: _INDEX_COLUMN_TYPE}
    ).items():
        found_type = panel_pl_df[col].dtype.to_python()
        if not found_type == expected_type:
            errors.append(
                f"Column {col} has incorrect type. Expected {expected_type}, got {found_type}"
            )
    return errors


def _validate_unique_values(panel_df: pd.DataFrame) -> list[str]:
    """Return errors when a unique column or the marker id repeats."""
    errors = []
    for col in _UNIQUE_COLUMNS:
        if not len(panel_df[col].unique()) == len(panel_df[col]):
            errors.append(f"All values in column: {col} were not unique")

    if panel_df.index.duplicated().any():
        duplicated = panel_df.index[panel_df.index.duplicated()].unique().tolist()
        errors.append(
            "All values in column: marker_id were not unique. "
            f"Offending values: {duplicated}"
        )
    return errors


def _validate_control_column(panel_df: pd.DataFrame) -> list[str]:
    """Return an error when the control column is not boolean."""
    if panel_df["control"].dtype != bool:
        return ["`control` column is not boolean"]
    return []


def _validate_uniprot_ids(panel_df: pd.DataFrame) -> list[str]:
    """Return an error when a present UniProt id breaks the naming convention."""
    if "uniprot_id" not in panel_df.columns:
        return []

    bad_ids = panel_df[~panel_df["uniprot_id"].apply(_uniprot_ids_are_valid)][
        "uniprot_id"
    ]
    if len(bad_ids) > 0:
        return [
            "Invalid UniProt IDs found."
            "Please conform to the naming convention or remove the following IDs:"
            f"{bad_ids.tolist()}"
        ]
    return []


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


def validate_antibody_panel(
    panel_df: pd.DataFrame, validate_types: bool = True
) -> list[str]:
    """Validate antibody panel schema and content.

    Args:
        panel_df: Dataframe containing panel markers and sequences.
        validate_types: If True, validate dataframe column types.

    Returns:
        A list of validation error messages. Empty means valid input.
    """
    errors = _validate_structure(panel_df)
    if errors:
        return errors
    if validate_types:
        errors += _validate_column_types(panel_df)
    errors += _validate_unique_values(panel_df)
    errors += _validate_marker_names(panel_df)
    errors += _validate_control_column(panel_df)
    errors += _validate_uniprot_ids(panel_df)
    errors += _validate_sequences(panel_df, "sequence_1")
    errors += _validate_sequences(panel_df, "sequence_2")
    errors += _validate_hashing_marker_ids(panel_df)
    return errors
