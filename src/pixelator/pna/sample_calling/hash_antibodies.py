"""Hash antibody mapping for sample calling.

Copyright © 2025 Pixelgen Technologies AB.
"""

from collections import defaultdict

import polars as pl


def _verify_no_antibody_in_multiple_samples(mapping: dict[str, list[str]]) -> None:
    """Raise ValueError if any antibody is assigned to more than one sample."""
    antibody_to_samples: dict[str, list[str]] = defaultdict(list)
    for sample, antibodies in mapping.items():
        for ab in antibodies:
            antibody_to_samples[ab].append(sample)
    for ab, samples in antibody_to_samples.items():
        if len(samples) > 1:
            raise ValueError(
                f"Antibody hash '{ab}' is assigned to multiple samples: {samples}. "
                "Please ensure each antibody hash is unique to a sample."
            )


class HashedAntibodyMapping(dict[str, list[str]]):
    """Mapping of samples to hashing antibodies.

    Also stores information about all hashing antibodies, including those that are not mapped to any
    sample.
    """

    def __init__(
        self,
        mapping: dict[str, list[str]],
        *,
        all_hashing_antibodies: list[str] | set[str],
    ):
        """Initialize the HashedAntibodyMapping.

        Args:
            mapping: Sample name -> list of hashed antibody names for that sample.
            all_hashing_antibodies: Full list of all hashing antibodies (e.g. from panel). Required.

        Raises:
            ValueError: If any antibody is assigned to more than one sample.

        """
        _verify_no_antibody_in_multiple_samples(mapping)
        self._hashing_antibodies = set(all_hashing_antibodies)
        super().__init__(mapping)

    @classmethod
    def from_samplesheet(
        cls,
        samplesheet_df: pl.DataFrame,
        all_hashing_antibodies: list[str] | set[str],
        pool_name: str,
    ) -> "HashedAntibodyMapping":
        """Create a HashedAntibodyMapping from a samplesheet and a full list of hashing antibodies.

        Args:
            samplesheet_df: Must have columns pool, sample, hash_index.
            all_hashing_antibodies: Full list/set of hashing antibodies (e.g. from panel). Required.
            pool_name: Pool to filter by in the samplesheet.

        """
        if "hash_index" not in samplesheet_df.columns:
            raise ValueError(
                "The samplesheet is missing the 'hash_index' column. "
                "Add a column with that exact name and put one number per row (1, 2, 3, ...) "
                "so each row matches the numeric suffix on hashing antibody names in your panel "
                "(for example names ending in -1, -2)."
            )
        if samplesheet_df["hash_index"].dtype != pl.Int64:
            raise ValueError(
                "The 'hash_index' column must contain numbers as integers (i.e. 1 not 1.0). "
                "Fix your spreadsheet or CSV so that column is plain integers (no quotes, no spaces "
                "around digits, no decimal points). If you build the table in code"
            )
        if "pool" not in samplesheet_df.columns:
            raise ValueError(
                "The samplesheet is missing the 'pool' column. "
                "Add a column named exactly 'pool' and fill it with the pool identifier for each row; "
                "those values must include the pool name used for your run (the same pool you pass "
                "or that is inferred from the input file)."
            )
        if "sample" not in samplesheet_df.columns:
            raise ValueError(
                "The samplesheet is missing the 'sample' column. "
                "Add a column named exactly 'sample' and put the sample name for each row there."
            )

        hash_antibodies = set(all_hashing_antibodies)
        mapping = {}
        for row in (
            samplesheet_df.filter(pl.col("pool") == pool_name)
            .select("sample", "hash_index")
            .iter_rows(named=True)
        ):
            sample = row["sample"]
            hash_index = str(row["hash_index"])
            mapping[sample] = sorted(
                [ab for ab in hash_antibodies if ab.endswith(f"-{hash_index}")]
            )

        if not mapping:
            raise ValueError(
                f"No matching entries found in samplesheet for pool '{pool_name}'. "
                + "Please make sure that the pool column in the samplesheet matches the pool name "
                + "derived from the input file."
            )
        return cls(mapping, all_hashing_antibodies=sorted(hash_antibodies))

    @property
    def hashing_antibodies(self) -> set[str]:
        """All hashing antibodies (regardless of whether they are mapped to a sample)."""
        return self._hashing_antibodies

    @property
    def unmapped_hashing_antibodies(self) -> set[str]:
        """Hashing antibodies that are not mapped to any sample."""
        mapped = {ab for ab_list in self.values() for ab in ab_list}
        return self._hashing_antibodies - mapped
