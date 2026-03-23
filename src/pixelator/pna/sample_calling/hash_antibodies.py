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
