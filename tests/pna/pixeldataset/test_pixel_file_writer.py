"""Copyright © 2025 Pixelgen Technologies AB."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from anndata import AnnData

from pixelator.common.config import AntibodyPanelMetadata
from pixelator.pna.config.panel import PNAAntibodyPanel
from pixelator.pna.pixeldataset.io import PixelFileWriter, PxlFile


class TestPixelFileWriter:
    """Represent test pixel file writer."""

    def test_open_honors_duckdb_temp_dir_env(self, tmp_path, monkeypatch):
        """The writer's DuckDB connection should use PIXELATOR_DUCKDB_TEMP_DIR for spilling.

        Args:
            tmp_path: tmp path.
            monkeypatch: pytest monkeypatch fixture.
        """
        duckdb_tmp = tmp_path / "duckdb_scratch"
        duckdb_tmp.mkdir()
        monkeypatch.setenv("PIXELATOR_DUCKDB_TEMP_DIR", str(duckdb_tmp))
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            setting = (
                writer.get_connection()
                .execute("SELECT current_setting('temp_directory')")
                .fetchone()[0]
            )
        assert setting == str(duckdb_tmp.absolute())

    def test_open_defaults_temp_directory_to_tmp(self, tmp_path, monkeypatch):
        """The writer should default the DuckDB spill directory to /tmp when env is unset.

        Args:
            tmp_path: tmp path.
            monkeypatch: pytest monkeypatch fixture.
        """
        monkeypatch.delenv("PIXELATOR_DUCKDB_TEMP_DIR", raising=False)
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            setting = (
                writer.get_connection()
                .execute("SELECT current_setting('temp_directory')")
                .fetchone()[0]
            )
        assert setting == str(Path("/tmp").absolute())

    def test_write_edgelist(self, tmp_path, edgelist_parquet_path):
        """Verify write edgelist.

        Args:
            tmp_path: tmp path.
            edgelist_parquet_path: edgelist parquet path.
        """
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            writer.write_edgelist(edgelist_parquet_path)

    def test_write_adata(self, tmp_path, adata_data):
        """Verify write adata.

        Args:
            tmp_path: tmp path.
            adata_data: adata data.
        """
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            writer.write_adata(adata_data)

    def test_write_adata_multiple_samples_raises(self, tmp_path, adata_data):
        """Verify write adata raises when obs contains multiple samples.

        Args:
            tmp_path: tmp path.
            adata_data: adata data.
        """
        adata_data = adata_data.copy()
        adata_data.obs["sample"] = "sample_1"
        adata_data.obs.iloc[0, adata_data.obs.columns.get_loc("sample")] = "sample_2"
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            with pytest.raises(ValueError, match="multiple samples"):
                writer.write_adata(adata_data)

    def test_write_edgelist_multiple_samples_raises(self, tmp_path, edgelist_dataframe):
        """Verify write edgelist raises when the edgelist contains multiple samples.

        Args:
            tmp_path: tmp path.
            edgelist_dataframe: edgelist dataframe.
        """
        edgelist_dataframe = edgelist_dataframe.with_columns(sample=pl.lit("sample_1"))
        edgelist_dataframe[0, "sample"] = "sample_2"
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            with pytest.raises(ValueError, match="multiple samples"):
                writer.write_edgelist(edgelist_dataframe)

    def test_write_metadata(self, tmp_path):
        """Verify write metadata.

        Args:
            tmp_path: tmp path.
        """
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            writer.write_metadata({"sample": "test_sample", "version": "0.1.0"})

    def test_write_proximity(self, tmp_path, proximity_parquet_path):
        """Verify write proximity.

        Args:
            tmp_path: tmp path.
            proximity_parquet_path: proximity parquet path.
        """
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            writer.write_proximity(proximity_parquet_path)

    def test_write_layouts(self, tmp_path, layout_parquet_path):
        """Verify write layouts.

        Args:
            tmp_path: tmp path.
            layout_parquet_path: layout parquet path.
        """
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            writer.write_layouts(layout_parquet_path)

    def test_write_adata_omits_panel_columns_already_stored_in_the_panel_tables(
        self, tmp_path
    ):
        """Panel fields joined onto var in memory stay out of the written var."""
        panel = PNAAntibodyPanel(
            pd.DataFrame(
                {
                    "control": [False],
                    "sequence_1": ["AAAA"],
                    "sequence_2": ["AAAA"],
                    "uniprot_id": ["P01730"],
                },
                index=pd.Index(["CD3"], name="marker_id"),
            ),
            AntibodyPanelMetadata(name="base", version="1.0.0", product="proxiome"),
        )
        adata = AnnData(
            X=np.zeros((1, 1)),
            obs=pd.DataFrame(index=["c1"]),
            var=pd.DataFrame(
                {
                    "antibody_count": [1],
                    "control": [False],
                    "sequence_1": ["AAAA"],
                    "sequence_2": ["AAAA"],
                    "uniprot_id": ["P01730"],
                },
                index=pd.Index(["CD3"], name="marker_id"),
            ),
        )
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            writer.write_panel(panel)
            writer.write_adata(adata)
            columns = set(
                writer.get_connection()
                .execute("SELECT * FROM __adata__var LIMIT 0")
                .fetchdf()
                .columns
            )
        assert "antibody_count" in columns
        assert "control" not in columns
        assert "sequence_1" not in columns
        assert "sequence_2" not in columns
        assert "uniprot_id" not in columns

    def test_write_adata_keeps_var_columns_when_no_panel_is_stored(self, tmp_path):
        """A file without panel tables keeps whatever columns var already has."""
        adata = AnnData(
            X=np.zeros((1, 1)),
            obs=pd.DataFrame(index=["c1"]),
            var=pd.DataFrame(
                {"antibody_count": [1], "control": [False]},
                index=pd.Index(["CD3"], name="marker_id"),
            ),
        )
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            writer.write_adata(adata)
            columns = set(
                writer.get_connection()
                .execute("SELECT * FROM __adata__var LIMIT 0")
                .fetchdf()
                .columns
            )
        assert "antibody_count" in columns
        assert "control" in columns

    def test_write_adata_stores_a_legacy_panel_before_replacing_it(self, tmp_path):
        """A 0.22–0.30 file keeps its panel when AnnData is written again."""
        legacy = AnnData(
            X=np.zeros((1, 1)),
            obs=pd.DataFrame(index=["c1"]),
            var=pd.DataFrame(
                {
                    "antibody_count": [1],
                    "control": [False],
                    "sequence_1": ["AAAA"],
                    "sequence_2": ["AAAA"],
                },
                index=pd.Index(["CD3"], name="marker_id"),
            ),
        )
        legacy.uns["panel_metadata"] = {
            "name": "base",
            "version": "1.0.0",
            "product": "proxiome",
            "panel_columns": ["control", "sequence_1", "sequence_2"],
        }
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            writer.write_adata(legacy)

        rewritten = AnnData(
            X=np.zeros((1, 1)),
            obs=pd.DataFrame({"k_core_1": [3]}, index=["c1"]),
            var=legacy.var.copy(),
        )
        with PixelFileWriter(target) as writer:
            writer.write_adata(rewritten)
            columns = set(
                writer.get_connection()
                .execute("SELECT * FROM __adata__var LIMIT 0")
                .fetchdf()
                .columns
            )
            uns_row = (
                writer.get_connection()
                .execute("SELECT value FROM __adata__uns")
                .fetchone()
            )

        uns = json.loads(uns_row[0]) if isinstance(uns_row[0], str) else uns_row[0]
        panel = PxlFile(target).read_panel()
        assert panel is not None
        assert panel.name == "base"
        assert panel.version == "1.0.0"
        assert list(panel.markers) == ["CD3"]
        assert "panel_metadata" not in uns
        assert "antibody_count" in columns
        assert "control" not in columns
        assert "sequence_1" not in columns
        assert "sequence_2" not in columns
