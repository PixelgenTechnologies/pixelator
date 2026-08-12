"""Copyright © 2025 Pixelgen Technologies AB."""

from pathlib import Path

from pixelator.pna.pixeldataset.io import PixelFileWriter


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
