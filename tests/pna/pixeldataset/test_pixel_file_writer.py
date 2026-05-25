"""Copyright © 2025 Pixelgen Technologies AB."""

from pixelator.pna.pixeldataset.io import PixelFileWriter


class TestPixelFileWriter:
    """Represent test pixel file writer."""

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
