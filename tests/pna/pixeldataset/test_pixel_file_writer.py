"""Copyright © 2025 Pixelgen Technologies AB."""

from pixelator.pna.pixeldataset.io import PixelFileWriter


class TestPixelFileWriter:
    def test_write_edgelist(self, tmp_path, edgelist_parquet_path):
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            writer.write_edgelist(edgelist_parquet_path)

    def test_write_adata(self, tmp_path, adata_data):
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            writer.write_adata(adata_data)

    def test_write_metadata(self, tmp_path):
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            writer.write_metadata({"sample": "test_sample", "version": "0.1.0"})

    def test_write_proximity(self, tmp_path, proximity_parquet_path):
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            writer.write_proximity(proximity_parquet_path)

    def test_write_layouts(self, tmp_path, layout_parquet_path):
        target = tmp_path / "file.pxl"
        with PixelFileWriter(target) as writer:
            writer.write_layouts(layout_parquet_path)
