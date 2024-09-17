"""Tests for the pixeldataset.io.datastores module.

Copyright © 2024 Pixelgen Technologies AB.
"""

from pathlib import Path
from shutil import make_archive
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd
import polars as pl
import pytest
from anndata import AnnData
from fsspec.implementations.zip import ZipFileSystem
from pandas.core.frame import DataFrame
from pandas.testing import assert_frame_equal

from pixelator.pixeldataset import PixelDataset
from pixelator.pixeldataset.datastores import (
    CannotOverwriteError,
    FileFormatNotRecognizedError,
    PixelDataStore,
    ZipBasedPixelFile,
    ZipBasedPixelFileWithCSV,
    ZipBasedPixelFileWithParquet,
    _CustomZipFileSystem,
)
from pixelator.pixeldataset.precomputed_layouts import PreComputedLayouts


class TestPixelDataStore:
    def test_pixel_data_store_guess_from_path_parquet(self, pixel_dataset_file: Path):
        res = PixelDataStore.guess_datastore_from_path(pixel_dataset_file)
        assert isinstance(res, ZipBasedPixelFileWithParquet)

    def test_pixel_data_store_guess_from_path_csv(
        self,
        setup_basic_pixel_dataset: tuple[
            PixelDataset, DataFrame, AnnData, dict[str, int], DataFrame, DataFrame
        ],
        tmp_path: Path,
    ):
        dataset, *_ = setup_basic_pixel_dataset
        file_target = tmp_path / "dataset.pxl"
        # Writing pre-computed layouts is not supported for csv files
        dataset.precomputed_layouts = None  # type: ignore
        dataset.save(str(file_target), file_format="csv")
        res = PixelDataStore.guess_datastore_from_path(file_target)
        assert isinstance(res, ZipBasedPixelFileWithCSV)

    def test_pixel_data_store_from_file_provides_correct_datastore_parquet(
        self,
        setup_basic_pixel_dataset: tuple[
            PixelDataset, DataFrame, AnnData, dict[str, int], DataFrame, DataFrame
        ],
        tmp_path: Path,
    ):
        dataset, *_ = setup_basic_pixel_dataset
        file_target = tmp_path / "dataset.pxl"
        dataset.save(str(file_target), file_format="parquet")
        res = PixelDataStore.from_path(file_target)
        assert isinstance(res, ZipBasedPixelFileWithParquet)

    def test_pixel_data_store_from_file_provides_correct_datastore_csv(
        self,
        setup_basic_pixel_dataset: tuple[
            PixelDataset, DataFrame, AnnData, dict[str, int], DataFrame, DataFrame
        ],
        tmp_path: Path,
    ):
        dataset, *_ = setup_basic_pixel_dataset
        # Writing pre-computed layouts is not supported for csv files
        dataset.precomputed_layouts = None  # type: ignore
        file_target = tmp_path / "dataset.pxl"
        dataset.save(str(file_target), file_format="csv")
        res = PixelDataStore.from_path(file_target)
        assert isinstance(res, ZipBasedPixelFileWithCSV)

    def test_pixelfile_datastore_can_read_lazy_edgelist(self, pixel_dataset_file: Path):
        datastore = PixelDataStore.guess_datastore_from_path(pixel_dataset_file)
        lazy_edgelist = datastore.read_edgelist_lazy()
        assert_frame_equal(
            lazy_edgelist.collect().to_pandas(), datastore.read_edgelist()
        )

    def test_pixelfile_datastore_file_format_version(self, pixel_dataset_file: Path):
        dataset = PixelDataStore.guess_datastore_from_path(pixel_dataset_file)
        result = dataset.file_format_version()
        assert result == 1

    def test_pixelfile_datastore_trying_to_write_with_same_name_raises_for_parquet(
        self,
        setup_basic_pixel_dataset: tuple[
            PixelDataset, DataFrame, AnnData, dict[str, int], DataFrame, DataFrame
        ],
        tmp_path: Path,
    ):
        dataset, *_ = setup_basic_pixel_dataset
        file_target = tmp_path / "dataset.pxl"
        dataset.save(
            str(file_target),
            file_format="parquet",
        )

        datastore_inst = ZipBasedPixelFileWithParquet(file_target)

        with pytest.raises(CannotOverwriteError):
            # Reading something / write something / read something
            adata = datastore_inst.read_anndata()
            datastore_inst.write_anndata(adata)
            _ = datastore_inst.read_anndata()

        with pytest.raises(CannotOverwriteError):
            edgelists = datastore_inst.read_edgelist()
            datastore_inst.write_edgelist(edgelists)
            _ = datastore_inst.read_edgelist()

        with pytest.raises(CannotOverwriteError):
            metadata = datastore_inst.read_metadata()
            datastore_inst.write_metadata(metadata)
            _ = datastore_inst.read_metadata()

        with pytest.raises(CannotOverwriteError):
            polarization = datastore_inst.read_polarization()
            datastore_inst.write_polarization(polarization)
            _ = datastore_inst.read_polarization()

        with pytest.raises(CannotOverwriteError):
            colocalization = datastore_inst.read_colocalization()
            datastore_inst.write_colocalization(colocalization)
            _ = datastore_inst.read_colocalization()

        with pytest.raises(CannotOverwriteError):
            layouts = datastore_inst.read_precomputed_layouts()
            datastore_inst.write_precomputed_layouts(layouts)
            _ = datastore_inst.read_precomputed_layouts()

    def test_pixelfile_datastore_trying_to_write_with_same_name_raises_for_csv(
        self,
        setup_basic_pixel_dataset: tuple[
            PixelDataset, DataFrame, AnnData, dict[str, int], DataFrame, DataFrame
        ],
        tmp_path: Path,
    ):
        dataset, *_ = setup_basic_pixel_dataset
        # Writing pre-computed layouts is not supported for csv files
        dataset.precomputed_layouts = None  # type: ignore
        file_target = tmp_path / "dataset.pxl"
        dataset.save(
            str(file_target),
            file_format="csv",
        )

        datastore_inst = ZipBasedPixelFileWithCSV(file_target)

        with pytest.raises(CannotOverwriteError):
            # Reading something / write something / read something
            adata = datastore_inst.read_anndata()
            datastore_inst.write_anndata(adata)
            _ = datastore_inst.read_anndata()

        with pytest.raises(CannotOverwriteError):
            edgelists = datastore_inst.read_edgelist()
            datastore_inst.write_edgelist(edgelists)
            _ = datastore_inst.read_edgelist()

        with pytest.raises(CannotOverwriteError):
            metadata = datastore_inst.read_metadata()
            datastore_inst.write_metadata(metadata)
            _ = datastore_inst.read_metadata()

        with pytest.raises(CannotOverwriteError):
            polarization = datastore_inst.read_polarization()
            datastore_inst.write_polarization(polarization)
            _ = datastore_inst.read_polarization()

        with pytest.raises(CannotOverwriteError):
            colocalization = datastore_inst.read_colocalization()
            datastore_inst.write_colocalization(colocalization)
            _ = datastore_inst.read_colocalization()

        # csv files do not support reading/writing layouts
        # since they require hive-style parquet files

    def test_read_metadata(self, pixel_dataset_file: Path):
        datastore = PixelDataStore.guess_datastore_from_path(pixel_dataset_file)

        result = datastore.read_metadata()
        assert result == {"A": 1, "B": 2, "file_format_version": 1}

    def test_read_metadata_returns_empty_when_not_set(self, pixel_dataset_file: Path):
        datastore = PixelDataStore.guess_datastore_from_path(pixel_dataset_file)
        with patch.object(ZipBasedPixelFile, "_file_system") as mock_file_system:
            mock_file_system.open.side_effect = FileNotFoundError
            result = datastore.read_metadata()
            assert result == {}


class TestZipBasedPixelFile:
    def test_zip_based_pixel_file_from_file(self, pixel_dataset_file: Path):
        result = ZipBasedPixelFile.from_file(pixel_dataset_file)
        assert isinstance(result, ZipBasedPixelFileWithParquet)

    def test_zip_based_pixel_guess_file_format(self, pixel_dataset_file: Path):
        result = ZipBasedPixelFile.guess_file_format(pixel_dataset_file)
        assert isinstance(result, ZipBasedPixelFileWithParquet)

    def test_zip_based_pixel_guess_file_format_raises(self, pixel_dataset_file: Path):
        with patch(
            "pixelator.pixeldataset.datastores.ZipBasedPixelFileWithParquet"
        ) as mock:
            mock.EDGELIST_KEY = "non_existent_key"
            with pytest.raises(FileFormatNotRecognizedError):
                _ = ZipBasedPixelFile.guess_file_format(pixel_dataset_file)

    def test_pixelfile_datastore_can_write_with_partitioning(
        self,
        setup_basic_pixel_dataset: tuple[
            PixelDataset, DataFrame, AnnData, dict[str, int], DataFrame, DataFrame
        ],
        tmp_path: Path,
    ):
        dataset, *_ = setup_basic_pixel_dataset
        file_target = tmp_path / "dataset.pxl"
        datastore = ZipBasedPixelFileWithParquet(file_target)

        partitioning = ["component"]
        datastore.write_edgelist(dataset.edgelist, partitioning=partitioning)

        assert set(list(datastore._file_system.walk("/edgelist.parquet/"))[0][1]) == {
            "component=2ac2ca983a4b82dd",
            "component=6ed5d4e4cfe588bd",
            "component=701ec72d3bda62d5",
            "component=ce2709afa8ebd1c9",
            "component=bec92437d668cfa1",
        }

    def test_pixelfile_datastore_can_read_layouts(
        self, tmp_path: Path, layout_df: pd.DataFrame
    ):
        file_target = tmp_path / "dataset.pxl"

        precomputed_layout = PreComputedLayouts(pl.DataFrame(layout_df).lazy())
        with ZipBasedPixelFileWithParquet(file_target) as datastore:
            datastore.write_precomputed_layouts(precomputed_layout)

        with ZipBasedPixelFileWithParquet(file_target) as datastore:
            result = datastore.read_precomputed_layouts()
            assert not result.is_empty

    def test_pixelfile_datastore_can_write_layouts(
        self,
        tmp_path: Path,
        layout_df: pd.DataFrame,
    ):
        file_target = tmp_path / "dataset.pxl"
        precomputed_layout = PreComputedLayouts(pl.DataFrame(layout_df).lazy())
        with ZipBasedPixelFileWithParquet(file_target) as datastore:
            datastore.write_precomputed_layouts(layouts=precomputed_layout)

        assert set(ZipFile(file_target).namelist()) == {
            "layouts.parquet/graph_projection=a-node/layout=fr/component=6ed5d4e4cfe588bd/part-0.parquet",
            "layouts.parquet/graph_projection=a-node/layout=fr/component=2ac2ca983a4b82dd/part-0.parquet",
            "layouts.parquet/graph_projection=a-node/layout=fr/component=bec92437d668cfa1/part-0.parquet",
            "layouts.parquet/graph_projection=bipartite/layout=fr/component=701ec72d3bda62d5/part-0.parquet",
            "layouts.parquet/graph_projection=a-node/layout=pmds/component=2ac2ca983a4b82dd/part-0.parquet",
            "layouts.parquet/graph_projection=bipartite/layout=fr/component=2ac2ca983a4b82dd/part-0.parquet",
            "layouts.parquet/graph_projection=a-node/layout=pmds/component=ce2709afa8ebd1c9/part-0.parquet",
            "layouts.parquet/graph_projection=bipartite/layout=pmds/component=ce2709afa8ebd1c9/part-0.parquet",
            "layouts.parquet/graph_projection=bipartite/layout=fr/component=6ed5d4e4cfe588bd/part-0.parquet",
            "layouts.parquet/graph_projection=a-node/layout=pmds/component=701ec72d3bda62d5/part-0.parquet",
            "layouts.parquet/graph_projection=a-node/layout=pmds/component=6ed5d4e4cfe588bd/part-0.parquet",
            "layouts.parquet/graph_projection=bipartite/layout=pmds/component=701ec72d3bda62d5/part-0.parquet",
            "layouts.parquet/graph_projection=bipartite/layout=fr/component=ce2709afa8ebd1c9/part-0.parquet",
            "layouts.parquet/graph_projection=a-node/layout=fr/component=701ec72d3bda62d5/part-0.parquet",
            "layouts.parquet/graph_projection=a-node/layout=fr/component=ce2709afa8ebd1c9/part-0.parquet",
            "layouts.parquet/graph_projection=bipartite/layout=pmds/component=bec92437d668cfa1/part-0.parquet",
            "layouts.parquet/graph_projection=bipartite/layout=pmds/component=2ac2ca983a4b82dd/part-0.parquet",
            "layouts.parquet/graph_projection=bipartite/layout=pmds/component=6ed5d4e4cfe588bd/part-0.parquet",
            "layouts.parquet/graph_projection=a-node/layout=pmds/component=bec92437d668cfa1/part-0.parquet",
            "layouts.parquet/graph_projection=bipartite/layout=fr/component=bec92437d668cfa1/part-0.parquet",
        }

    def test_pixelfile_datastore_can_write_with_partitioning_with_multiple_partitions(
        self,
        setup_basic_pixel_dataset: tuple[
            PixelDataset, DataFrame, AnnData, dict[str, int], DataFrame, DataFrame
        ],
        tmp_path: Path,
    ):
        dataset, *_ = setup_basic_pixel_dataset
        file_target = tmp_path / "dataset.pxl"
        datastore = ZipBasedPixelFileWithParquet(file_target)

        # Just adding marker here as it is available in practice this
        # is not how we want to partition the files
        partitioning = ["component", "marker"]
        datastore.write_edgelist(dataset.edgelist, partitioning=partitioning)

        assert (
            list(datastore._file_system.walk("/edgelist.parquet/"))[1][0]
            == "edgelist.parquet/component=2ac2ca983a4b82dd"
        )
        assert set(list(datastore._file_system.walk("/edgelist.parquet/"))[1][1]) == {
            "marker=CD20",
            "marker=CD3",
            "marker=CD45",
            "marker=CD45RA",
            "marker=CD72",
            "marker=IsoT_ctrl",
            "marker=hashtag",
        }


class TestZipBasedPixelFileWithParquet:
    def test_pixel_file_parquet_format_spec_can_save(
        self,
        setup_basic_pixel_dataset: tuple[
            PixelDataset, DataFrame, AnnData, dict[str, int], DataFrame, DataFrame
        ],
        tmp_path: Path,
    ):
        """test_pixel_file_parquet_format_spec_can_save."""
        dataset, *_ = setup_basic_pixel_dataset
        file_target = tmp_path / "dataset.pxl"
        assert not file_target.is_file()
        ZipBasedPixelFileWithParquet(file_target).save(dataset)
        assert file_target.is_file()


class TestZipBasedPixelFileWithCSV:
    def test_pixel_file_csv_format_spec_can_save(
        self,
        setup_basic_pixel_dataset: tuple[
            PixelDataset, DataFrame, AnnData, dict[str, int], DataFrame, DataFrame
        ],
        tmp_path: Path,
    ):
        """test_pixel_file_csv_format_spec_can_save."""
        dataset, *_ = setup_basic_pixel_dataset
        file_target = tmp_path / "dataset.pxl"
        assert not file_target.is_file()
        # Writing pre-computed layouts is not supported for csv files
        dataset.precomputed_layouts = None  # type: ignore
        ZipBasedPixelFileWithCSV(file_target).save(dataset)
        assert file_target.is_file()


@pytest.mark.test_this
class TestCustomZipFileSystem:
    @pytest.fixture(name="zip_file")
    def zip_file_fixture(self, tmp_path):
        data_dir = tmp_path / "data/"
        data_dir.mkdir()
        file1 = data_dir / "file1.txt"
        file1.write_text("Hello, World!")
        file2 = data_dir / "file2.txt"
        file2.write_text("Lorem ipsum dolor sit amet")

        empty_dir = data_dir / "dir1"
        empty_dir.mkdir()

        dir_with_files = data_dir / "dir2"
        dir_with_files.mkdir()
        file3 = dir_with_files / "file3.txt"
        file3.write_text("Hello!")

        zip_file = tmp_path / "test"
        return Path(make_archive(zip_file, "zip", data_dir))

    @pytest.mark.parametrize("detail", [True, False])
    @pytest.mark.parametrize("withdirs", [True, False])
    @pytest.mark.parametrize("max_depth", [None, 1, 2])
    def test_ensure_parity(self, zip_file, detail, withdirs, max_depth):
        custom_zip_file_system = _CustomZipFileSystem(zip_file)
        zip_file_system = ZipFileSystem(zip_file)
        result = custom_zip_file_system.find(
            "/", detail=detail, withdirs=withdirs, max_depth=max_depth
        )
        expected_result = zip_file_system.find(
            "/", detail=detail, withdirs=withdirs, max_depth=max_depth
        )
        assert result
        assert result == expected_result

    def test_find_returns_expected_result_detail_true(self, zip_file):
        custom_zip_file_system = _CustomZipFileSystem(zip_file)
        zip_file_system = ZipFileSystem(zip_file)

        result = custom_zip_file_system.find("/", detail=True)
        expected_result = zip_file_system.find("/", detail=True)

        assert result
        assert result == expected_result

    def test_find_returns_expected_result_detail_false(self, zip_file):
        custom_zip_file_system = _CustomZipFileSystem(zip_file)
        zip_file_system = ZipFileSystem(zip_file)

        result = custom_zip_file_system.find("/", detail=False)
        expected_result = zip_file_system.find("/", detail=False)

        assert result
        assert result == expected_result

    def test_find_returns_expected_result_detail_true_include_dirs(self, zip_file):
        custom_zip_file_system = _CustomZipFileSystem(zip_file)
        zip_file_system = ZipFileSystem(zip_file)

        result = custom_zip_file_system.find("/", detail=True, withdirs=True)
        expected_result = zip_file_system.find("/", detail=True, withdirs=True)

        assert result
        assert result == expected_result

    def test_find_returns_expected_result_detail_false_include_dirs(self, zip_file):
        custom_zip_file_system = _CustomZipFileSystem(zip_file)
        zip_file_system = ZipFileSystem(zip_file)

        result = custom_zip_file_system.find("/", detail=False, withdirs=True)
        expected_result = zip_file_system.find("/", detail=False, withdirs=True)

        assert result
        assert result == expected_result

    def test_find_returns_expected_result_recursion_depth_set(self, zip_file):
        custom_zip_file_system = _CustomZipFileSystem(zip_file)
        zip_file_system = ZipFileSystem(zip_file)

        result = custom_zip_file_system.find("/", maxdepth=1)
        expected_result = zip_file_system.find("/", maxdepth=1)

        assert result
        assert result == expected_result

    def test_find_returns_expected_result_path_set(self, zip_file):
        custom_zip_file_system = _CustomZipFileSystem(zip_file)
        zip_file_system = ZipFileSystem(zip_file)

        result = custom_zip_file_system.find("/dir2")
        expected_result = zip_file_system.find("/dir2")

        assert result
        assert result == expected_result
