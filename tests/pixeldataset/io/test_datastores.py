from unittest.mock import patch
from anndata import AnnData
from pandas.core.frame import DataFrame
from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds

import pytest
from pandas.testing import assert_frame_equal
from pixelator.pixeldataset import PixelDataset

from pixelator.pixeldataset.io.datastores import (
    FileFormatNotRecognizedError,
    PixelDataStore,
    ZipBasedPixelFile,
    ZipBasedPixelFileWithCSV,
    ZipBasedPixelFileWithParquet,
)


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
        dataset.save(str(file_target), file_format="csv")
        res = PixelDataStore.guess_datastore_from_path(file_target)
        assert isinstance(res, ZipBasedPixelFileWithCSV)

    def test_pixel_data_store_from_file_provides_correct_datastore(
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

        dataset, *_ = setup_basic_pixel_dataset
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

    @pytest.mark.parametrize(
        "datastore", [ZipBasedPixelFileWithParquet, ZipBasedPixelFileWithCSV]
    )
    def test_pixelfile_datastore_can_switch_between_reading_and_writing(
        self,
        datastore: type[ZipBasedPixelFileWithParquet] | type[ZipBasedPixelFileWithCSV],
        setup_basic_pixel_dataset: tuple[
            PixelDataset, DataFrame, AnnData, dict[str, int], DataFrame, DataFrame
        ],
        tmp_path: Path,
    ):
        dataset, *_ = setup_basic_pixel_dataset
        file_target = tmp_path / "dataset.pxl"
        dataset.save(
            str(file_target),
            file_format="csv"
            if datastore.EDGELIST_KEY.endswith(".csv.gz")
            else "parquet",
        )

        datastore = datastore(file_target)

        # Reading something / write something / read something
        adata = datastore.read_anndata()
        datastore.write_anndata(adata)
        _ = datastore.read_anndata()

        edgelists = datastore.read_edgelist()
        datastore.write_edgelist(edgelists)
        _ = datastore.read_edgelist()

        metadata = datastore.read_metadata()
        datastore.write_metadata(metadata)
        _ = datastore.read_metadata()

        polarization = datastore.read_polarization()
        datastore.write_polarization(polarization)
        _ = datastore.read_polarization()

        colocalization = datastore.read_colocalization()
        datastore.write_colocalization(colocalization)
        _ = datastore.read_colocalization()

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
            "pixelator.pixeldataset.io.datastores.ZipBasedPixelFileWithParquet"
        ) as mock:
            mock.EDGELIST_KEY = "non_existent_key"
            with pytest.raises(FileFormatNotRecognizedError):
                _ = ZipBasedPixelFile.guess_file_format(pixel_dataset_file)

    @pytest.mark.test_this
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
        # datastore.write_edgelist(dataset.edgelist)

        # partitioning = (
        #    ds.partitioning(pa.schema([("component", pa.string())]), flavor="hive"),
        # )
        partitioning = ["component"]
        datastore._current_mode
        datastore.write_edgelist(dataset.edgelist, partitioning=partitioning)
        # TODO Add asserts


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
        ZipBasedPixelFileWithCSV(file_target).save(dataset)
        assert file_target.is_file()
