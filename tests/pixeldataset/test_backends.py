"""Tests for the pixeldataset.backends module.

Copyright (c) 2024 Pixelgen Technologies AB.
"""

from pixelator.pixeldataset.backends import (
    FileBasedPixelDatasetBackend,
    ObjectBasedPixelDatasetBackend,
)


def assert_backend_can_set_values(pixel_dataset_backend):
    """assert_backend_can_set_values."""
    assert pixel_dataset_backend.adata
    pixel_dataset_backend.adata = None
    assert not pixel_dataset_backend.adata

    assert not pixel_dataset_backend.edgelist.empty
    pixel_dataset_backend.edgelist = None
    assert not pixel_dataset_backend.edgelist

    assert not pixel_dataset_backend.polarization.empty
    pixel_dataset_backend.polarization = None
    assert not pixel_dataset_backend.polarization

    assert not pixel_dataset_backend.colocalization.empty
    pixel_dataset_backend.colocalization = None
    assert not pixel_dataset_backend.colocalization

    assert pixel_dataset_backend.metadata
    pixel_dataset_backend.metadata = None
    assert not pixel_dataset_backend.metadata


def test_file_based_pixel_dataset_backend_set_attrs(pixel_dataset_file):
    """test_file_based_pixel_dataset_backend_set_attrs."""
    pixel_dataset_backend = FileBasedPixelDatasetBackend(pixel_dataset_file)
    assert_backend_can_set_values(pixel_dataset_backend)


def test_object_based_pixel_dataset_backend_set_attrs(setup_basic_pixel_dataset):
    """test_object_based_pixel_dataset_backend_set_attrs."""
    (
        _,
        edgelist,
        adata,
        metadata,
        polarization_scores,
        colocalization_scores,
    ) = setup_basic_pixel_dataset
    pixel_dataset_backend = ObjectBasedPixelDatasetBackend(
        adata=adata,
        edgelist=edgelist,
        polarization=polarization_scores,
        colocalization=colocalization_scores,
        metadata=metadata,
    )
    assert_backend_can_set_values(pixel_dataset_backend)
