"""Tests that the stored panel is kept when a pxl file is rewritten.

Copyright © 2026 Pixelgen Technologies AB.
"""

import duckdb

from pixelator.pna.config.panel.tables import stored_panel_marker_columns
from pixelator.pna.pixeldataset import PixelDatasetSaver

COMPONENT = "fc07dea9b679aca7"


def test_the_panel_survives_saving_a_filtered_dataset(pxl_dataset, panel, tmp_path):
    filtered = pxl_dataset.filter(components={COMPONENT})

    saved = PixelDatasetSaver(filtered).save("test_sample", tmp_path / "saved.pxl")

    assert saved.read_panel() == panel


def test_a_rewritten_file_does_not_store_the_panel_columns_on_var_again(
    pxl_dataset, tmp_path
):
    filtered = pxl_dataset.filter(components={COMPONENT})

    saved = PixelDatasetSaver(filtered).save("test_sample", tmp_path / "saved.pxl")

    with duckdb.connect(str(saved.path), read_only=True) as connection:
        panel_columns = stored_panel_marker_columns(connection)
        var_columns = set(
            connection.execute("SELECT * FROM __adata__var LIMIT 0").fetchdf().columns
        )
    assert panel_columns
    assert not panel_columns & var_columns
