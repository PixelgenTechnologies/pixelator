"""Graph-collapse reads the panel stored in the pxl file.

Copyright © 2026 Pixelgen Technologies AB.
"""

from click.testing import CliRunner

from pixelator import cli


def test_graph_collapse_rejects_files_without_a_stored_panel(pna_pxl_file, tmp_path):
    """A file that only records a panel name is not resolved by name."""
    result = CliRunner().invoke(
        cli.main_cli,
        [
            "--cores",
            "1",
            "single-cell-pna",
            "graph-collapse",
            str(pna_pxl_file),
            "--output",
            str(tmp_path),
            "--design",
            "proxiome-v1",
        ],
    )
    assert isinstance(result.exception, KeyError)
    assert "regenerate your data" in str(result.exception)
