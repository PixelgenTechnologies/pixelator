"""Test fixtures for reporting tests.

Copyright © 2022 Pixelgen Technologies AB.
"""

import itertools
import shutil
import subprocess
import uuid
from pathlib import Path

import pandas as pd
import pytest

from pixelator import PixelDataset
from pixelator.report.common import PixelatorWorkdir


@pytest.fixture(scope="session")
def local_assets_dir() -> Path:
    return Path(__file__).parent / "assets"


@pytest.fixture()
def local_assets_dir_reports_only() -> Path:
    return Path(__file__).parent / "assets/reports_only"


@pytest.fixture()
def pixelator_rundir(tmpdir_factory) -> Path:
    tmpdir = tmpdir_factory.mktemp(f"pixelator-{uuid.uuid4().hex}")
    return Path(tmpdir)


@pytest.fixture()
def pixelator_workdir(pixelator_rundir) -> PixelatorWorkdir:
    return PixelatorWorkdir(pixelator_rundir)


@pytest.fixture()
def amplicon_stage_report_pbmcs(
    local_assets_dir_reports_only, pixelator_workdir
) -> Path:
    stage_dir = pixelator_workdir.stage_dir("amplicon")
    filename = local_assets_dir_reports_only / "amplicon/pbmcs_unstimulated.report.json"
    dst = shutil.copyfile(filename, stage_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def amplicon_stage_report_uropod(
    local_assets_dir_reports_only, pixelator_workdir
) -> Path:
    stage_dir = pixelator_workdir.stage_dir("amplicon")
    filename = local_assets_dir_reports_only / "amplicon/uropod_control.report.json"
    dst = shutil.copyfile(filename, stage_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def amplicon_stage_all_reports(
    amplicon_stage_report_pbmcs, amplicon_stage_report_uropod
) -> list[Path]:
    return [amplicon_stage_report_pbmcs, amplicon_stage_report_uropod]


@pytest.fixture()
def preqc_stage_report_pbmcs(local_assets_dir_reports_only, pixelator_workdir) -> Path:
    preqc_dir = pixelator_workdir.stage_dir("preqc")
    filename = local_assets_dir_reports_only / "preqc/pbmcs_unstimulated.report.json"
    dst = shutil.copyfile(filename, preqc_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def preqc_stage_report_uropod(local_assets_dir_reports_only, pixelator_workdir) -> Path:
    preqc_dir = pixelator_workdir.stage_dir("preqc")
    filename = local_assets_dir_reports_only / "preqc/uropod_control.report.json"
    dst = shutil.copyfile(filename, preqc_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def preqc_stage_all_reports(
    preqc_stage_report_pbmcs, preqc_stage_report_uropod
) -> list[Path]:
    return [preqc_stage_report_pbmcs, preqc_stage_report_uropod]


@pytest.fixture()
def adapterqc_stage_report_pbmcs(
    local_assets_dir_reports_only, pixelator_workdir
) -> Path:
    preqc_dir = pixelator_workdir.stage_dir("adapterqc")
    filename = (
        local_assets_dir_reports_only / "adapterqc/pbmcs_unstimulated.report.json"
    )
    dst = shutil.copyfile(filename, preqc_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def adapterqc_stage_report_uropod(
    local_assets_dir_reports_only, pixelator_workdir
) -> Path:
    preqc_dir = pixelator_workdir.stage_dir("adapterqc")
    filename = local_assets_dir_reports_only / "adapterqc/uropod_control.report.json"
    dst = shutil.copyfile(filename, preqc_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def adapterqc_stage_all_reports(
    adapterqc_stage_report_pbmcs, adapterqc_stage_report_uropod
) -> list[Path]:
    return [adapterqc_stage_report_pbmcs, adapterqc_stage_report_uropod]


@pytest.fixture()
def demux_stage_report_pbmcs(local_assets_dir_reports_only, pixelator_workdir) -> Path:
    preqc_dir = pixelator_workdir.stage_dir("demux")
    filename = local_assets_dir_reports_only / "demux/pbmcs_unstimulated.report.json"
    dst = shutil.copyfile(filename, preqc_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def demux_stage_report_uropod(local_assets_dir_reports_only, pixelator_workdir) -> Path:
    preqc_dir = pixelator_workdir.stage_dir("demux")
    filename = local_assets_dir_reports_only / "demux/uropod_control.report.json"
    dst = shutil.copyfile(filename, preqc_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def demux_stage_all_reports(
    demux_stage_report_pbmcs, demux_stage_report_uropod
) -> list[Path]:
    return [demux_stage_report_pbmcs, demux_stage_report_uropod]


@pytest.fixture()
def collapse_stage_report_pbmcs(
    local_assets_dir_reports_only, pixelator_workdir
) -> Path:
    collapse_dir = pixelator_workdir.stage_dir("collapse")
    filename = local_assets_dir_reports_only / "collapse/pbmcs_unstimulated.report.json"
    dst = shutil.copyfile(filename, collapse_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def collapse_stage_report_uropod(
    local_assets_dir_reports_only, pixelator_workdir
) -> Path:
    collapse_dir = pixelator_workdir.stage_dir("collapse")
    filename = local_assets_dir_reports_only / "collapse/uropod_control.report.json"
    dst = shutil.copyfile(filename, collapse_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def collapse_stage_all_reports(
    collapse_stage_report_pbmcs, collapse_stage_report_uropod
) -> list[Path]:
    return [collapse_stage_report_pbmcs, collapse_stage_report_uropod]


@pytest.fixture()
def graph_stage_report_pbmcs(local_assets_dir_reports_only, pixelator_workdir) -> Path:
    preqc_dir = pixelator_workdir.stage_dir("graph")
    filename = local_assets_dir_reports_only / "graph/pbmcs_unstimulated.report.json"
    dst = shutil.copyfile(filename, preqc_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def graph_stage_report_uropod(local_assets_dir_reports_only, pixelator_workdir) -> Path:
    preqc_dir = pixelator_workdir.stage_dir("graph")
    filename = local_assets_dir_reports_only / "graph/uropod_control.report.json"
    dst = shutil.copyfile(filename, preqc_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def graph_stage_all_reports(
    graph_stage_report_pbmcs, graph_stage_report_uropod
) -> list[Path]:
    return [graph_stage_report_pbmcs, graph_stage_report_uropod]


@pytest.fixture()
def annotate_stage_report_pbmcs(
    local_assets_dir_reports_only, pixelator_workdir
) -> Path:
    dir = pixelator_workdir.stage_dir("annotate")
    filename = local_assets_dir_reports_only / "annotate/pbmcs_unstimulated.report.json"
    dst = shutil.copyfile(filename, dir / filename.name)
    return Path(dst)


@pytest.fixture()
def annotate_stage_report_uropod(
    local_assets_dir_reports_only, pixelator_workdir
) -> Path:
    dir = pixelator_workdir.stage_dir("annotate")
    filename = local_assets_dir_reports_only / "annotate/uropod_control.report.json"
    dst = shutil.copyfile(filename, dir / filename.name)
    return Path(dst)


@pytest.fixture()
def annotate_stage_all_reports(
    annotate_stage_report_pbmcs, annotate_stage_report_uropod
) -> list[Path]:
    return [annotate_stage_report_pbmcs, annotate_stage_report_uropod]


@pytest.fixture()
def analysis_stage_report_pbmcs(
    local_assets_dir_reports_only, pixelator_workdir
) -> Path:
    dir = pixelator_workdir.stage_dir("analysis")
    filename = local_assets_dir_reports_only / "analysis/pbmcs_unstimulated.report.json"
    dst = shutil.copyfile(filename, dir / filename.name)
    return Path(dst)


@pytest.fixture()
def analysis_stage_report_uropod(
    local_assets_dir_reports_only, pixelator_workdir
) -> Path:
    dir = pixelator_workdir.stage_dir("analysis")
    filename = local_assets_dir_reports_only / "analysis/uropod_control.report.json"
    dst = shutil.copyfile(filename, dir / filename.name)
    return Path(dst)


@pytest.fixture()
def analysis_stage_all_reports(
    analysis_stage_report_pbmcs, analysis_stage_report_uropod
) -> list[Path]:
    return [analysis_stage_report_pbmcs, analysis_stage_report_uropod]


@pytest.fixture()
def layout_stage_report_pbmcs(local_assets_dir_reports_only, pixelator_workdir) -> Path:
    dir = pixelator_workdir.stage_dir("layout")
    filename = local_assets_dir_reports_only / "layout/pbmcs_unstimulated.report.json"
    dst = shutil.copyfile(filename, dir / filename.name)
    return Path(dst)


@pytest.fixture()
def layout_stage_report_uropod(
    local_assets_dir_reports_only, pixelator_workdir
) -> Path:
    dir = pixelator_workdir.stage_dir("layout")
    filename = local_assets_dir_reports_only / "layout/uropod_control.report.json"
    dst = shutil.copyfile(filename, dir / filename.name)
    return Path(dst)


@pytest.fixture()
def layout_stage_all_reports(
    layout_stage_report_pbmcs, layout_stage_report_uropod
) -> list[Path]:
    return [layout_stage_report_pbmcs, layout_stage_report_uropod]


@pytest.fixture()
def all_stages_all_reports(
    amplicon_stage_all_reports,
    preqc_stage_all_reports,
    adapterqc_stage_all_reports,
    demux_stage_all_reports,
    collapse_stage_all_reports,
    graph_stage_all_reports,
    annotate_stage_all_reports,
    analysis_stage_all_reports,
) -> list[Path]:
    return list(
        itertools.chain.from_iterable(
            [
                amplicon_stage_all_reports,
                preqc_stage_all_reports,
                adapterqc_stage_all_reports,
                demux_stage_all_reports,
                collapse_stage_all_reports,
                graph_stage_all_reports,
                annotate_stage_all_reports,
                analysis_stage_all_reports,
            ]
        )
    )


@pytest.fixture()
def all_stages_all_reports_and_meta(local_assets_dir_reports_only, pixelator_workdir):
    shutil.copytree(
        str(local_assets_dir_reports_only),
        pixelator_workdir.basedir,
        dirs_exist_ok=True,
    )
    pixelator_workdir.scan()


@pytest.fixture()
def filtered_dataset_pxl_workdir(local_assets_dir, pixelator_workdir):
    pixelator_workdir.stage_dir("annotate")
    shutil.copy(
        str(local_assets_dir / "uropod_control.annotate.dataset.pxl"),
        pixelator_workdir.basedir / "annotate",
    )
    return pixelator_workdir


@pytest.fixture()
def raw_component_metrics_workdir(local_assets_dir, pixelator_workdir):
    shutil.copy(
        str(local_assets_dir / "uropod_control.raw_components_metrics.csv.gz"),
        pixelator_workdir.basedir / "annotate",
    )
    return pixelator_workdir


@pytest.fixture(scope="module")
def raw_component_metrics_data():
    test_file = (
        Path(__file__).parent / "assets/uropod_control.raw_components_metrics.csv.gz"
    )
    df = pd.read_csv(test_file, compression="gzip")
    return df


@pytest.fixture()
def filtered_dataset_pxl_data(filtered_dataset_pxl_workdir) -> PixelDataset:
    dataset_path = filtered_dataset_pxl_workdir.filtered_dataset()[0]
    dataset = PixelDataset.from_file(dataset_path)
    return dataset


@pytest.fixture(scope="session")
def full_run_assets_dir(request) -> Path:
    subprocess.run(
        "task tests:update-web-test-data",
        shell=True,
        cwd=str(request.config.rootdir),
        capture_output=True,
        check=True,
    )
    return Path(__file__).parent / "assets/full_run"


@pytest.fixture(scope="session")
def qc_report_assets_dir(request) -> Path:
    return Path(request.config.rootdir) / "tests/report/assets/qc_report_data"
