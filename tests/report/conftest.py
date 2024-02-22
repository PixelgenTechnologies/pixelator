"""Test fixtures for reporting tests.

Copyright © 2022 Pixelgen Technologies AB.
"""

import itertools
import shutil
import uuid
from pathlib import Path

import pytest

from pixelator.report.common import PixelatorWorkdir


@pytest.fixture()
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
def filtered_datasets(local_assets_dir, pixelator_workdir):
    shutil.copy(
        str(local_assets_dir / "uropod_control_300k_S1_001.annotate.dataset.pxl"),
        pixelator_workdir.basedir / "annotate",
    )
    return pixelator_workdir


@pytest.fixture()
def raw_component_metrics(local_assets_dir, pixelator_workdir):
    shutil.copy(
        str(
            local_assets_dir
            / "uropod_control_300k_S1_001.raw_components_metrics.csv.gz"
        ),
        pixelator_workdir.basedir / "annotate",
    )
    return pixelator_workdir
