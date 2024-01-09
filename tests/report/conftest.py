import itertools
import json
import shutil
import uuid

import pytest
from pathlib import Path
import uuid

from pixelator.report.workdir import PixelatorWorkdir
from pixelator.report import PixelatorReporting


@pytest.fixture()
def local_assets_dir() -> Path:
    return Path(__file__).parent / "assets"


@pytest.fixture()
def pixelator_rundir(tmpdir_factory) -> Path:
    tmpdir = tmpdir_factory.mktemp(f"pixelator-{uuid.uuid4().hex}")
    return Path(tmpdir)


@pytest.fixture()
def pixelator_workdir(pixelator_rundir) -> PixelatorWorkdir:
    return PixelatorWorkdir(pixelator_rundir)


@pytest.fixture()
def amplicon_stage_report_pbmcs(pixelator_rundir) -> Path:
    amplicon_dir = pixelator_rundir / "amplicon"
    amplicon_dir.mkdir(exist_ok=True, parents=True)

    filepath = amplicon_dir / "pbmcs_unstimulated.report.json"
    with open(filepath, "w") as fp:
        json.dump(
            {
                "phred_result": {
                    "fraction_q30": 0.9606927317923855,
                    "fraction_q30_bc": 0.9660959533395918,
                    "fraction_q30_pbs1": 0.9683133231643615,
                    "fraction_q30_pbs2": 0.9477632357943524,
                    "fraction_q30_umi": 0.965813230797693,
                    "fraction_q30_upia": 0.9602759816852228,
                    "fraction_q30_upib": 0.9671804579646854,
                }
            },
            fp,
        )

    return filepath


@pytest.fixture()
def amplicon_stage_report_uropod(pixelator_rundir) -> Path:
    amplicon_dir = pixelator_rundir / "amplicon"
    amplicon_dir.mkdir(exist_ok=True, parents=True)

    filepath = amplicon_dir / "uropod_control.report.json"
    with open(filepath, "w") as fp:
        json.dump(
            {
                "phred_result": {
                    "fraction_q30": 0.9602150898560277,
                    "fraction_q30_bc": 0.9624244207098193,
                    "fraction_q30_pbs1": 0.9721885264356955,
                    "fraction_q30_pbs2": 0.9535804849402897,
                    "fraction_q30_umi": 0.9627836138165338,
                    "fraction_q30_upia": 0.9523661337597942,
                    "fraction_q30_upib": 0.9642700474644683,
                }
            },
            fp,
        )

    return filepath


@pytest.fixture()
def amplicon_stage_all_reports(
    amplicon_stage_report_pbmcs, amplicon_stage_report_uropod
) -> list[Path]:
    return [amplicon_stage_report_pbmcs, amplicon_stage_report_uropod]


@pytest.fixture()
def preqc_stage_report_pbmcs(local_assets_dir, pixelator_workdir) -> Path:
    preqc_dir = pixelator_workdir.stage_dir("preqc")
    filename = local_assets_dir / "preqc/pbmcs_unstimulated.report.json"
    dst = shutil.copyfile(filename, preqc_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def preqc_stage_report_uropod(local_assets_dir, pixelator_workdir) -> Path:
    preqc_dir = pixelator_workdir.stage_dir("preqc")
    filename = local_assets_dir / "preqc/uropod_control.report.json"
    dst = shutil.copyfile(filename, preqc_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def preqc_stage_all_reports(
    preqc_stage_report_pbmcs, preqc_stage_report_uropod
) -> list[Path]:
    return [preqc_stage_report_pbmcs, preqc_stage_report_uropod]


@pytest.fixture()
def adapterqc_stage_report_pbmcs(local_assets_dir, pixelator_workdir) -> Path:
    preqc_dir = pixelator_workdir.stage_dir("adapterqc")
    filename = local_assets_dir / "adapterqc/pbmcs_unstimulated.report.json"
    dst = shutil.copyfile(filename, preqc_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def adapterqc_stage_report_uropod(local_assets_dir, pixelator_workdir) -> Path:
    preqc_dir = pixelator_workdir.stage_dir("adapterqc")
    filename = local_assets_dir / "adapterqc/uropod_control.report.json"
    dst = shutil.copyfile(filename, preqc_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def adapterqc_stage_all_reports(
    adapterqc_stage_report_pbmcs, adapterqc_stage_report_uropod
) -> list[Path]:
    return [adapterqc_stage_report_pbmcs, adapterqc_stage_report_uropod]


@pytest.fixture()
def demux_stage_report_pbmcs(local_assets_dir, pixelator_workdir) -> Path:
    preqc_dir = pixelator_workdir.stage_dir("demux")
    filename = local_assets_dir / "demux/pbmcs_unstimulated.report.json"
    dst = shutil.copyfile(filename, preqc_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def demux_stage_report_uropod(local_assets_dir, pixelator_workdir) -> Path:
    preqc_dir = pixelator_workdir.stage_dir("demux")
    filename = local_assets_dir / "demux/uropod_control.report.json"
    dst = shutil.copyfile(filename, preqc_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def demux_stage_all_reports(
    demux_stage_report_pbmcs, demux_stage_report_uropod
) -> list[Path]:
    return [demux_stage_report_pbmcs, demux_stage_report_uropod]


@pytest.fixture()
def collapse_stage_report_pbmcs(local_assets_dir, pixelator_workdir) -> Path:
    collapse_dir = pixelator_workdir.stage_dir("collapse")
    filename = local_assets_dir / "collapse/pbmcs_unstimulated.report.json"
    dst = shutil.copyfile(filename, collapse_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def collapse_stage_report_uropod(local_assets_dir, pixelator_workdir) -> Path:
    collapse_dir = pixelator_workdir.stage_dir("collapse")
    filename = local_assets_dir / "collapse/uropod_control.report.json"
    dst = shutil.copyfile(filename, collapse_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def collapse_stage_all_reports(
    collapse_stage_report_pbmcs, collapse_stage_report_uropod
) -> list[Path]:
    return [collapse_stage_report_pbmcs, collapse_stage_report_uropod]


@pytest.fixture()
def graph_stage_report_pbmcs(local_assets_dir, pixelator_workdir) -> Path:
    preqc_dir = pixelator_workdir.stage_dir("graph")
    filename = local_assets_dir / "graph/pbmcs_unstimulated.report.json"
    dst = shutil.copyfile(filename, preqc_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def graph_stage_report_uropod(local_assets_dir, pixelator_workdir) -> Path:
    preqc_dir = pixelator_workdir.stage_dir("graph")
    filename = local_assets_dir / "graph/uropod_control.report.json"
    dst = shutil.copyfile(filename, preqc_dir / filename.name)
    return Path(dst)


@pytest.fixture()
def graph_stage_all_reports(
    graph_stage_report_pbmcs, graph_stage_report_uropod
) -> list[Path]:
    return [graph_stage_report_pbmcs, graph_stage_report_uropod]


@pytest.fixture()
def annotate_stage_report_pbmcs(local_assets_dir, pixelator_workdir) -> Path:
    dir = pixelator_workdir.stage_dir("annotate")
    filename = local_assets_dir / "annotate/pbmcs_unstimulated.report.json"
    dst = shutil.copyfile(filename, dir / filename.name)
    return Path(dst)


@pytest.fixture()
def annotate_stage_report_uropod(local_assets_dir, pixelator_workdir) -> Path:
    dir = pixelator_workdir.stage_dir("annotate")
    filename = local_assets_dir / "annotate/uropod_control.report.json"
    dst = shutil.copyfile(filename, dir / filename.name)
    return Path(dst)


@pytest.fixture()
def annotate_stage_all_reports(
    annotate_stage_report_pbmcs, annotate_stage_report_uropod
) -> list[Path]:
    return [annotate_stage_report_pbmcs, annotate_stage_report_uropod]


@pytest.fixture()
def analysis_stage_report_pbmcs(local_assets_dir, pixelator_workdir) -> Path:
    dir = pixelator_workdir.stage_dir("analysis")
    filename = local_assets_dir / "analysis/pbmcs_unstimulated.report.json"
    dst = shutil.copyfile(filename, dir / filename.name)
    return Path(dst)


@pytest.fixture()
def analysis_stage_report_uropod(local_assets_dir, pixelator_workdir) -> Path:
    dir = pixelator_workdir.stage_dir("analysis")
    filename = local_assets_dir / "analysis/uropod_control.report.json"
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
    return itertools.chain.from_iterable(
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
