"""Tests for the PixelatorPNAReporting class not related to any specific pixelator stage.

Copyright © 2023 Pixelgen Technologies AB.
"""

import shutil
import uuid
from pathlib import Path

import pytest

from pixelator.pna.report.common import PixelatorPNAWorkdir


@pytest.fixture()
def pixelator_rundir(tmpdir_factory) -> Path:
    tmpdir = tmpdir_factory.mktemp(f"pixelator-{uuid.uuid4().hex}")
    return Path(tmpdir)


@pytest.fixture()
def pixelator_workdir(pixelator_rundir) -> PixelatorPNAWorkdir:
    return PixelatorPNAWorkdir(pixelator_rundir)


@pytest.fixture()
def all_stages_all_reports_and_meta(pixelator_workdir, full_run_dir):
    shutil.rmtree(pixelator_workdir.basedir, ignore_errors=True)
    res = shutil.copytree(full_run_dir, pixelator_workdir.basedir, dirs_exist_ok=True)
    return pixelator_workdir
