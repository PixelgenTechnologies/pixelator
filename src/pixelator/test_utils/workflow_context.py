"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from click.testing import CliRunner, Result as CliRunnerResult

import pixelator.cli as pixelator_cli

# Do not propogate to parent logger since
logger = logging.getLogger("pixelator.test_utils")


class PixelatorWorkflowContext:
    """
    Helper class for workflow tests.

    This class is used to run the pixelator CLI and store the results.
    It also provides access to the test configuration in yaml or as class variables
    in the test class.
    """

    def __init__(self, test_id: str, workdir: Path):
        self.test_id = test_id
        self.workdir: Path = workdir
        self.results: Dict[str, Any] = {}
        # self.config: WorkflowConfig = config
        self.commands: Dict[str, List[str]] = {}
        self.logs: Dict[str, str] = {}

    def set_logs(self, key: str, logs: str):
        self.logs[key] = logs

    def get_logs(self, key: str) -> str:
        return self.logs[key]

    def run_command(
        self, key: str, command: List[str], link_files: Optional[List[str]] = None
    ) -> CliRunnerResult:
        runner = CliRunner()

        command = [str(s) for s in command]

        with pytest.MonkeyPatch.context() as mp:
            mp.chdir(self.workdir)

            if link_files:
                self._link_files(link_files)

            logger.info(
                f"[{self.test_id}] Running pixelator command: {' '.join(command)}"
            )
            result = runner.invoke(
                pixelator_cli.main_cli, command, catch_exceptions=False
            )

            logger.info(
                f"[{self.test_id}] Command completed [exit_code: {result.exit_code}]"
            )

            if result.exit_code != 0:
                logger.warning(f"[{self.test_id}] Command failed: {result.output}")

        self.results[key] = result
        self.commands[key] = command

        pixelator_logger = logging.getLogger("pixelator")
        # Reset handlers to make sure we do not get duplicate log messages
        # This is important, otherwise the old handlers will remain active
        # and logs for all stages will be written to the stage specific log files.
        pixelator_logger.handlers = []

        return result

    def _link_files(self, files: List[str], suffix: str = ""):
        """
        Symlink test files into the temporary working directory.

        If no suffix is given files will be linked directly into the working directory.

        :param files: list of files to link
        :param suffix: path relative to the workflow working directory to link
            the files in.
        """
        for f in files:
            f_path = Path(f)
            if not f_path.is_absolute():
                raise ValueError(f"File {f} is not absolute")

            if suffix:
                target = self.workdir / suffix / f_path.name
            else:
                target = self.workdir / f_path.name

            if not target.exists():
                if f_path.is_file():
                    target.symlink_to(f_path)
                if f_path.is_dir():
                    target.symlink_to(f_path, target_is_directory=True)


@pytest.fixture(scope="class")
def use_workflow_context(request, tmp_path_factory):
    """
    Fixture to set up a working directory and WorkflowContext helper.

    This fixture is class scoped so that the working directory is reused across
    individual pixelator commands in a single BasePixelatorWorkflow subclass.

    The working directory is cleaned up after the test class has run unless the
    `--keep_workdirs` is passed on the commandline, which can be useful for debugging.
    """
    keep_workdirs = request.config.getoption("--keep-workdirs", default=False)
    cls = request.cls
    classname = cls.__name__
    test_id = cls.test_id

    if not hasattr(cls, "context") or cls.context is None:
        logger.info(f"Setting up working directory for {classname}")
        job_root = tmp_path_factory.mktemp(classname)
        cls.context = PixelatorWorkflowContext(test_id, job_root)

    yield

    if not keep_workdirs:
        logger.info(f"Cleaning up working directory for {classname}")
        shutil.rmtree(cls.context.workdir)
