"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""
from pathlib import Path
from typing import Any, Dict

import ruamel.yaml as yaml

from pixelator.types import PathType


class WorkflowConfig:
    """
    Class used to load and query the workflow tests configuration file.
    """

    def __init__(self, config_file: PathType):
        self.config_file = config_file
        self._config = self._parse(self.config_file)

        self._validate()

    def get_test_config(self, test_name: str) -> Any:
        """
        Retrieve config data for a specific test case.

        Currently only "small", "single-cell" and "tissue" are implemented.
        You can use arbitrary names for your own tests, and then link your
        :class:`PixelatorWorkfklowTest` subclass to the test name in the config file
        by defining the class variable :attr:`test_id` on the subclass.

        :param test_name: name of the test case as given in the config.
        :returns: the config object for the test case
        """
        return self._config[test_name]

    def keys(self):
        return self._config.keys()

    @classmethod
    def _parse(cls, config_file: PathType) -> Dict[str, Any]:
        """
        Load and resolve relative paths in the config file.

        :param config_file: path to the config file
        :returns: the config object parsed from the `config_file`
        """
        yaml_loader = yaml.YAML(typ="safe")
        with open(str(config_file), "r") as f:
            raw_config = yaml_loader.load(f)

        config = cls._resolve_relative_paths(Path(config_file), raw_config)
        return config

    @classmethod
    def _resolve_relative_paths(cls, config_file: Path, config: Dict[str, Any]):
        """
        Resolve relative paths in the config file.

        All relative paths in the config file are resolved relative to the parent
        directory of the config file.

        :param config_file: path to the config file
        :param config: the config object parsed from the `config_file`
        :returns: the config object with relative paths resolved
        """
        for test_id, test_config in config.items():
            panel_file = test_config.get("panel_file")
            if panel_file:
                test_config["panel_file"] = (config_file.parent / panel_file).resolve()

            input_files = test_config.get("input_files")
            if input_files:
                new_input_files = [
                    (config_file.parent / f).resolve() for f in input_files
                ]
                test_config["input_files"] = new_input_files

        return config

    def _validate(self):
        cfg = self._config
        for wf_name, wf in cfg.items():
            if wf.get("panel_file") and wf.get("panel"):
                raise ValueError(
                    f"Both `panel` and `panel_file` are defined for "
                    f"workflow {wf_name}. Only one can be set."
                )
