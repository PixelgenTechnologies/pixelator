"""
This module contains classes and functions related to
the configuration file for pixelator (assay settings).

Copyright (c) 2022 Pixelgen Technologies AB.
"""
from __future__ import annotations

import itertools
import typing
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import importlib_resources
import semver

from pixelator.config.assay import Assay
from pixelator.config.panel import AntibodyPanel
from pixelator.types import PathType

DNA_CHARS = {"A", "C", "G", "T"}

RangeType = typing.TypeVar(
    "RangeType", Tuple[int, int], Tuple[Optional[int], Optional[int]]
)


class Config:
    """
    Class containing the pixelator configuration (assay settings)
    """

    def __init__(
        self,
        assays: Optional[List[Assay]] = None,
        panels: Optional[List[AntibodyPanel]] = None,
    ) -> None:
        self.assays: Dict[str, Assay] = {}
        self.panels: typing.MutableMapping[str, List[AntibodyPanel]] = defaultdict(list)

        if assays is not None:
            self.assays.update({a.name: a for a in assays})

        if panels is not None:
            for p in panels:
                key = p.name if p.name is not None else str(p.filename)
                self.panels[key].append(p)

    def load_assay(self, path: PathType) -> None:
        """
        Load an assay from a yaml file.
        """
        assay = Assay.from_yaml(path)
        self.assays[assay.name] = assay

    def load_panel_file(self, path: PathType) -> None:
        panel = AntibodyPanel.from_csv(path)
        key = panel.name if panel.name is not None else str(panel.filename)
        self.panels[key].append(panel)

    def load_assays(self, path: PathType):
        """
        Load all assays from a directory containing yaml files.
        """
        search_path = Path(path)

        yaml_files = list(
            itertools.chain(
                search_path.glob("*.yaml"),
                search_path.glob("*.yml"),
            )
        )

        for f in yaml_files:
            self.load_assay(f)

    def load_panels(self, path: PathType):
        """
        Load all panel files from a directory containing csv files.
        """
        search_path = Path(path)

        csv_files = list(search_path.glob("*.csv"))

        for f in csv_files:
            self.load_panel_file(f)

    def get_assay(self, assay_name: str) -> Optional[Assay]:
        """
        Get an assay by name.
        """
        return self.assays.get(assay_name)

    def get_panel(
        self, panel_name: str, version: Optional[str] = None
    ) -> Optional[AntibodyPanel]:
        """
        Get a panel by name.
        """
        panels_with_key = self.panels.get(panel_name)
        if panels_with_key is None:
            return None

        def keyfunc(p):
            version = p.version
            if version is None:
                v = semver.Version.parse("0.0.0")

            v = semver.Version.parse(version)
            return v

        panels_with_key = sorted(panels_with_key, key=keyfunc, reverse=True)
        if version is None:
            return panels_with_key[0]

        selected_panel = next(
            (p for p in panels_with_key if p.version == version), None
        )
        return selected_panel


def load_assays_package(config: Config, package_name: str) -> Config:
    """
    Load default assays from a resources package.

    :param config: The config object to load assays into
    :param package_name: The name of the package to load assays from
    """
    # TODO: Consider switching to base importlib.resources after
    #       dropping python3.8 support.
    for resource in importlib_resources.files(package_name).iterdir():
        if resource.is_file():
            with importlib_resources.as_file(resource) as file_path:
                config.load_assay(file_path)

    return config


def load_panels_package(config: Config, package_name: str) -> Config:
    """
    Load default panels from a resources package.

    :param config: The config object to load panel files into
    :param package_name: The name of the package to load panels from
    """
    # TODO: Consider switching to base importlib.resources after
    #       dropping python3.8 support.
    for resource in importlib_resources.files(package_name).iterdir():
        if resource.is_file():
            with importlib_resources.as_file(resource) as file_path:
                config.load_panel_file(file_path)

    return config
