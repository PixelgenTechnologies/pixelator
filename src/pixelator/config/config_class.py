"""Module contains classes and functions related to the configuration file for pixelator (assay settings).

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import itertools
import typing
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import importlib_resources
import semver

from pixelator.config.assay import Assay
from pixelator.config.panel import AntibodyPanel
from pixelator.exceptions import PixelatorBaseException
from pixelator.types import PathType

DNA_CHARS = {"A", "C", "G", "T"}

RangeType = typing.TypeVar(
    "RangeType", Tuple[int, int], Tuple[Optional[int], Optional[int]]
)


class PanelException(PixelatorBaseException):
    """Exception raised for failures to load a panel into the global configuration."""

    pass


class Config:
    """Class containing the pixelator configuration (assay settings)."""

    def __init__(
        self,
        assays: Optional[List[Assay]] = None,
        panels: Optional[List[AntibodyPanel]] = None,
    ) -> None:
        """Initialize the config object."""
        self.assays: Dict[str, Assay] = {}
        self.panels: typing.MutableMapping[str, List[AntibodyPanel]] = defaultdict(list)
        self.panel_aliases: Dict[str, str] = {}

        if assays is not None:
            self.assays.update({a.name: a for a in assays})

        if panels is not None:
            for p in panels:
                key = p.name if p.name is not None else str(p.filename)
                self.panels[key].append(p)

    def load_assay(self, path: PathType) -> None:
        """Load an assay from a yaml file."""
        assay = Assay.from_yaml(path)
        self.assays[assay.name] = assay

    def load_panel_file(self, path: PathType) -> None:
        """Load the panel file.

        :param path: The path to the panel file.
        :raises PanelException: If the panel alias already exists in the config.
        """
        panel = AntibodyPanel.from_csv(path)
        key = panel.name if panel.name is not None else str(panel.filename)
        self.panels[key].append(panel)

        # Enable panel lookup by aliases
        for alias in panel.aliases:
            if (alias in self.panel_aliases) and (key != self.panel_aliases[alias]):
                raise PanelException(
                    f'Panel alias "{alias}" already exists in the '
                    f'config for panel "{self.panel_aliases[alias]}".'
                    "If you provided your own panel file, please "
                    "ensure the panel name an aliases are unique "
                    "in the header of the file."
                )
                continue

            self.panel_aliases[alias] = key

    def load_assays(self, path: PathType):
        """Load all assays from a directory containing yaml files."""
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
        """Load all panel files from a directory containing csv files."""
        search_path = Path(path)

        csv_files = list(search_path.glob("*.csv"))

        for f in csv_files:
            self.load_panel_file(f)

    def get_assay(self, assay_name: str) -> Optional[Assay]:
        """Get an assay by name."""
        return self.assays.get(assay_name)

    def list_panel_names(self, include_aliases: bool = False) -> List[str]:
        """Return a list of all panel names.

        :param include_aliases: Include panel aliases in the list
        :returns: A list of panel names
        """
        out = sorted(list(self.panels.keys()))

        if not include_aliases:
            return out

        out += sorted(list(self.panel_aliases.keys()))
        return out

    def get_panel(
        self,
        panel_name: str,
        version: Optional[str] = None,
        allow_aliases: bool = True,
    ) -> Optional[AntibodyPanel]:
        """Get a panel by name.

        :param panel_name: The name of the panel
        :param version: The optional version of a panel to return
        :param allow_aliases: Allow panel aliases to be used
        """
        panels_with_key = self.panels.get(panel_name)

        # Try to load using an alias if no name matches are found
        if panels_with_key is None and allow_aliases:
            panel_alias = self.panel_aliases.get(panel_name)
            if panel_alias is not None:
                panels_with_key = self.panels.get(panel_alias)

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
    """Load default assays from a resources package.

    :param config: The config object to load assays into
    :param package_name: The name of the package to load assays from
    :return: The updated config object
    """
    # TODO: Consider switching to base importlib.resources after
    #       dropping python3.8 support.
    for resource in importlib_resources.files(package_name).iterdir():
        if resource.is_file():
            with importlib_resources.as_file(resource) as file_path:
                config.load_assay(file_path)

    return config


def load_panels_package(config: Config, package_name: str) -> Config:
    """Load default panels from a resources package.

    :param config: The config object to load panel files into
    :param package_name: The name of the package to load panels from
    :return: The updated config object
    """
    # TODO: Consider switching to base importlib.resources after
    #       dropping python3.8 support.
    for resource in importlib_resources.files(package_name).iterdir():
        if resource.is_file():
            with importlib_resources.as_file(resource) as file_path:
                config.load_panel_file(file_path)

    return config
