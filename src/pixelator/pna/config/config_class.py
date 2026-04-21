"""Module contains classes and functions related to the configuration file for pixelator (assay settings).

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import importlib
import importlib.resources
import itertools
import re
import typing
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import semver
from packaging.specifiers import SpecifierSet

from pixelator.common.config.config_class import Config, PanelException
from pixelator.common.types import PathType
from pixelator.common.utils import logger
from pixelator.pna.config.assay import PNAAssay
from pixelator.pna.config.panel import PNAAntibodyPanel

DNA_CHARS = {"A", "C", "G", "T"}

RangeType = typing.TypeVar(
    "RangeType", Tuple[int, int], Tuple[Optional[int], Optional[int]]
)


class PNAConfig:
    """Class containing the pixelator configuration (assay settings)."""

    def __init__(
        self,
        assays: Optional[List[PNAAssay]] = None,
        panels: Optional[List[PNAAntibodyPanel]] = None,
    ) -> None:
        """Initialize the config object."""
        self.assays: Dict[str, PNAAssay] = {}
        self.panels: typing.MutableMapping[str, List[PNAAntibodyPanel]] = defaultdict(
            list
        )
        self.panel_aliases: Dict[str, str] = {}
        self.products: typing.MutableMapping[str, List[PNAAntibodyPanel]] = defaultdict(
            list
        )

        if assays is not None:
            self.assays.update({a.name: a for a in assays})

        if panels is not None:
            for p in panels:
                self.add_panel(p)

    def load_assay(self, path: PathType) -> None:
        """Load an assay from a yaml file."""
        assay = PNAAssay.from_yaml(path)
        if assay.name in self.assays:
            raise ValueError(
                f"Assay with name {assay.name} ({str(path)}) already exists in the config. "
            )
        self.assays[assay.name] = assay

    def load_panel_file(self, path: PathType) -> None:
        """Load the panel file.

        :param path: The path to the panel file.
        :raises PanelException: If the panel alias already exists in the config.
        """
        panel = PNAAntibodyPanel.from_csv(path)
        self.add_panel(panel)

    def add_panel(self, panel: PNAAntibodyPanel) -> None:
        """Add a panel to the config."""
        key = panel.name if panel.name is not None else str(panel.filename)
        self.panels[key].append(panel)

        # allow to also get panel by product name if provided in the panel file
        if panel.product is not None:
            self.products[panel.product].append(panel)

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
            self.panel_aliases[alias] = key
            # TODO: add test for panel with non uniqe alias to see we hit this error

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

    def get_assay(self, assay_name: str) -> Optional[PNAAssay]:
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
    ) -> Optional[PNAAntibodyPanel]:
        """Get a panel by name.

        :param panel_name: The name of the panel
        :param version: The optional version of a panel to return
        :param allow_aliases: Allow panel aliases to be used
        """
        version_stripped_name, specified_version = parse_versioned_panel_name(
            panel_name
        )
        if version is not None and specified_version is not None:
            raise ValueError(
                "Version specified both in panel_name and as a separate argument. "
                + "Please specify the version in only one place."
            )

        # First try to load the provided panel_name
        panels_with_key = self.panels.get(panel_name)
        # try to load using the version stripped name if the panel name contains a version specifier
        if panels_with_key is None and version_stripped_name is not None:
            panels_with_key = self.panels.get(version_stripped_name)

        # try to load the provided panel_name as a product name if no panel name matches are found
        if panels_with_key is None:
            panels_with_key = self.products.get(panel_name)
        if panels_with_key is None and version_stripped_name is not None:
            panels_with_key = self.products.get(version_stripped_name)

        # Try to load using an alias if no name matches are found
        if panels_with_key is None and allow_aliases:
            logger.debug(
                'No panel found with name "%s". Trying to find it among panel aliases...',
                panel_name,
            )
            panel_alias = self.panel_aliases.get(panel_name)
            if panel_alias is None and version_stripped_name is not None:
                panel_alias = self.panel_aliases.get(version_stripped_name)
            if panel_alias is not None:
                panels_with_key = self.panels.get(panel_alias)

        if panels_with_key is None:
            return None

        # if the version is specified, filter the panels
        if specified_version or version:
            panel_versions = set(
                SpecifierSet(specified_version or "==" + version).filter(  # type: ignore
                    [p.version for p in panels_with_key]
                )
            )
            panels_with_key = [
                p for p in panels_with_key if p.version in panel_versions
            ]
            if not panels_with_key:
                return None

        # If there are multiple panels with the same name and version, raise an error
        if len(set(p.version.split(".")[0] for p in panels_with_key)) > 1:
            raise ValueError(
                f"Multiple major versions found for panel {panel_name}. "
                + "Please specify the major and minor version in the panel name or "
                + "alias to disambiguate."
            )

        # If there are multiple panels with the same name and major version but different minor
        # versions, raise a error and promt for more fine grained specification
        elif len(set(p.version.split(".")[1] for p in panels_with_key)) > 1:
            raise ValueError(
                f"Multiple minor versions found for panel {panel_name}. "
                + "Refusing to automatically select the latest out of multiple minor versions. "
                + "Minor versions usually means there was a change in clones used for one or "
                + "more markers. Panels might not be fully compatible!\n"
                + "Please specify the minor version in the panel name or "
                + "alias to disambiguate.",
            )

        def keyfunc(p):
            version = p.version
            if version is None:
                v = semver.Version.parse("0.0.0")

            v = semver.Version.parse(version)
            return v

        panels_with_key = sorted(panels_with_key, key=keyfunc, reverse=True)
        return panels_with_key[0]


ConfigType = typing.TypeVar("ConfigType", Config, PNAConfig)


def load_assays_package(config: ConfigType, package_name: str) -> ConfigType:
    """Load default assays from a resources package.

    :param config: The config object to load assays into
    :param package_name: The name of the package to load assays from
    :return: The updated config object
    """
    for resource in importlib.resources.files(package_name).iterdir():
        if resource.is_file():
            with importlib.resources.as_file(resource) as file_path:
                config.load_assay(file_path)

    return config


def load_panels_package(config: ConfigType, package_name: str) -> ConfigType:
    """Load default panels from a resources package.

    :param config: The config object to load panel files into
    :param package_name: The name of the package to load panels from
    :return: The updated config object
    """
    for resource in importlib.resources.files(package_name).iterdir():
        if resource.is_file():
            with importlib.resources.as_file(resource) as file_path:
                config.load_panel_file(file_path)

    return config


def parse_versioned_panel_name(panel_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse a panel name that may contain a version specifier.

    :param panel_name: The panel name to parse
    :return: A tuple containing the stripped panel name and the version specifier (if any)
    """
    if match := re.search(
        # here we allow panel names matching [A-Za-z0-9-.]+)
        # followed by a version specifier ==, >=, <= ... etc
        "^(?P<name>[A-Za-z0-9-.]+)(?P<spec>([<>=]{1,2}))(?P<major>\d)(?P<minor>\.\d)?(?P<patch>\.\d)?$",
        panel_name,
    ):
        version_stripped_name = match.group("name")
        specified_version = (
            match.group("spec")
            + match.group("major")
            + (
                match.group("minor")
                or (
                    ".*"
                    if ">" not in match.group("spec") and "<" not in match.group("spec")
                    else ""
                )
            )
            + (
                match.group("patch")
                or (
                    ".*"
                    if match.group("minor")
                    and ">" not in match.group("spec")
                    and "<" not in match.group("spec")
                    else ""
                )
            )
        )
        logger.debug(
            'Parsed panel name "%s" into name "%s" and version specifier "%s".',
            panel_name,
            version_stripped_name,
            specified_version,
        )
    else:
        version_stripped_name = None
        specified_version = None

    return version_stripped_name, specified_version
