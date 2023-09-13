"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""

from pathlib import Path
from typing import Any

from ruamel import yaml

from pixelator.types import PathType


def load_yaml_file(path: PathType) -> Any:
    """
    Load an arbitrary yaml file.

    :param path: path to the yaml file
    :raises FileExistsError: If the path does not exist
    :raises TypeError: If the path is not a yaml file
    :returns: a yaml object
    """
    path = Path(path)
    if not path.is_file():
        raise FileExistsError(f"{path} is not a file")

    if not path.suffix == ".yaml":
        raise TypeError(f"{path} is not a yaml file")

    yaml_loader = yaml.YAML(typ="safe")
    with open(path, "r") as cf:
        data = yaml_loader.load(cf)

    return data
