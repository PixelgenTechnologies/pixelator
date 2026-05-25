"""Copyright © 2023 Pixelgen Technologies AB."""

from pathlib import Path
from typing import Any

from ruamel import yaml

from pixelator.common.types import PathType


def load_yaml_file(path: PathType) -> Any:
    """Load an arbitrary yaml file.

    Args:
    path: path to the yaml file

    Raises:
    FileExistsError: If the path does not exist
    TypeError: If the path is not a yaml file
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
