"""
Tests for the plugin system is working

Copyright (c) 2022 Pixelgen Technologies AB.
"""
from importlib.metadata import EntryPoint
from unittest import mock

import click

from pixelator.cli.main import main_cli
from pixelator.cli.plugin import add_cli_plugins, fetch_cli_plugins
from pixelator.config import Config, config
from pixelator.config.plugin import fetch_config_plugins, load_config_plugins


@click.group()
def a_plugin():
    """A plugin to test loading on"""
    pass


def a_config_plugin(current_config: Config) -> Config:
    current_config.test_attr = True  # type: ignore[attr-defined]
    return current_config


def test_that_cli_plugins_are_loaded_in_main():
    with mock.patch(
        "pixelator.cli.plugin.fetch_cli_plugins",
        return_value=[
            EntryPoint(
                name="a_plugin",
                value="tests.test_plugin:a_plugin",
                group="pixelator.cli_plugin",
            )
        ],
    ):
        add_cli_plugins(main_cli)
        assert {"single-cell", "a-plugin"}.issubset(set(main_cli.commands.keys()))


def test_that_config_plugins_are_loaded_in_main():
    with mock.patch(
        "pixelator.config.plugin.fetch_config_plugins",
        return_value=[
            EntryPoint(
                name="a_config_plugin",
                value="tests.test_plugin:a_config_plugin",
                group="pixelator.config_plugin",
            )
        ],
    ):
        new_config = load_config_plugins(config)
        assert new_config.test_attr


class MockEntryPoints:
    pass


def test_fetch_cli_plugins():
    mock_entrypoints = MockEntryPoints()
    mock_entrypoints.select = mock.MagicMock(
        return_value=[
            EntryPoint(
                name="a_plugin",
                value="tests.test_plugin:a_plugin",
                group="pixelator.cli_plugin",
            )
        ]
    )

    with mock.patch("importlib.metadata.entry_points", return_value=mock_entrypoints):
        plugins = list(fetch_cli_plugins())
        assert len(plugins) == 1
        mock_entrypoints.select.assert_called_once()


def test_fetch_config_plugins():
    mock_entrypoints = MockEntryPoints()
    mock_entrypoints.select = mock.MagicMock(
        return_value=[
            EntryPoint(
                name="a_config_plugin",
                value="tests.test_plugin:a_config_plugin",
                group="pixelator.config_plugin",
            )
        ]
    )

    with mock.patch("importlib.metadata.entry_points", return_value=mock_entrypoints):
        plugins = list(fetch_config_plugins())
        assert len(plugins) == 1
        mock_entrypoints.select.assert_called_once()
