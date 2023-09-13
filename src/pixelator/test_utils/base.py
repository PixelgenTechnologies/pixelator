"""
Copyright (c) 2023 Pixelgen Technologies AB.
"""
from typing import Any, ClassVar, List

from click.testing import Result as CliRunnerResult


class BaseWorkflowTestMixin:
    """
    Base class for WorkflowTestMixin classes.

    These mixin classes are always used in conjunction with the PixlatorWorkflowTest.
    """

    __stage_key__: ClassVar[str]

    # Type hints for these methods that are injected through __init_subclass__.
    def __get_data(self, key: str) -> Any:
        """
        Try to retrieve a key from various data sources.

        Following sources are tried in order:

        1. Stage specific data
        2. Common data
        3. Test config data
        """
        ...

    def __get_parameters(self) -> List[str]:  # type: ignore
        """Retrieve the parameters for this stage."""
        ...

    def __get_options(self, key: str) -> Any:  # type: ignore
        """Retrieve a key from the options objects."""
        ...

    def __get_common_data(self, key: str) -> Any:  # type: ignore
        """Retrieve a key from the common data."""
        ...

    @property
    def __this_result(self) -> CliRunnerResult:  # type: ignore
        """Retrieve the execution result object for this stage."""
        ...

    @property
    def __this_command(self) -> List[str]:  # type: ignore
        """Retrieve the CLI object for this stage."""
        ...

    @property
    def __this_logs(self) -> str:  # type: ignore
        """Retrieve the logs of this stage if set."""
        ...

    def __init_subclass__(cls, **kwargs):
        """
        Some hackery to bind data access functions.
        """
        super().__init_subclass__(**kwargs)
        thisclass = BaseWorkflowTestMixin
        clsname = cls.__name__
        datakey = cls.__stage_key__

        def get_parameters_helper(instance):
            return thisclass.__base_get_parameters(instance, datakey)

        def get_data_helper(instance, key):
            return thisclass.__base_get_data(instance, key, data_key=datakey)

        def get_options_helper(instance, key):
            return thisclass.__base_get_options(instance, key)

        def get_common_options_helper(instance, key):
            return thisclass.__base_get_common_options(instance, key)

        setattr(cls, f"_{clsname}__get_data", get_data_helper)

        setattr(cls, f"_{clsname}__get_parameters", get_parameters_helper)

        setattr(cls, f"_{clsname}__get_options", get_options_helper)

        setattr(cls, f"_{clsname}__get_common_options", get_common_options_helper)

        def get_this_result(self):
            return self.context.results[datakey]

        def get_this_command(self):
            return self.context.commands[datakey]

        def get_this_logs(self):
            return self.context.logs[datakey]

        setattr(cls, f"_{clsname}__this_result", property(get_this_result))
        setattr(cls, f"_{clsname}__this_command", property(get_this_command))
        setattr(cls, f"_{clsname}__this_logs", property(get_this_logs))

    def __base_get_data(self, name: str, *, data_key=None) -> Any:
        """Retrieve undefined attributes from the class variable dicts."""
        return getattr(self, name, None)

    def __base_get_options(self, key: str):
        """
        Retrieve test config data from the class variables defined in the test class.
        """
        if not hasattr(self, "options"):
            raise ValueError("No options section found in test config.")

        options = getattr(self, "options")
        return options.get(key)

    def __base_get_common_options(self, key: str):
        if not hasattr(self, "options"):
            raise ValueError("No options section found in test config.")

        options = getattr(self, "options")
        common = options.get("common")
        return common or []

    def __base_get_parameters(self, stage):
        if not hasattr(self, "options"):
            raise ValueError("No options section found in test config.")

        options = getattr(self, "options")
        stage_data = options.get(stage)
        stage_params = stage_data.get("params") if stage_data else None

        return stage_params or []
