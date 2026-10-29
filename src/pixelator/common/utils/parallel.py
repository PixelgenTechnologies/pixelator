"""Utility functions for determining available CPU cores and building worker pools.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from logging.handlers import SocketHandler
from multiprocessing.pool import Pool

import click

logger = logging.getLogger(__name__)


def _add_handlers_to_root_logger(port, log_level):
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    socket_handler = SocketHandler("localhost", port)
    root_logger.addHandler(socket_handler)


# cgroup CPU bandwidth quota files. Container runtimes (``docker run --cpus``,
# Kubernetes ``limits.cpu``) enforce CPU limits through the CFS bandwidth quota,
# which leaves the CPU affinity mask untouched and is therefore invisible to
# ``os.process_cpu_count`` / ``os.sched_getaffinity``.
_CGROUP_V2_CPU_MAX = "/sys/fs/cgroup/cpu.max"
_CGROUP_V1_CFS_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
_CGROUP_V1_CFS_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"


def _get_affinity_cpu_count() -> int:
    """Return the number of CPU cores in the process' CPU affinity mask."""
    # os.process_cpu_count() (Python 3.13+) already accounts for CPU affinity.
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        count = process_cpu_count()
        if count:
            return count

    # os.sched_getaffinity() accounts for CPU affinity but is only available on
    # some Unix platforms (e.g. Linux); this covers the pre-3.13 pod use case.
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        try:
            return len(sched_getaffinity(0))
        except OSError:
            pass

    # Fall back to the total number of cores on the machine (e.g. on Windows).
    return os.cpu_count() or 1


def _read_cgroup_cpu_quota() -> int | None:
    """Return the CPU count allowed by the cgroup CPU bandwidth quota, if any.

    Reads the CFS quota that container runtimes use to enforce CPU limits, which
    the CPU affinity APIs do not observe. A fractional quota is rounded up and
    the result is never below 1. Unreadable or unexpected file contents are
    ignored.

    Returns:
        The number of allowed cores, or None if no quota applies or it cannot be
        determined (e.g. no cgroup CPU controller, or on non-Linux platforms).
    """
    quota = None
    period = None

    # cgroup v2: a single "<quota> <period>" line; "max" means no limit.
    try:
        with open(_CGROUP_V2_CPU_MAX) as fh:
            fields = fh.read().split()
        if len(fields) == 2:
            if fields[0] == "max":
                return None
            quota, period = int(fields[0]), int(fields[1])
    except (OSError, ValueError):
        quota = period = None

    # cgroup v1: quota and period live in separate files; a non-positive quota
    # means no limit.
    if quota is None or period is None:
        try:
            with open(_CGROUP_V1_CFS_QUOTA) as fh:
                quota = int(fh.read().strip())
            with open(_CGROUP_V1_CFS_PERIOD) as fh:
                period = int(fh.read().strip())
        except (OSError, ValueError):
            return None

    if quota is None or period is None or quota <= 0 or period <= 0:
        return None

    return max(1, math.ceil(quota / period))


def get_available_cpu_count() -> int:
    """Return the number of CPU cores available to the current process.

    Combines the process' CPU affinity mask with the cgroup CPU bandwidth quota
    and takes the most restrictive of the two, so that running inside a
    constrained environment such as a container or a scheduler pod does not
    report the host machine's core count. ``multiprocessing.cpu_count()`` reports
    the host core count regardless of the assigned cores, which leads to
    oversubscription and slower multiprocessing.

    Returns:
        The number of usable CPU cores, always at least 1.
    """
    cpu_count = _get_affinity_cpu_count()

    quota = _read_cgroup_cpu_quota()
    if quota is not None:
        cpu_count = min(cpu_count, quota)

    return max(1, cpu_count)


def _pre_multiprocessing_args(
    nbr_cores=None, logging_setup=None, context="spawn", **kwargs
):
    # If these variable are not set we will try to pick them
    # up from the click context
    current_click_context = click.get_current_context(silent=True)
    click_logging_setup = None
    click_nbr_cores = None
    if current_click_context:
        click_logging_setup = current_click_context.obj.get("LOGGER")
        click_nbr_cores = current_click_context.obj.get("CORES")

    nbr_cores = nbr_cores if nbr_cores else click_nbr_cores or get_available_cpu_count()
    args_dict = {
        "max_workers": nbr_cores,
        "mp_context": multiprocessing.get_context(context),
    }

    if logging_setup or click_logging_setup:
        args_dict = args_dict | dict(
            initializer=_add_handlers_to_root_logger,
            initargs=(
                (logging_setup or click_logging_setup).port,
                (logging_setup or click_logging_setup).log_level,
            ),
        )
    args_dict = args_dict | kwargs
    return args_dict


def get_process_pool_executor(
    nbr_cores=None, logging_setup=None, context="spawn", **kwargs
) -> ProcessPoolExecutor:
    """Return a ProcessPool with some default settings."""
    args_dict = _pre_multiprocessing_args(nbr_cores, logging_setup, context, **kwargs)
    return ProcessPoolExecutor(**args_dict)


def get_pool_executor(
    nbr_cores=None, logging_setup=None, context="spawn", **kwargs
) -> Pool:
    """Return a Pool with some default settings."""
    args_dict = _pre_multiprocessing_args(nbr_cores, logging_setup, context, **kwargs)
    nbr_of_processes = args_dict.pop("max_workers")
    args_dict.pop("mp_context")
    return multiprocessing.get_context(context).Pool(
        processes=nbr_of_processes, **args_dict
    )
