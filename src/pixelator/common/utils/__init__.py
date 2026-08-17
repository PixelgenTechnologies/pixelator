"""Common functions and utilities for Pixelator.

Copyright © 2022 Pixelgen Technologies AB.
"""

from __future__ import annotations

import logging

from pixelator.common.utils.cli import (
    click_echo,
    log_step_start,
    timer,
    write_parameters_file,
)
from pixelator.common.utils.fastq import (
    R1_REGEX,
    R2_REGEX,
    get_read_sample_name,
    group_input_reads,
    is_read_file,
)
from pixelator.common.utils.iteration import batched, flatten, single_value
from pixelator.common.utils.parallel import (
    get_available_cpu_count,
    get_pool_executor,
    get_process_pool_executor,
)
from pixelator.common.utils.paths import (
    create_output_stage_dir,
    get_extension,
    get_sample_name,
    gz_size,
    sanity_check_inputs,
)
from pixelator.common.utils.sequences import reverse_complement
from pixelator.common.utils.serialization import np_encoder, remove_csv_whitespaces

logger = logging.getLogger(__name__)

__all__ = [
    "R1_REGEX",
    "R2_REGEX",
    "batched",
    "click_echo",
    "create_output_stage_dir",
    "flatten",
    "get_available_cpu_count",
    "get_extension",
    "get_pool_executor",
    "get_process_pool_executor",
    "get_read_sample_name",
    "get_sample_name",
    "group_input_reads",
    "gz_size",
    "is_read_file",
    "log_step_start",
    "logger",
    "np_encoder",
    "remove_csv_whitespaces",
    "reverse_complement",
    "sanity_check_inputs",
    "single_value",
    "timer",
    "write_parameters_file",
]
