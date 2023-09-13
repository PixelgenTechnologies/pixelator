"""
This module exports functions to create a summary interactive web report.

Copyright (c) 2022 Pixelgen Technologies AB.
"""
from .builder import WebreportBuilder
from .collect import collect_report_data
from .types import InfoAndMetrics, Metrics, SampleInfo, WebreportData

__all__ = [
    "WebreportData",
    "InfoAndMetrics",
    "Metrics",
    "SampleInfo",
    "collect_report_data",
    "WebreportBuilder",
]
