"""Functions to create an interactive qc report.

Copyright © 2022 Pixelgen Technologies AB.
"""

from .builder import QCReportBuilder
from .collect import collect_report_data
from .main import create_dynamic_report, make_report
from .types import InfoAndMetrics, Metrics, QCReportData, SampleInfo

__all__ = [
    "QCReportData",
    "InfoAndMetrics",
    "Metrics",
    "SampleInfo",
    "collect_report_data",
    "QCReportBuilder",
    "create_dynamic_report",
    "make_report",
]
