"""Entrypoint for the native edgelist module.

The stage itself lives in Rust and is driven through
`pixelator_core.run_edgelist`; this package holds the Python-side
reporting that turns the statistics it returns into a sample report.

Copyright © 2026 Pixelgen Technologies AB.
"""

from .report import EdgelistSampleReport, Q30Statistics

__all__ = [
    "EdgelistSampleReport",
    "Q30Statistics",
]
