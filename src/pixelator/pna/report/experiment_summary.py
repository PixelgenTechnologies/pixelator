"""Experiment summary for a pixelator single-cell-pna run.

The summary keeps every sample, including ones represented by a null pxl
file. Statistics from steps that ran before a recoverable failure stay in
the summary, and each step is marked passed, failed, or not run.

Copyright © 2026 Pixelgen Technologies AB.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

import pydantic

from pixelator.pna.pixeldataset.io import PxlFile

PIPELINE_STEPS: tuple[str, ...] = (
    "amplicon",
    "demux",
    "collapse",
    "graph",
    "denoise",
    "analysis",
    "layout",
    "sample_calling",
)

# Partial or aggregate reports that should not define a sample on their own.
_SKIP_REPORT_TYPES = frozenset({"collapse-umi", "sample_calling_total"})


class StepSummary(pydantic.BaseModel):
    """Pass/fail record for one pipeline step of one sample."""

    step: str
    status: Literal["passed", "failed", "not_run"]
    null_reason: str | None = None
    statistics: dict | None = None


class SampleExperimentSummary(pydantic.BaseModel):
    """Per-sample view of which steps passed or failed."""

    sample_id: str
    status: Literal["passed", "failed"]
    null_reason: str | None = None
    steps: list[StepSummary]

    def describe(self) -> str:
        """Return a one-line pass/fail description of the steps that ran."""
        parts: list[str] = []
        for step in self.steps:
            if step.status == "not_run":
                continue
            if step.status == "failed":
                parts.append(f"{step.step}: failed ({step.null_reason})")
            else:
                parts.append(f"{step.step}: passed")
        return "; ".join(parts)


class ExperimentSummary(pydantic.BaseModel):
    """Summary of every sample found in a pixelator output folder."""

    samples: list[SampleExperimentSummary]

    def write_json_file(self, path: Path) -> None:
        """Write the summary as JSON.

        Args:
            path: Destination path. Parent directories are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=4))


def _load_report(path: Path) -> dict | None:
    try:
        with path.open() as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _choose_report(sample_id: str, candidates: list[tuple[Path, dict]]) -> dict:
    """Prefer the canonical ``{sample}.report.json`` over partial reports."""
    exact_name = f"{sample_id}.report.json"
    exact = [data for path, data in candidates if path.name == exact_name]
    if exact:
        return exact[0]
    non_partial = [
        data
        for path, data in candidates
        if "part_" not in path.name
        and data.get("report_type") not in _SKIP_REPORT_TYPES
    ]
    if non_partial:
        return non_partial[0]
    return candidates[0][1]


def build_experiment_summary(workdir: Path) -> ExperimentSummary:
    """Collect step reports and null pxl files from a pixelator output folder.

    Args:
        workdir: Directory that contains stage folders such as ``graph`` and
            ``denoise``.

    Returns:
        An experiment summary covering every sample found.
    """
    workdir = Path(workdir)
    reports: dict[str, dict[str, dict]] = defaultdict(dict)
    null_reasons: dict[str, dict[str, str]] = defaultdict(dict)

    for step in PIPELINE_STEPS:
        stage_dir = workdir / step
        if not stage_dir.is_dir():
            continue

        grouped: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
        for report_path in sorted(stage_dir.glob("*.report.json")):
            data = _load_report(report_path)
            if data is None:
                continue
            if data.get("report_type") in _SKIP_REPORT_TYPES:
                continue
            sample_id = data.get("sample_id")
            if not sample_id:
                continue
            grouped[str(sample_id)].append((report_path, data))
        for sample_id, candidates in grouped.items():
            reports[sample_id][step] = _choose_report(sample_id, candidates)

        for pxl_path in sorted(stage_dir.glob("*.pxl")):
            pxl_file = PxlFile(pxl_path)
            if not pxl_file.is_null():
                continue
            reason = pxl_file.null_reason()
            if not reason:
                raise ValueError(
                    f"{pxl_path} is a null pxl file but has no reason. "
                    "This is not a recoverable data error."
                )
            null_reasons[pxl_file.sample_name][step] = reason
            reports.setdefault(pxl_file.sample_name, {})

    samples: list[SampleExperimentSummary] = []
    for sample_id in sorted(reports):
        step_summaries: list[StepSummary] = []
        sample_failed = False
        sample_reason: str | None = None
        for step in PIPELINE_STEPS:
            report = reports[sample_id].get(step)
            pxl_reason = null_reasons.get(sample_id, {}).get(step)
            if report is None and pxl_reason is None:
                step_summaries.append(StepSummary(step=step, status="not_run"))
                continue
            report_status = (report or {}).get("status", "passed")
            reason = pxl_reason or (report or {}).get("null_reason")
            if pxl_reason or report_status == "failed":
                status: Literal["passed", "failed"] = "failed"
                sample_failed = True
                if sample_reason is None and reason:
                    sample_reason = str(reason)
            else:
                status = "passed"
                reason = None
            step_summaries.append(
                StepSummary(
                    step=step,
                    status=status,
                    null_reason=reason,
                    statistics=report,
                )
            )
        samples.append(
            SampleExperimentSummary(
                sample_id=sample_id,
                status="failed" if sample_failed else "passed",
                null_reason=sample_reason,
                steps=step_summaries,
            )
        )
    return ExperimentSummary(samples=samples)


def write_experiment_summary(workdir: Path, output_path: Path) -> ExperimentSummary:
    """Build and write an experiment summary.

    Args:
        workdir: Pixelator output folder to read.
        output_path: JSON file to write.

    Returns:
        The summary that was written.
    """
    summary = build_experiment_summary(workdir)
    summary.write_json_file(output_path)
    return summary
