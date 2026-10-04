from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import atomic_json

BASE_METRICS = (
    "validation_loss",
    "validation_perplexity",
)
CURVATURE_METRICS = (
    "hessian_trace",
    "hessian_lambda_max",
)
FINAL_METRICS = (*BASE_METRICS, *CURVATURE_METRICS)
TRAJECTORY_METRICS = FINAL_METRICS


def write_report(
    output_dir: Path,
    endpoint_records: Sequence[Mapping[str, Any]],
    trajectory_records: Sequence[Mapping[str, Any]],
    *,
    branch_order: Sequence[str],
    fork_step: int,
) -> dict[str, Any]:
    """Write dynamic tables and figures for every metric present in the records.

    The roster is supplied by the endpoint manifest and configuration;
    this module deliberately has no method, branch, or endpoint-count constants.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    endpoints = [dict(record) for record in endpoint_records]
    trajectory = _expanded_trajectory(
        trajectory_records,
        branch_order=branch_order,
        fork_step=int(fork_step),
    )
    atomic_json(output_dir / "final_results.json", endpoints)
    atomic_json(output_dir / "trajectory_results.json", trajectory)
    _write_csv(output_dir / "final_results.csv", endpoints)
    _write_csv(output_dir / "trajectory_results.csv", trajectory)
    final_metrics = _available_metrics(endpoints, FINAL_METRICS)
    trajectory_metrics = _available_metrics(trajectory, TRAJECTORY_METRICS)

    produced = [
        output_dir / "final_results.json",
        output_dir / "trajectory_results.json",
        output_dir / "final_results.csv",
        output_dir / "trajectory_results.csv",
    ]
    for metric in final_metrics:
        produced.extend(_plot_final(output_dir, endpoints, metric))
    for metric in trajectory_metrics:
        produced.extend(_plot_trajectory(output_dir, trajectory, branch_order, metric))

    manifest = {
        "schema_version": 1,
        "complete": True,
        "endpoint_count": len(endpoints),
        "trajectory_record_count": len(trajectory),
        "branch_order": list(branch_order),
        "final_metrics": list(final_metrics),
        "trajectory_metrics": list(trajectory_metrics),
        "artifacts": [
            {
                "relative_path": path.relative_to(output_dir).as_posix(),
                "bytes": int(path.stat().st_size),
            }
            for path in produced
        ],
    }
    atomic_json(output_dir / "report_manifest.json", manifest)
    return manifest


def _available_metrics(
    rows: Sequence[Mapping[str, Any]], candidates: Sequence[str]
) -> tuple[str, ...]:
    if not rows:
        return ()
    result: list[str] = []
    for metric in candidates:
        presence = [metric in row for row in rows]
        if any(presence) and not all(presence):
            raise ValueError(
                f"report metric is present in only part of the roster: {metric}"
            )
        if all(presence):
            result.append(metric)
        elif metric in BASE_METRICS:
            raise ValueError(f"report roster lacks required metric: {metric}")
    return tuple(result)


def _expanded_trajectory(
    records: Sequence[Mapping[str, Any]],
    *,
    branch_order: Sequence[str],
    fork_step: int,
) -> list[dict[str, Any]]:
    deduplicated: dict[tuple[str, int], dict[str, Any]] = {}
    fork_record: dict[str, Any] | None = None
    for raw in records:
        record = dict(raw)
        branch = str(record["branch"])
        step = int(record["step"])
        if branch == "fork" and step == fork_step:
            fork_record = record
        else:
            deduplicated[(branch, step)] = record
    if fork_record is not None:
        for branch in branch_order:
            copied = dict(fork_record)
            copied["branch"] = branch
            copied["shared_fork_measurement"] = True
            deduplicated[(branch, fork_step)] = copied
    order = {name: index for index, name in enumerate(branch_order)}
    return [
        value
        for _, value in sorted(
            deduplicated.items(),
            key=lambda item: (order.get(item[0][0], len(order)), item[0][1]),
        )
    ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in seen or isinstance(value, (dict, list, tuple)):
                continue
            seen.add(key)
            keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_final(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> list[Path]:
    labels = [str(row["name"]) for row in rows]
    values = [float(row[metric]) for row in rows]
    colors = [
        "#2563eb"
        if row.get("category") == "stable"
        else "#f97316"
        if row.get("category") == "decay"
        else "#64748b"
        for row in rows
    ]
    height = max(5.0, 0.26 * len(rows) + 1.5)
    figure, axis = plt.subplots(figsize=(9.5, height))
    positions = list(range(len(rows)))
    axis.scatter(values, positions, c=colors, s=30, zorder=3)
    axis.set_yticks(positions, labels=labels, fontsize=8)
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.25)
    axis.set_xlabel(_metric_label(metric))
    axis.set_title(f"Final endpoints: {_metric_label(metric)}")
    figure.tight_layout()
    stem = output_dir / f"final_{metric}"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def _plot_trajectory(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    branch_order: Sequence[str],
    metric: str,
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    palette = plt.get_cmap("tab10")
    for index, branch in enumerate(branch_order):
        selected = sorted(
            (row for row in rows if row.get("branch") == branch),
            key=lambda row: int(row["step"]),
        )
        if not selected:
            continue
        axis.plot(
            [int(row["step"]) for row in selected],
            [float(row[metric]) for row in selected],
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=_trajectory_color(branch, index, palette),
            label=_trajectory_label(branch),
        )
    axis.set_xlabel("Optimizer Step")
    axis.set_ylabel(_metric_label(metric))
    axis.set_title(f"Full training trajectory: {_metric_label(metric)}")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    stem = output_dir / f"trajectory_{metric}"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def _trajectory_color(branch: str, index: int, palette: Any) -> Any:
    return "#2563eb" if branch == "stable" else palette((index + 1) % 10)


def _trajectory_label(branch: str) -> str:
    return "Stable" if branch == "stable" else branch


def _metric_label(metric: str) -> str:
    return {
        "validation_loss": "Validation Loss",
        "validation_perplexity": "Validation Perplexity",
        "hessian_trace": "Hessian Trace",
        "hessian_lambda_max": "Largest Hessian Eigenvalue",
    }[metric]
