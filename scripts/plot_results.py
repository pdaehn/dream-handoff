#!/usr/bin/env python3
"""Generate the four publication figures from checked-in curated results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"blue": "#3274A1", "orange": "#E1812C", "green": "#3A923A", "gray": "#707070"}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema_version") not in {1, 2, 3, 4}:
        raise ValueError(f"unsupported result schema in {path}")
    return value


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "DreamHandoff"},
    )
    figure.savefig(
        output_dir / f"{stem}.png",
        bbox_inches="tight",
        metadata={"Software": "DreamHandoff"},
    )
    plt.close(figure)


def _rollout_figure(result: dict[str, Any]) -> plt.Figure:
    figure = plt.figure(figsize=(8.0, 3.8))
    grid = figure.add_gridspec(2, 2, width_ratios=(0.9, 1.6), hspace=0.65, wspace=0.40)
    geometry_axes = (figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[1, 0]))
    geometry = result["candidate_matching_geometry"]
    for index, (axis, key, label) in enumerate(
        zip(
            geometry_axes,
            ("d1", "margin_12"),
            ("Nearest distance $d_1$", "Separation $d_2-d_1$"),
            strict=True,
        )
    ):
        values = geometry[key]
        axis.hlines(0, values["p25"], values["p75"], color=COLORS["blue"], linewidth=7)
        axis.hlines(0, values["p25"], values["p90"], color=COLORS["blue"], linewidth=1.5)
        axis.scatter(
            values["median"], 0, color="white", edgecolor=COLORS["blue"], zorder=3, label="Median"
        )
        axis.scatter(
            values["mean"], 0, color=COLORS["orange"], marker="D", s=24, zorder=3, label="Mean"
        )
        axis.set_yticks([])
        axis.set_xlabel(label)
        axis.grid(axis="x", alpha=0.25)
        axis.set_title(("A  Candidate-to-live matching geometry" if index == 0 else ""), loc="left")
    geometry_axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    geometry_axes[1].text(
        0.98,
        0.88,
        "Thick: p25–p75\nLine: p25–p90",
        transform=geometry_axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color=COLORS["gray"],
    )

    axis = figure.add_subplot(grid[:, 1])
    keys = ("first_difference", "second_difference")
    labels = ("First difference", "Second difference")
    x = np.arange(len(keys))
    width = 0.36
    interior = [
        result["action_space_differences"][key]["strict_bank_interior_l2"]["p95"] for key in keys
    ]
    boundary = [result["action_space_differences"][key]["bank_boundary_l2"]["p95"] for key in keys]
    axis.bar(x - width / 2, interior, width, label="Strict interior", color=COLORS["blue"])
    axis.bar(x + width / 2, boundary, width, label="Boundary", color=COLORS["orange"])
    axis.set_xticks(x, labels)
    axis.set_ylim(bottom=0)
    axis.set_ylabel("95th-percentile L2 magnitude")
    axis.set_title("B  Canonical action-space differences", loc="left")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.suptitle("Candidate-bank behavior and chunk handoffs")
    figure.subplots_adjust(top=0.84, bottom=0.17, left=0.07, right=0.98)
    return figure


def plot_rollout(result: dict[str, Any], output_dir: Path) -> None:
    figure = _rollout_figure(result)
    _save(figure, output_dir, "rollout_discontinuity")


def plot_hysteresis(result: dict[str, Any], output_dir: Path) -> None:
    memoryless = result["memoryless_nearest"]
    stabilized = result["absolute_hysteresis"]
    calibration = result["offline_calibration"]
    pareto = calibration["pareto_curve"]
    axes_x = [point["switch_rate"] * 100 for point in pareto]
    axes_y = [point["recorded_continuation_regret"] for point in pareto]
    selected = calibration["calibration_selected_point"]
    reference = calibration["calibration_memoryless"]
    regret_limit = calibration["moderate_rule"]["maximum_mean_regret"]

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.25))
    axes[0].plot(axes_x, axes_y, color=COLORS["blue"], marker="o", markersize=3)
    axes[0].scatter(
        [reference["switch_rate"] * 100],
        [reference["recorded_continuation_regret"]],
        color=COLORS["gray"],
        marker="s",
        label="Memoryless reference",
        zorder=3,
    )
    axes[0].scatter(
        [selected["switch_rate"] * 100],
        [selected["recorded_continuation_regret"]],
        color=COLORS["green"],
        marker="*",
        s=90,
        label=f"Selected τ={result['tau']:.4f}",
        zorder=4,
    )
    axes[0].axhline(
        regret_limit,
        color=COLORS["orange"],
        linestyle="--",
        linewidth=1,
        label="≤5% calibration-regret boundary",
    )
    axes[0].set_xlabel("Calibration switch rate (%)")
    axes[0].set_ylabel("Recorded-continuation regret")
    axes[0].set_title("A  Runtime-faithful calibration Pareto frontier", loc="left")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)

    categories = ("Operational\nswitches", "Rapid successive\nswitches", "A→B→A\nreversals")
    x = np.arange(len(categories))
    width = 0.36
    memoryless_counts = (
        memoryless["operational_switch_count"],
        memoryless["rapid_successive_switch_within_3_rows_count"],
        memoryless["reversal_aba_count"],
    )
    stabilized_counts = (
        stabilized["operational_switch_count"],
        stabilized["rapid_successive_switch_within_3_rows_count"],
        stabilized["reversal_aba_count"],
    )
    axes[1].bar(
        x - width / 2,
        memoryless_counts,
        width,
        label="Memoryless replay",
        color=COLORS["gray"],
    )
    axes[1].bar(
        x + width / 2,
        stabilized_counts,
        width,
        label="Absolute hysteresis",
        color=COLORS["green"],
    )
    axes[1].set_xticks(x, categories)
    axes[1].set_ylabel("Prospective physical replay count")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].set_title("B  Prospective physical stabilization", loc="left")
    figure.suptitle("Offline Pareto calibration and prospective physical stabilization")
    figure.tight_layout()
    _save(figure, output_dir, "hysteresis")


def plot_handoff(result: dict[str, Any], output_dir: Path) -> None:
    strata = ("0", "1", ">=2")
    blocks = result["switch_strata"]
    figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.3))
    mismatch = [blocks[key]["factual_path_mismatch"]["mean"] for key in strata]
    axes[0].bar(strata, mismatch, color=COLORS["blue"], width=0.62)
    axes[0].set_xlabel("Selector switches during request→activation")
    axes[0].set_ylabel("Mean factual path mismatch")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(axis="y", alpha=0.25)

    x = np.arange(len(strata))
    width = 0.36
    for offset, comparator, label, color in (
        (-width / 2, "target_selected_best_of_ten", "Target-selected best of 10", COLORS["orange"]),
        (width / 2, "action_path_nearest", "Action-path-nearest", COLORS["green"]),
    ):
        values = [
            blocks[key][comparator]["static_minus_exact_history_error"]["mean"] for key in strata
        ]
        lower = [
            value
            - blocks[key][comparator]["static_minus_exact_history_error"]["mean_clustered_ci95_low"]
            for key, value in zip(strata, values, strict=True)
        ]
        upper = [
            blocks[key][comparator]["static_minus_exact_history_error"]["mean_clustered_ci95_high"]
            - value
            for key, value in zip(strata, values, strict=True)
        ]
        axes[1].bar(
            x + offset,
            values,
            width,
            yerr=np.asarray([lower, upper]),
            capsize=3,
            label=label,
            color=color,
        )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, strata)
    axes[1].set_xlabel("Selector switches during request→activation")
    axes[1].set_ylabel("Static error − exact-history error")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=7
    )
    axes[1].set_title("Positive values mean exact history helps", loc="left", fontsize=8)
    figure.suptitle("Activation-time grounding: path mismatch grows with switching")
    figure.tight_layout()
    _save(figure, output_dir, "handoff_prediction")


def plot_prefix(result: dict[str, Any], output_dir: Path) -> None:
    means = result["condition_means"]
    keys = ("plain", "static", "gt", "plain", "static_live")
    labels = ("Plain", "Static", "GT\n(exact)", "Plain", "Static-live")
    positions = np.asarray([0, 1, 2, 4.2, 5.5], dtype=float)
    colors = (
        COLORS["gray"],
        COLORS["blue"],
        COLORS["orange"],
        COLORS["gray"],
        COLORS["green"],
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    maximum = max(
        means[key][metric]["mean"]
        for key in keys
        for metric in ("command_seam", "action_difference_seam")
    )
    for axis, metric, title in zip(
        axes,
        ("command_seam", "action_difference_seam"),
        ("Command seam", "Action-difference seam"),
        strict=True,
    ):
        values = [means[key][metric]["mean"] for key in keys]
        axis.bar(positions, values, color=colors, width=0.72)
        axis.set_xticks(positions, labels)
        axis.set_ylim(0, maximum * 1.1)
        axis.grid(axis="y", alpha=0.25)
        axis.axvline(3, color="#BBBBBB", linewidth=0.8)
        axis.text(
            1,
            -0.30,
            "Matched actual delay",
            transform=axis.get_xaxis_transform(),
            ha="center",
            fontsize=8,
        )
        axis.text(
            4.85,
            -0.30,
            "Deployment comparison",
            transform=axis.get_xaxis_transform(),
            ha="center",
            fontsize=8,
        )
        axis.set_title(title)
    axes[0].set_ylabel("Mean canonical action-space L2 magnitude")
    figure.suptitle("Prefix conditioning reduces asynchronous handoff seams")
    figure.tight_layout()
    figure.subplots_adjust(bottom=0.30)
    _save(figure, output_dir, "prefix_conditioning")


def generate_figures(results_dir: Path, output_dir: Path) -> None:
    _style()
    plot_rollout(_read(results_dir / "rollout_characterization.json"), output_dir)
    plot_hysteresis(_read(results_dir / "hysteresis.json"), output_dir)
    plot_handoff(_read(results_dir / "handoff_prediction.json"), output_dir)
    plot_prefix(_read(results_dir / "prefix_conditioning.json"), output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    generate_figures(args.results, args.output)


if __name__ == "__main__":
    main()
