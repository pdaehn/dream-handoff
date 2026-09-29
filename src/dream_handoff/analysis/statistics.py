"""Small statistical reductions shared by the final handoff analyses."""

from __future__ import annotations

from typing import Any

import numpy as np


def distribution_summary(values: list[float | None]) -> dict[str, Any]:
    """Summarize finite values with the exact frozen reduction semantics."""
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
    }


def episode_clustered_bootstrap(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    num_resamples: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap event statistics by resampling whole physical episodes."""
    if num_resamples <= 0:
        raise ValueError("num_resamples must be positive")
    episode_ids = np.asarray(sorted({int(row["episode_id"]) for row in rows}), dtype=np.int64)
    if not episode_ids.size:
        return {
            "episodes": 0,
            "resamples": num_resamples,
            "mean": None,
            "mean_ci95_low": None,
            "mean_ci95_high": None,
            "median": None,
            "median_ci95_low": None,
            "median_ci95_high": None,
        }
    values_by_episode = {
        episode_id: np.asarray(
            [row[value_key] for row in rows if row["episode_id"] == episode_id],
            dtype=np.float64,
        )
        for episode_id in episode_ids
    }
    observed = np.concatenate(list(values_by_episode.values()))
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(num_resamples, dtype=np.float64)
    bootstrap_medians = np.empty(num_resamples, dtype=np.float64)
    for index in range(num_resamples):
        sampled = rng.choice(episode_ids, size=len(episode_ids), replace=True)
        sample = np.concatenate([values_by_episode[int(episode)] for episode in sampled])
        bootstrap_means[index] = np.mean(sample)
        bootstrap_medians[index] = np.median(sample)
    return {
        "episodes": int(len(episode_ids)),
        "resamples": num_resamples,
        "mean": float(np.mean(observed)),
        "mean_ci95_low": float(np.percentile(bootstrap_means, 2.5)),
        "mean_ci95_high": float(np.percentile(bootstrap_means, 97.5)),
        "median": float(np.median(observed)),
        "median_ci95_low": float(np.percentile(bootstrap_medians, 2.5)),
        "median_ci95_high": float(np.percentile(bootstrap_medians, 97.5)),
    }


__all__ = ["distribution_summary", "episode_clustered_bootstrap"]
