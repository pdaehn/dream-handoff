from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def test_curated_result_schemas_and_publication_values() -> None:
    rollout = _json(RESULTS / "rollout_characterization.json")
    hysteresis = _json(RESULTS / "hysteresis.json")
    handoff = _json(RESULTS / "handoff_prediction.json")
    prefix = _json(RESULTS / "prefix_conditioning.json")

    assert rollout["schema_version"] == 4
    assert hysteresis["schema_version"] == 3
    assert {handoff["schema_version"], prefix["schema_version"]} == {3}
    for result in (rollout, hysteresis, handoff, prefix):
        assert result["evidence"]["artifact"] == "prospective physical evaluation"
        assert result["evidence"]["capture_sha256"] == (
            "c2c709ca5fbbaa673d8c31d71683fb5b63756596083e2b5f3d00f61975170bba"
        )
    assert rollout["evidence"]["episodes"] == 24
    assert rollout["evidence"]["discarded_rerecord_capture_episode_ids"] == [4]
    assert rollout["candidate_selection"]["switch_rate"] == 0.07210031347962383
    geometry = rollout["candidate_matching_geometry"]
    assert geometry["operational_row_count"] == 11803
    assert geometry["d1"]["count"] == 11803
    assert geometry["margin_12"]["count"] == 11803
    first = rollout["action_space_differences"]["first_difference"]
    assert first["boundary_to_all_within_bank_p95_l2_ratio"] == 3.2211292926634845
    assert first["boundary_to_strict_interior_p95_l2_ratio"] == 3.143051208681689
    assert hysteresis["rule"].endswith("> tau")
    assert hysteresis["absolute_hysteresis"]["mean_matching_distance_penalty"] == (
        0.004998048303324843
    )
    assert hysteresis["memoryless_nearest"]["rapid_successive_switch_within_3_rows_count"] == (1050)
    assert hysteresis["offline_calibration"]["selected_tau"] == 0.06851652264595032
    assert "recorded-continuation regret" in hysteresis["offline_calibration"]["metric_distinction"]
    assert handoff["event_inclusion"]["included_events"] == 430
    assert handoff["analysis"] == "Activation-Time Grounding"
    assert (
        handoff["switch_strata"]["all"]["target_selected_best_of_ten"][
            "static_minus_exact_history_error"
        ]["mean_clustered_ci95_high"]
        < 0
    )
    assert (
        handoff["switch_strata"]["all"]["action_path_nearest"]["static_minus_exact_history_error"][
            "mean_clustered_ci95_low"
        ]
        > 0
    )
    assert handoff["metric_definitions"]["advantage_sign"].startswith(
        "selected_static_error - exact_history_error"
    )
    assert "positive means" in handoff["metric_definitions"]["advantage_sign"]
    assert prefix["plain_replay_gate"]["passed"] is True
    assert prefix["condition_means"]["static_live"]["command_seam"]["mean"] == (2.09665589274794)
    assert prefix["condition_means"]["plain"]["action_difference_seam"]["mean"] == (
        4.786028563266072
    )
    assert "velocity_seam" not in json.dumps(prefix)
    assert set(prefix["condition_definitions"]) == {
        "plain",
        "static",
        "gt",
        "static_live",
        "reactive",
    }
    assert set(prefix["display_order"]) == {"plain", "static", "gt", "static_live"}
    assert "gt_live" not in json.dumps(prefix).lower()


def test_publication_result_set_is_exact() -> None:
    assert {path.name for path in RESULTS.glob("*.json")} == {
        "handoff_prediction.json",
        "hysteresis.json",
        "hysteresis_calibration.json",
        "prefix_conditioning.json",
        "rollout_characterization.json",
    }


def test_calibration_is_independent_of_prospective_physical_evidence() -> None:
    calibration_text = (RESULTS / "hysteresis_calibration.json").read_text()
    assert "prospective physical evaluation" not in calibration_text
    assert "c2c709ca5fbbaa673d8c31d71683fb5b63756596083e2b5f3d00f61975170bba" not in (
        calibration_text
    )


def test_calibration_uncertainty_and_handoff_clustering_are_named_correctly() -> None:
    calibration = _json(RESULTS / "hysteresis_calibration.json")
    uncertainty = calibration["runtime_calibration"]["held_out_evaluation"]["paired_uncertainty"]
    regret = uncertainty["metrics"]["recorded_continuation_regret"]
    assert uncertainty["method"] == "paired bank-clustered bootstrap"
    assert uncertainty["unit"] == "(episode, bank_origin)"
    assert uncertainty["bank_clusters"] == 228
    assert regret["observed_delta"] == 0.050190291717157276
    assert regret["bootstrap_mean_delta"] == 0.05012974181340478
    assert regret["ci95_low"] == 0.029672905379086362
    assert regret["ci95_high"] == 0.07190693670297318

    method_doc = (ROOT / "docs/method.md").read_text()
    assert "paired bank-clustered bootstrap" in method_doc
    assert "observed regret delta" in method_doc
    assert "bootstrap mean" in method_doc
    handoff = _json(RESULTS / "handoff_prediction.json")
    assert handoff["bootstrap"]["unit"] == "physical episode cluster"
    assert "episode-clustered" in method_doc


def test_calibration_result_identities_distinguish_collection_and_publication() -> None:
    collection_sha = "a839c729eb290633382c3118998bc72b47fe4a9d6b5338ac542440f7f84c175f"
    manifest = _json(ROOT / "configs/artifacts.json")
    evidence = manifest["final_evidence"]
    publication_sha = evidence["publication_calibration_result_sha256"]
    assert evidence["prospective_collection_calibration_result_sha256"] == collection_sha
    assert publication_sha != collection_sha

    assert hashlib.sha256((RESULTS / "hysteresis_calibration.json").read_bytes()).hexdigest() == (
        publication_sha
    )
    provenance = _json(ROOT / "configs/prospective_evaluation_provenance.json")
    assert provenance["collection_time_calibration_result_sha256"] == collection_sha


def test_figure_inputs_and_public_entry_points_exist() -> None:
    rollout = _json(RESULTS / "rollout_characterization.json")
    hysteresis = _json(RESULTS / "hysteresis.json")
    handoff = _json(RESULTS / "handoff_prediction.json")
    prefix = _json(RESULTS / "prefix_conditioning.json")

    for order in ("first_difference", "second_difference"):
        block = rollout["action_space_differences"][order]
        for population in ("bank_boundary_l2", "strict_bank_interior_l2"):
            assert {"mean", "median", "p95"} <= set(block[population])
    assert {"memoryless_nearest", "absolute_hysteresis"} <= set(hysteresis)
    assert {"0", "1", ">=2"} <= set(handoff["switch_strata"])
    assert set(prefix["display_order"]) == {"plain", "static", "gt", "static_live"}

    entry_points = (
        "scripts/rollout.py",
        "scripts/train_r2.py",
        "scripts/generate_split.py",
        "scripts/reproduce.py",
        "scripts/plot_results.py",
    )
    readme = (ROOT / "README.md").read_text()
    method = (ROOT / "docs/method.md").read_text()
    reproduction = (ROOT / "docs/reproduction.md").read_text()
    for relative in entry_points:
        assert (ROOT / relative).is_file()
        assert relative in readme or relative in method or relative in reproduction


def test_main_rollout_figure_uses_geometry_boundary_and_strict_interior_only() -> None:
    import runpy

    import matplotlib.pyplot as plt

    plot_rollout = runpy.run_path(str(ROOT / "scripts/plot_results.py"))["_rollout_figure"]
    figure = plot_rollout(_json(RESULTS / "rollout_characterization.json"))
    try:
        text = " ".join(
            [
                value
                for axis in figure.axes
                for value in (axis.get_title(), axis.get_xlabel(), axis.get_ylabel())
            ]
            + [tick.get_text() for axis in figure.axes for tick in axis.get_xticklabels()]
            + [
                legend.get_text()
                for axis in figure.axes
                if axis.get_legend()
                for legend in axis.get_legend().texts
            ]
        ).lower()
        assert "all within" not in text
        assert "strict interior" in text
        assert "boundary" in text
        assert "nearest distance" in text
        assert "separation" in text
    finally:
        plt.close(figure)


def test_prefix_main_figure_keeps_only_the_two_primary_seam_metrics() -> None:
    import runpy

    import matplotlib.pyplot as plt

    plot_prefix = runpy.run_path(str(ROOT / "scripts/plot_results.py"))["plot_prefix"]
    saved: list[plt.Figure] = []
    namespace = plot_prefix.__globals__
    original_save = namespace["_save"]
    namespace["_save"] = lambda figure, *_args: saved.append(figure)
    try:
        plot_prefix(_json(RESULTS / "prefix_conditioning.json"), ROOT / "figures")
        text = " ".join(axis.get_title() for axis in saved[0].axes).lower()
        assert "command seam" in text
        assert "action-difference seam" in text
        assert "reactive" not in text
        assert "state" not in text
        assert "diversity" not in text
    finally:
        namespace["_save"] = original_save
        for figure in saved:
            plt.close(figure)


def test_public_story_omits_obsolete_calibration_terminology() -> None:
    old_tau = "0.027628261595964432"
    for relative in ("README.md", "docs/method.md", "docs/reproduction.md"):
        text = (ROOT / relative).read_text()
        assert old_tau not in text
        assert "NumPy phase-zero" not in text
        assert "NumPy phase-0" not in text
        assert re.search(r"\bv[12]\b(?!\.[A-Za-z0-9])", text, flags=re.IGNORECASE) is None


def test_current_publication_surface_uses_final_artifact_names() -> None:
    reproduction = (ROOT / "docs/reproduction.md").read_text()
    for filename in ("capture.npz", "generated_banks.npz"):
        assert filename in reproduction
    assert "prospective physical evaluation" in reproduction


def test_machine_diagnostics_and_legal_files_remain_available() -> None:
    prefix = _json(RESULTS / "prefix_conditioning.json")
    assert all(
        "candidate_diversity" in prefix["condition_means"][key] for key in prefix["display_order"]
    )
    assert all(
        "reactive_reference_error" in prefix["condition_means"][key]
        for key in prefix["display_order"]
    )
    assert all("state_seam" in prefix["condition_means"][key] for key in prefix["display_order"])
    assert hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest() == (
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    )
    assert not (ROOT / "NOTICE").exists()
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    assert "NM512/r2dreamer" in notices
    assert "546e4fab8146ea4b14e1d7726bbc1a8a1d50322f" in notices
    assert "MIT License" in notices
    assert "Copyright (c) 2026 Naoki Morihira" in notices
    assert "Permission is hereby granted, free of charge" in notices
    assert "The above copyright notice and this permission notice shall be included" in notices
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in notices
    assert "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS" in notices
    assert "## LaProp" in notices
    assert "Copyright (c) 2020 Wang, T. Zhikang" in notices
    assert "src/dream_handoff/r2dreamer/optimization.py" in notices

    readme = (ROOT / "README.md").read_text()
    license_heading = "## Attribution, citation, and licensing\n"
    assert readme.count(license_heading) == 1
    license_section = readme.split(license_heading, maxsplit=1)[1].strip()
    assert "Chen et al., [*DREAM-Chunk:" in license_section
    assert "The adapted R2Dreamer source is attributed" in license_section
    assert "If you use DreamHandoff in academic work" in license_section
    assert "@misc{dreamhandoff2026" in license_section
    assert "DreamHandoff v1.0.0" in license_section
    assert "DreamHandoff is licensed under [Apache-2.0](LICENSE)" in license_section
    assert "Adapted R2Dreamer components retain their upstream MIT license" in license_section
    assert "[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)" in license_section
    assert "Apache-2.0 AND MIT" not in license_section
    assert "[NOTICE](NOTICE)" not in license_section


def test_public_docs_keep_license_and_compatibility_distinct() -> None:
    old_project_name = "Dream" + "Reflex"
    for relative in ("README.md", "docs/method.md", "docs/reproduction.md"):
        text = (ROOT / relative).read_text()
        assert "[NOTICE](NOTICE)" not in text
        assert "Apache-2.0 AND MIT" not in text
        assert f"{old_project_name} contributors" not in text
        assert "DreamHandoff" + " contributors" not in text
        assert old_project_name not in text

    method = (ROOT / "docs/method.md").read_text()
    assert "`dream-reflex-r2dreamer`" in method
    assert "frozen schema identifier kept for compatibility" in method


def test_artifact_manifest_and_public_surface_are_portable() -> None:
    manifest = _json(ROOT / "configs/artifacts.json")
    assert (ROOT / manifest["dataset"]["split_manifest"]).is_file()
    assert (ROOT / manifest["smolvla"]["reproduction_config"]).is_file()
    assert (ROOT / manifest["r2"]["system_config"]).is_file()
    assert manifest["prefix_conditioning_replay"]["sha256"] == (
        "3ce2ab0c8eb39b9650de98f3377f1b676e5131205f808dca16490c73982f3258"
    )

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["authors"] == [{"name": "Paul Dähn"}]
    assert project["project"]["license"] == "Apache-2.0"
    assert project["project"]["license-files"] == [
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
    ]
    assert "placeholder" not in (ROOT / "LICENSE").read_text().lower()

    public_files = [
        ROOT / "README.md",
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "pyproject.toml",
        *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / "configs").rglob("*.json")),
        *sorted(RESULTS.glob("*.json")),
    ]
    for path in public_files:
        assert "/home/" not in path.read_text(), path
