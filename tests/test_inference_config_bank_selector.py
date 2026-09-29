from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dream_handoff.inference import (
    FINAL_HYSTERESIS_TAU,
    CandidateBank,
    CandidateSelector,
    DreamHandoffInferenceConfig,
    GeneratedCandidateActions,
)

FIXTURE = Path(__file__).parent / "fixtures" / "wp3_frozen_lifecycle.json"


def test_final_public_configuration_is_valid() -> None:
    config = DreamHandoffInferenceConfig(r2_checkpoint=Path("external/latest.pt"))
    assert (
        config.num_candidates,
        config.max_bank_age,
        config.bank_refill_threshold,
        config.guidance_horizon,
    ) == (10, 35, 15, 15)
    assert config.async_bank_guidance == "plain"
    assert config.selector_mode == "absolute_hysteresis"
    assert config.hysteresis_tau == FINAL_HYSTERESIS_TAU == 0.06851652264595032
    assert config.phase0_seed == 0
    assert config.compile_world_model_imagination is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"selector_mode": "absolute_hysteresis", "hysteresis_tau": None}, "requires"),
        ({"bank_refill_threshold": 35}, "smaller"),
        ({"guidance_horizon": 16}, "cannot exceed"),
        ({"max_bank_age": 0}, "positive"),
    ],
)
def test_invalid_configuration_is_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DreamHandoffInferenceConfig(r2_checkpoint=Path("checkpoint.pt"), **overrides)


def test_generated_and_active_bank_invariants_allow_distinct_action_spaces() -> None:
    policy = torch.zeros(3, 4, 5)
    control = torch.ones(3, 4, 2)
    generated = GeneratedCandidateActions(policy, control, request_action_count=7)
    bank = CandidateBank(policy, control, torch.zeros(3, 4, 6), origin_action_count=9)
    assert generated.horizon == bank.horizon == 4
    assert generated.num_candidates == bank.num_candidates == 3


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: GeneratedCandidateActions(torch.zeros(2, 3, 1), torch.zeros(3, 3, 1), 0),
        lambda: CandidateBank(torch.zeros(2, 3, 1), torch.zeros(2, 2, 1), torch.zeros(2, 3, 4), 0),
        lambda: CandidateBank(
            torch.zeros(2, 3, 1),
            torch.zeros(2, 3, 1),
            torch.full((2, 3, 4), torch.nan),
            0,
        ),
        lambda: CandidateBank(
            torch.zeros(2, 3, 1),
            torch.zeros(2, 3, 1),
            torch.empty(2, 3, 4, device="meta"),
            0,
        ),
    ],
)
def test_malformed_candidate_records_fail(constructor) -> None:
    with pytest.raises(ValueError):
        constructor()


def test_selector_matches_frozen_phase_zero_sequence_and_rng_spans_resets() -> None:
    expected = json.loads(FIXTURE.read_text())["phase0_seed_0_n10"]
    selector = CandidateSelector("baseline", hysteresis_tau=None, seed=0)
    choices = []
    for _ in expected:
        selector.reset_bank()
        result = selector.select(torch.zeros(1), torch.zeros(10, 1), phase=0)
        choices.append(result.index)
    assert choices == expected


def test_baseline_uses_first_argmin_on_equal_nonzero_phase() -> None:
    selector = CandidateSelector("baseline", hysteresis_tau=None, seed=0)
    selector.select(torch.zeros(1), torch.zeros(2, 1), phase=0)
    result = selector.select(torch.zeros(1), torch.tensor([[1.0], [1.0]]), phase=1)
    assert result.index == result.challenger_index == 0
    assert result.rule == "nearest"


def test_absolute_hysteresis_stays_switches_and_uses_strict_threshold() -> None:
    stay = CandidateSelector("absolute_hysteresis", hysteresis_tau=0.25, seed=0)
    assert stay.select(torch.zeros(1), torch.zeros(2, 1), phase=0).index == 0
    boundary = stay.select(torch.zeros(1), torch.tensor([[0.25], [0.0]]), phase=1)
    assert boundary.index == 0
    assert boundary.advantage == 0.25

    switch = CandidateSelector("absolute_hysteresis", hysteresis_tau=0.25, seed=0)
    switch.select(torch.zeros(1), torch.zeros(2, 1), phase=0)
    result = switch.select(torch.zeros(1), torch.tensor([[0.5], [0.0]]), phase=1)
    assert result.index == 1
    assert result.switched is True


def test_selector_values_match_frozen_fixture() -> None:
    fixture = json.loads(FIXTURE.read_text())["selection"]
    tau = fixture["hysteresis_tau"]

    baseline = CandidateSelector("baseline", hysteresis_tau=None, seed=0)
    baseline.select(torch.zeros(1), torch.zeros(3, 1), phase=0)
    distances = torch.tensor(fixture["baseline_distances"]).unsqueeze(1)
    assert baseline.select(torch.zeros(1), distances, phase=1).index == fixture["baseline_selected"]

    stay = CandidateSelector("absolute_hysteresis", hysteresis_tau=tau, seed=0)
    stay.select(torch.zeros(1), torch.zeros(2, 1), phase=0)
    stay_case = fixture["stay"]
    stayed = stay.select(
        torch.zeros(1),
        torch.tensor([[stay_case["incumbent_distance"]], [stay_case["challenger_distance"]]]),
        phase=1,
    )
    assert stayed.index == stay_case["selected"]

    switch = CandidateSelector("absolute_hysteresis", hysteresis_tau=tau, seed=0)
    switch.select(torch.zeros(1), torch.zeros(2, 1), phase=0)
    switch_case = fixture["switch"]
    switched = switch.select(
        torch.zeros(1),
        torch.tensor([[switch_case["incumbent_distance"]], [switch_case["challenger_distance"]]]),
        phase=1,
    )
    assert switched.index == switch_case["selected"]


def test_new_bank_clears_incumbent_without_reseeding_phase_zero_rng() -> None:
    selector = CandidateSelector("absolute_hysteresis", hysteresis_tau=0.1, seed=0)
    first = selector.select(torch.zeros(1), torch.zeros(10, 1), phase=0).index
    selector.reset_bank()
    assert selector.incumbent is None
    second = selector.select(torch.zeros(1), torch.zeros(10, 1), phase=0).index
    assert (first, second) == (4, 9)
