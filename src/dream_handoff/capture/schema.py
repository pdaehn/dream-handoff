"""Reader and validation surface for final diagnostic format 16."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib import format as npy_format

DIAGNOSTIC_FORMAT_VERSION = 16
HISTORICAL_DIAGNOSTIC_FORMAT = "dream-reflex-rollout-diagnostic"
DREAMHANDOFF_DIAGNOSTIC_FORMAT = "dream-handoff-rollout-diagnostic"
SUPPORTED_DIAGNOSTIC_FORMATS = {
    HISTORICAL_DIAGNOSTIC_FORMAT,
    DREAMHANDOFF_DIAGNOSTIC_FORMAT,
}

PER_COMMAND_FIELDS = (
    "timestamp",
    "wall_timestamp",
    "command_step",
    "control_step",
    "episode_id",
    "bank_origin_action_count",
    "phase",
    "selected_candidate_index",
    "challenger_candidate_index",
    "incumbent_candidate_index",
    "hysteresis_advantage",
    "candidate_distances",
    "live_world_model_stoch",
    "live_world_model_deter",
    "candidate_switched",
    "selection_rule",
    "selected_policy_action",
    "selected_control_action",
    "final_executed_policy_action",
    "final_executed_action",
    "interpolated_action",
    "action_processor_output",
    "command_sent_to_robot",
    "measured_joint_position",
)

CANDIDATE_BANK_FIELDS = (
    "candidate_bank_episode_id",
    "candidate_bank_origin_action_count",
    "candidate_bank_sampled_noise",
    "candidate_bank_policy_actions",
    "candidate_bank_control_actions",
    "candidate_bank_dreamed_matching_features",
    "candidate_bank_horizon",
)

ASYNC_REQUEST_FIELDS = (
    "async_request_episode_id",
    "async_request_id",
    "async_request_generation",
    "async_request_origin_action_count",
    "async_request_launch_wall_time",
    "async_request_ready_wall_time",
    "async_request_build_latency_seconds",
    "async_request_actions_consumed_while_building",
    "async_request_wall_clock_delay_steps",
    "async_request_actual_consumed_delay_steps",
    "async_request_delay_difference_steps",
    "async_request_activation_action_count",
    "async_request_activation_phase",
    "async_request_stale",
    "async_request_stale_reason",
    "async_request_rtc_guidance_mode",
    "async_request_rtc_guidance_horizon",
    "async_request_rtc_request_phase",
    "async_request_rtc_predicted_inference_delay",
    "async_request_bank_refill_threshold",
    "async_request_cropped_bank_horizon",
    "async_request_activation_reimagination_seconds",
    "async_request_predicted_inference_delay_steps",
    "async_request_generation_noise",
    "async_request_rtc_prefix_actions",
)

SPARSE_OBSERVATION_FIELDS = (
    "sparse_observation_episode_id",
    "sparse_observation_request_id",
    "sparse_observation_generation",
    "sparse_observation_origin_action_count",
    "sparse_observation_activation_action_count",
    "sparse_observation_action_count",
    "sparse_observation_event_role",
)

REQUIRED_FORMAT16_FIELDS = (
    *PER_COMMAND_FIELDS,
    *CANDIDATE_BANK_FIELDS,
    *ASYNC_REQUEST_FIELDS,
    *SPARSE_OBSERVATION_FIELDS,
    "action_keys",
    "metadata_json",
)

# Every final WP6 result can be reconstructed from these factual arrays. Derived
# diversity and counterfactual-summary arrays in the frozen file are intentionally
# not requirements of the public writer.
WP6_REQUIRED_FIELDS = frozenset(REQUIRED_FORMAT16_FIELDS)


@dataclass(frozen=True)
class ArrayInfo:
    """Shape and dtype read from an embedded NPY header without loading its payload."""

    shape: tuple[int, ...]
    dtype: np.dtype
    fortran_order: bool


@dataclass(frozen=True)
class RequestRecord:
    """One request-to-ready-to-activation lineage row."""

    index: int
    episode_id: int
    request_id: int
    generation: int
    origin_action_count: int
    activation_action_count: int
    actual_elapsed: int
    stale: bool
    stale_reason: str
    guidance_mode: str
    guidance_horizon: int
    request_phase: int
    predicted_delay: int
    cropped_bank_horizon: int

    @property
    def activated(self) -> bool:
        return not self.stale and self.activation_action_count >= 0

    @property
    def is_handoff(self) -> bool:
        return self.activated and self.activation_action_count > self.origin_action_count

    @property
    def generation_completed(self) -> bool:
        """Whether generation produced a replayable payload, activated or stale."""

        return self.actual_elapsed >= 0


@dataclass(frozen=True)
class HandoffRecord:
    """An activated non-initial request joined to its incoming candidate bank."""

    request: RequestRecord
    candidate_bank_index: int


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest without loading an artifact into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_array_headers(path: Path) -> dict[str, ArrayInfo]:
    headers: dict[str, ArrayInfo] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if not member.endswith(".npy"):
                    continue
                with archive.open(member) as stream:
                    version = npy_format.read_magic(stream)
                    if version == (1, 0):
                        shape, fortran, dtype = npy_format.read_array_header_1_0(stream)
                    elif version in {(2, 0), (3, 0)}:
                        shape, fortran, dtype = npy_format.read_array_header_2_0(stream)
                    else:
                        raise ValueError(f"unsupported NPY header version {version} in {member}")
                headers[member.removesuffix(".npy")] = ArrayInfo(
                    tuple(int(value) for value in shape),
                    np.dtype(dtype),
                    bool(fortran),
                )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{path} is not a readable NPZ diagnostic artifact") from exc
    return headers


class DiagnosticCapture:
    """Lazy, context-managed view of a validated format-16 NPZ capture."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.array_info = _read_array_headers(self.path)
        missing = sorted(set(REQUIRED_FORMAT16_FIELDS) - set(self.array_info))
        if missing:
            raise ValueError(f"incomplete format-16 diagnostic; missing fields: {missing}")
        self._archive = np.load(self.path, allow_pickle=False)
        self._cache: dict[str, np.ndarray] = {}
        try:
            self.metadata = json.loads(str(self._archive["metadata_json"].item()))
            self._validate()
        except Exception:
            self._archive.close()
            raise

    def __enter__(self) -> DiagnosticCapture:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._archive.close()
        self._cache.clear()

    def array(self, name: str) -> np.ndarray:
        """Load one named array on demand and cache it for the reader lifetime."""

        if name not in self.array_info:
            raise KeyError(name)
        if name not in self._cache:
            self._cache[name] = np.asarray(self._archive[name])
        return self._cache[name]

    @property
    def format_version(self) -> int:
        return int(self.metadata["format_version"])

    @property
    def control_step_count(self) -> int:
        return int(self.array_info["control_step"].shape[0])

    @property
    def candidate_bank_count(self) -> int:
        return int(self.array_info["candidate_bank_origin_action_count"].shape[0])

    @property
    def request_count(self) -> int:
        return int(self.array_info["async_request_id"].shape[0])

    @property
    def episode_count(self) -> int:
        episodes = self.array("episode_id")
        return int(np.unique(episodes).size) if len(episodes) else 0

    @property
    def recorded_episode_ids(self) -> tuple[int, ...]:
        """Return capture episode IDs retained by the rollout dataset.

        Pinned LeRobot reuses the current dataset episode index when the
        operator discards an episode and requests a re-record.  The diagnostic
        capture intentionally remains append-only, so in that case more than
        one capture episode maps to the same dataset episode.  The final
        occurrence is the one saved by LeRobot.
        """

        episode = self.array("episode_id").astype(np.int64, copy=False)
        if "dataset_episode_index" not in self.array_info:
            return tuple(int(value) for value in np.unique(episode))
        dataset_episode = self.array("dataset_episode_index").astype(np.int64, copy=False)
        if dataset_episode.shape != episode.shape or bool((dataset_episode < 0).any()):
            raise ValueError("capture has incomplete dataset episode provenance")
        capture_to_dataset: dict[int, int] = {}
        retained_by_dataset: dict[int, int] = {}
        for capture_id, dataset_id in zip(episode, dataset_episode, strict=True):
            capture_value = int(capture_id)
            dataset_value = int(dataset_id)
            previous = capture_to_dataset.setdefault(capture_value, dataset_value)
            if previous != dataset_value:
                raise ValueError("one capture episode maps to multiple dataset episodes")
            retained_by_dataset[dataset_value] = capture_value
        return tuple(retained_by_dataset[index] for index in sorted(retained_by_dataset))

    @property
    def discarded_episode_ids(self) -> tuple[int, ...]:
        """Return append-only capture episodes discarded before dataset save."""

        all_ids = {int(value) for value in np.unique(self.array("episode_id"))}
        retained = set(self.recorded_episode_ids)
        return tuple(sorted(all_ids - retained))

    @property
    def recorded_episode_count(self) -> int:
        return len(self.recorded_episode_ids)

    @property
    def action_keys(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.array("action_keys"))

    def requests(self) -> tuple[RequestRecord, ...]:
        arrays = {
            name: self.array(name)
            for name in (
                "async_request_episode_id",
                "async_request_id",
                "async_request_generation",
                "async_request_origin_action_count",
                "async_request_activation_action_count",
                "async_request_actual_consumed_delay_steps",
                "async_request_stale",
                "async_request_stale_reason",
                "async_request_rtc_guidance_mode",
                "async_request_rtc_guidance_horizon",
                "async_request_rtc_request_phase",
                "async_request_predicted_inference_delay_steps",
                "async_request_cropped_bank_horizon",
            )
        }
        records = []
        for index in range(self.request_count):
            records.append(
                RequestRecord(
                    index=index,
                    episode_id=int(arrays["async_request_episode_id"][index]),
                    request_id=int(arrays["async_request_id"][index]),
                    generation=int(arrays["async_request_generation"][index]),
                    origin_action_count=int(arrays["async_request_origin_action_count"][index]),
                    activation_action_count=int(
                        arrays["async_request_activation_action_count"][index]
                    ),
                    actual_elapsed=int(arrays["async_request_actual_consumed_delay_steps"][index]),
                    stale=bool(arrays["async_request_stale"][index]),
                    stale_reason=str(arrays["async_request_stale_reason"][index]),
                    guidance_mode=str(arrays["async_request_rtc_guidance_mode"][index]),
                    guidance_horizon=int(arrays["async_request_rtc_guidance_horizon"][index]),
                    request_phase=int(arrays["async_request_rtc_request_phase"][index]),
                    predicted_delay=int(
                        arrays["async_request_predicted_inference_delay_steps"][index]
                    ),
                    cropped_bank_horizon=int(arrays["async_request_cropped_bank_horizon"][index]),
                )
            )
        return tuple(records)

    def handoffs(self) -> tuple[HandoffRecord, ...]:
        bank_episode = self.array("candidate_bank_episode_id")
        bank_origin = self.array("candidate_bank_origin_action_count")
        index_by_identity = {
            (int(episode), int(origin)): index
            for index, (episode, origin) in enumerate(zip(bank_episode, bank_origin, strict=True))
        }
        handoffs = []
        for request in self.requests():
            if not request.is_handoff:
                continue
            identity = (request.episode_id, request.activation_action_count)
            if identity not in index_by_identity:
                raise ValueError(f"activated request has no candidate bank row: {identity}")
            handoffs.append(HandoffRecord(request, index_by_identity[identity]))
        return tuple(handoffs)

    def sparse_observation(
        self,
        *,
        episode_id: int,
        generation: int,
        request_id: int,
        event_role: str,
    ) -> dict[str, np.ndarray]:
        """Recover one exact request-origin or activation observation."""

        identities = zip(
            self.array("sparse_observation_episode_id"),
            self.array("sparse_observation_generation"),
            self.array("sparse_observation_request_id"),
            self.array("sparse_observation_event_role"),
            strict=True,
        )
        wanted = (int(episode_id), int(generation), int(request_id), str(event_role))
        matches = [
            index
            for index, values in enumerate(identities)
            if (int(values[0]), int(values[1]), int(values[2]), str(values[3])) == wanted
        ]
        if len(matches) != 1:
            raise KeyError(f"expected one sparse observation {wanted}, found {len(matches)}")
        index = matches[0]
        result: dict[str, np.ndarray] = {}
        for item in self._sparse_key_metadata():
            result[str(item["key"])] = self.array(str(item["array"]))[index].copy()
        return result

    def require_exact_replay(self) -> None:
        """Reject a structurally valid capture that lacks exact replay payloads."""

        noise = self.array_info["async_request_generation_noise"]
        if self.request_count == 0 or len(noise.shape) != 4 or noise.shape[0] != self.request_count:
            raise ValueError("capture has no complete per-request explicit generation noise")
        completed = self.array("async_request_actions_consumed_while_building") >= 0
        if not bool(completed.any()):
            raise ValueError("capture has no completed request generation to replay")
        activated = (~self.array("async_request_stale")) & (
            self.array("async_request_activation_action_count") >= 0
        )
        if bool((activated & ~completed).any()):
            raise ValueError("capture has an activated request without completed generation")
        if not bool(np.isfinite(self.array("async_request_generation_noise")[completed]).all()):
            raise ValueError("capture has incomplete explicit noise for a completed generation")
        keys = self._sparse_key_metadata()
        if not keys:
            raise ValueError("capture has no sparse exact observation value arrays")
        roles = self.array("sparse_observation_event_role")
        if not {"request_origin", "activation"}.issubset(set(map(str, roles))):
            raise ValueError("capture does not contain both request-origin and activation frames")

    def _sparse_key_metadata(self) -> tuple[dict[str, Any], ...]:
        sparse = self.metadata.get("sparse_exact_observations", {})
        keys = sparse.get("keys", []) if isinstance(sparse, dict) else []
        return tuple(item for item in keys if isinstance(item, dict))

    def _validate(self) -> None:
        diagnostic_format = self.metadata.get("format")
        if diagnostic_format not in SUPPORTED_DIAGNOSTIC_FORMATS:
            raise ValueError(f"unsupported diagnostic format {diagnostic_format!r}")
        version = self.metadata.get("format_version")
        if version != DIAGNOSTIC_FORMAT_VERSION:
            raise ValueError(
                f"unsupported diagnostic format version {version!r}; "
                f"expected {DIAGNOSTIC_FORMAT_VERSION}"
            )

        self._validate_group(PER_COMMAND_FIELDS, "per-command")
        self._validate_group(CANDIDATE_BANK_FIELDS, "candidate-bank")
        self._validate_group(ASYNC_REQUEST_FIELDS[:-2], "async-request")
        self._validate_group(SPARSE_OBSERVATION_FIELDS, "sparse-observation")

        action_keys = self.array_info["action_keys"]
        if len(action_keys.shape) != 1 or action_keys.shape[0] <= 0:
            raise ValueError("action_keys must be a non-empty vector")
        for name in (
            "selected_policy_action",
            "selected_control_action",
            "final_executed_policy_action",
            "final_executed_action",
            "interpolated_action",
            "action_processor_output",
            "command_sent_to_robot",
            "measured_joint_position",
        ):
            info = self.array_info[name]
            if len(info.shape) != 2 or info.shape[1] != action_keys.shape[0]:
                raise ValueError(f"{name} must have shape [commands, action_keys]")

        for name in (
            "candidate_bank_policy_actions",
            "candidate_bank_control_actions",
            "candidate_bank_dreamed_matching_features",
        ):
            if len(self.array_info[name].shape) != 4:
                raise ValueError(f"{name} must have shape [banks,N,H,D]")
        request_noise = self.array_info["async_request_generation_noise"]
        if len(request_noise.shape) != 4:
            raise ValueError("async_request_generation_noise must be rank four")
        for name in (
            "async_request_generated_policy_actions",
            "async_request_generated_control_actions",
        ):
            if name in self.array_info:
                info = self.array_info[name]
                if len(info.shape) != 4 or info.shape[0] not in {0, self.request_count}:
                    raise ValueError(f"{name} must have shape [requests,N,H,D] when present")
        rtc_prefix = self.array_info["async_request_rtc_prefix_actions"]
        if len(rtc_prefix.shape) != 4:
            raise ValueError("async_request_rtc_prefix_actions must be rank four")

        sparse_count = self.array_info["sparse_observation_event_role"].shape[0]
        for item in self._sparse_key_metadata():
            array_name = str(item.get("array", ""))
            if array_name not in self.array_info:
                raise ValueError(f"sparse observation value array is missing: {array_name!r}")
            info = self.array_info[array_name]
            expected_shape = tuple(int(value) for value in item.get("shape_per_capture", ()))
            if info.shape != (sparse_count, *expected_shape):
                raise ValueError(f"sparse observation {array_name} has inconsistent shape")
            if str(info.dtype) != str(item.get("dtype")):
                raise ValueError(f"sparse observation {array_name} has inconsistent dtype")

    def _validate_group(self, fields: tuple[str, ...], label: str) -> None:
        lengths = {name: self.array_info[name].shape[0] for name in fields}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{label} arrays have inconsistent leading lengths: {lengths}")


def load_diagnostic_capture(path: str | Path) -> DiagnosticCapture:
    """Open and validate one final format-16 diagnostic artifact."""

    return DiagnosticCapture(path)


__all__ = [
    "ASYNC_REQUEST_FIELDS",
    "ArrayInfo",
    "CANDIDATE_BANK_FIELDS",
    "DIAGNOSTIC_FORMAT_VERSION",
    "DREAMHANDOFF_DIAGNOSTIC_FORMAT",
    "DiagnosticCapture",
    "HISTORICAL_DIAGNOSTIC_FORMAT",
    "HandoffRecord",
    "PER_COMMAND_FIELDS",
    "REQUIRED_FORMAT16_FIELDS",
    "RequestRecord",
    "SPARSE_OBSERVATION_FIELDS",
    "WP6_REQUIRED_FIELDS",
    "load_diagnostic_capture",
    "sha256_file",
]
