from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dream_handoff.capture import DiagnosticActionProcessor, DiagnosticRobotWrapper


class RecordingCapture:
    def __init__(self) -> None:
        self.processor_calls = []
        self.observations = []
        self.commands = []

    def record_action_processor(self, processor_input, processor_output) -> None:
        self.processor_calls.append((processor_input, processor_output))

    def record_observation(self, observation) -> None:
        self.observations.append(observation)

    def record_command_sent(self, command, **identity) -> None:
        self.commands.append((command, identity))


class Processor:
    def __init__(self, result: object, error: BaseException | None = None) -> None:
        self.result = result
        self.error = error
        self.calls = []
        self.reset_calls = 0

    def __call__(self, value):
        self.calls.append(value)
        if self.error is not None:
            raise self.error
        return self.result

    def reset(self) -> None:
        self.reset_calls += 1


class Robot:
    robot_type = "fake"

    def __init__(
        self,
        observation: object,
        send_result: object,
        *,
        observation_error: BaseException | None = None,
        send_error: BaseException | None = None,
    ) -> None:
        self.observation = observation
        self.send_result = send_result
        self.observation_error = observation_error
        self.send_error = send_error
        self.observation_calls = 0
        self.send_calls = []
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.is_connected = False

    def get_observation(self):
        self.observation_calls += 1
        if self.observation_error is not None:
            raise self.observation_error
        return self.observation

    def send_action(self, action):
        self.send_calls.append(action)
        if self.send_error is not None:
            raise self.send_error
        return self.send_result

    def connect(self) -> None:
        self.connect_calls += 1
        self.is_connected = True

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.is_connected = False


def test_action_processor_delegates_once_and_preserves_identity_and_values() -> None:
    result = {"joint": torch.tensor(3.0)}
    wrapped = Processor(result)
    capture = RecordingCapture()
    processor = DiagnosticActionProcessor(wrapped, capture)
    action = {"joint": np.asarray(2.0, dtype=np.float32)}
    observation = {"camera": np.zeros((2, 2, 3), dtype=np.uint8)}
    action_before = action["joint"].copy()
    observation_before = observation["camera"].copy()

    returned = processor((action, observation))

    assert returned is result
    assert wrapped.calls == [(action, observation)]
    assert wrapped.calls[0][0] is action
    assert wrapped.calls[0][1] is observation
    assert capture.processor_calls == [(action, result)]
    np.testing.assert_array_equal(action["joint"], action_before)
    np.testing.assert_array_equal(observation["camera"], observation_before)
    processor.reset()
    assert wrapped.reset_calls == 1


def test_action_processor_propagates_exact_exception_without_recording() -> None:
    error = RuntimeError("processor failed")
    wrapped = Processor(object(), error)
    capture = RecordingCapture()
    processor = DiagnosticActionProcessor(wrapped, capture)

    with pytest.raises(RuntimeError) as caught:
        processor(({"joint": 1.0}, {}))

    assert caught.value is error
    assert len(wrapped.calls) == 1
    assert capture.processor_calls == []


def test_robot_wrapper_reads_and_dispatches_exactly_once_and_delegates_state() -> None:
    observation = {"joint": 1.0, "camera": np.zeros((2, 2, 3), dtype=np.uint8)}
    send_result = object()
    robot = Robot(observation, send_result)
    capture = RecordingCapture()
    dataset = SimpleNamespace(
        writer=SimpleNamespace(episode_buffer={"episode_index": 5, "size": 9})
    )
    wrapper = DiagnosticRobotWrapper(robot, capture, dataset=dataset)

    wrapper.connect()
    assert wrapper.is_connected is True
    returned_observation = wrapper.get_observation()
    command = {"joint": 2.0}
    returned_send = wrapper.send_action(command)
    wrapper.disconnect()

    assert returned_observation is observation
    assert returned_send is send_result
    assert robot.observation_calls == 1
    assert robot.send_calls == [command]
    assert robot.send_calls[0] is command
    assert capture.observations == [observation]
    assert capture.commands == [(command, {"dataset_episode_index": 5, "dataset_frame_index": 9})]
    assert robot.connect_calls == robot.disconnect_calls == 1
    assert wrapper.is_connected is False


@pytest.mark.parametrize("method", ["get_observation", "send_action"])
def test_robot_wrapper_propagates_exact_exception(method: str) -> None:
    error = RuntimeError(f"{method} failed")
    robot = Robot(
        {"joint": 1.0},
        object(),
        observation_error=error if method == "get_observation" else None,
        send_error=error if method == "send_action" else None,
    )
    capture = RecordingCapture()
    wrapper = DiagnosticRobotWrapper(robot, capture)

    with pytest.raises(RuntimeError) as caught:
        if method == "get_observation":
            wrapper.get_observation()
        else:
            wrapper.send_action({"joint": 2.0})

    assert caught.value is error
    assert capture.observations == []
    assert capture.commands == []
    assert robot.observation_calls == (1 if method == "get_observation" else 0)
    assert len(robot.send_calls) == (1 if method == "send_action" else 0)
