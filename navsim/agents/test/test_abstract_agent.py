"""Unit tests for the macOS/MPS device-selection logic in AbstractAgent.

AbstractAgent is abstract (it declares abstract `name`, `get_sensor_config`, and
`initialize` methods), so a minimal concrete subclass is used to exercise `__init__`
and `compute_trajectory` without pulling in a real model, dataset, or checkpoint.
"""

import importlib
import os
from typing import Dict, List
from unittest.mock import MagicMock, patch

import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents import abstract_agent as abstract_agent_module
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder


class _MinimalTestAgent(AbstractAgent):
    """Smallest possible concrete AbstractAgent, used only to exercise device-selection logic."""

    def name(self) -> str:
        """Inherited, see superclass."""
        return "minimal_test_agent"

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_no_sensors()

    def initialize(self) -> None:
        """Inherited, see superclass."""

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass.

        Deliberately returns no builders: compute_trajectory's feature-building loop
        becomes a no-op, so the `agent_input` passed to compute_trajectory in these
        tests never needs to be a real AgentInput.
        """
        return []

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass. Returns a fixed-shape stand-in prediction."""
        return {"trajectory": torch.zeros(1, self._trajectory_sampling.num_poses, 3)}


def _make_agent() -> _MinimalTestAgent:
    trajectory_sampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    return _MinimalTestAgent(trajectory_sampling=trajectory_sampling)


class TestDeviceSelection:
    """AbstractAgent.__init__ picks a device in priority order: CUDA, then MPS, then CPU,
    unless NAVSIM_DEVICE forces a specific one."""

    def setup_method(self):
        self._orig_navsim_device = os.environ.pop("NAVSIM_DEVICE", None)

    def teardown_method(self):
        if self._orig_navsim_device is not None:
            os.environ["NAVSIM_DEVICE"] = self._orig_navsim_device

    def test_uses_cuda_when_available_even_if_mps_also_available(self):
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.backends.mps.is_available", return_value=True
        ):
            agent = _make_agent()
        assert agent._inference_device == torch.device("cuda")

    def test_uses_mps_when_cuda_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=True
        ):
            agent = _make_agent()
        assert agent._inference_device == torch.device("mps")

    def test_uses_cpu_when_neither_cuda_nor_mps_available(self):
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=False
        ):
            agent = _make_agent()
        assert agent._inference_device == torch.device("cpu")

    def test_navsim_device_env_var_overrides_auto_detection(self):
        os.environ["NAVSIM_DEVICE"] = "cpu"
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.backends.mps.is_available", return_value=True
        ):
            agent = _make_agent()
        assert agent._inference_device == torch.device(
            "cpu"
        ), "NAVSIM_DEVICE must override auto-detection even when a faster device is available"


class TestMoveToDeviceIsIdempotent:
    """compute_trajectory should move the model to the inference device at most once."""

    def test_to_is_only_called_once_across_multiple_compute_trajectory_calls(self):
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.backends.mps.is_available", return_value=False
        ):
            agent = _make_agent()

        assert agent._moved_to_inference_device is False

        # Shadow the instance's bound `to` method with a spy that still performs the
        # real move, so we can assert on call count without changing behavior.
        spy_to = MagicMock(wraps=agent.to)
        agent.to = spy_to

        agent.compute_trajectory(agent_input=object())
        assert spy_to.call_count == 1
        assert agent._moved_to_inference_device is True

        agent.compute_trajectory(agent_input=object())
        agent.compute_trajectory(agent_input=object())
        assert spy_to.call_count == 1, "self.to() must not be called again once the model has already been moved"


class TestMpsFallbackEnvVar:
    """
    `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")` runs once, at import
    time, as a module-level side effect in navsim/agents/abstract_agent.py. It's
    exercised here by clearing/seeding the environment and re-executing the module
    body via importlib.reload (the effect only fires once per process on first import,
    so a plain re-import wouldn't re-trigger it).
    """

    def setup_method(self):
        self._orig_value = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")

    def teardown_method(self):
        if self._orig_value is None:
            os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        else:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = self._orig_value
        # Leave the module's process-wide side effect consistent with the restored env
        # so later tests in the same session aren't affected by this test's reloads.
        importlib.reload(abstract_agent_module)

    def test_setdefault_sets_the_var_when_unset(self):
        os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        importlib.reload(abstract_agent_module)
        assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"

    def test_setdefault_does_not_clobber_a_user_set_value(self):
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        importlib.reload(abstract_agent_module)
        assert (
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
        ), "setdefault must not overwrite a value the user explicitly set before import"
