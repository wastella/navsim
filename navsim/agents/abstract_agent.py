# Modified from the upstream autonomousvision/navsim file of the same name (Apache-2.0):
# adds cross-platform inference device selection (CUDA/MPS/CPU) in place of the
# original CPU-only inference path. See docs/install.md's "macOS / Apple Silicon"
# section for background.
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

# MPS lacks kernels for some ops in torch==2.0.1; fall back to CPU for those instead of
# crashing. Only affects ops with no MPS kernel at all, so it never changes results, only
# whether unsupported ops run on CPU instead of erroring. Users debugging MPS issues can
# override by unsetting this before running.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _resolve_inference_device() -> torch.device:
    """
    Picks the inference device for AbstractAgent.compute_trajectory.
    Set NAVSIM_DEVICE=cuda|mps|cpu to force a specific device (e.g. to force CPU for
    debugging). Otherwise auto-detects in priority order: CUDA, then MPS, then CPU.
    """
    forced = os.environ.get("NAVSIM_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AbstractAgent(torch.nn.Module, ABC):
    """Interface for an agent in NAVSIM."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        requires_scene: bool = False,
    ):
        super().__init__()
        self.requires_scene = requires_scene
        self._trajectory_sampling = trajectory_sampling
        self._inference_device = _resolve_inference_device()
        self._moved_to_inference_device = False

    @abstractmethod
    def name(self) -> str:
        """
        :return: string describing name of this agent.
        """

    @abstractmethod
    def get_sensor_config(self) -> SensorConfig:
        """
        :return: Dataclass defining the sensor configuration for lidar and cameras.
        """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize agent
        :param initialization: Initialization class.
        """

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the agent.
        :param features: Dictionary of features.
        :return: Dictionary of predictions.
        """
        raise NotImplementedError

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """
        :return: List of target builders.
        """
        raise NotImplementedError("No feature builders. Agent does not support training.")

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """
        :return: List of feature builders.
        """
        raise NotImplementedError("No target builders. Agent does not support training.")

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        device = self._inference_device
        if not self._moved_to_inference_device:
            self.to(device)
            self._moved_to_inference_device = True

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0).to(device) for k, v in features.items()}

        # forward pass
        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["trajectory"].squeeze(0).cpu().numpy()

        # extract trajectory
        return Trajectory(poses, self._trajectory_sampling)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the loss used for backpropagation based on the features, targets and model predictions.
        """
        raise NotImplementedError("No loss. Agent does not support training.")

    def get_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, Dict[str, Union[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]],]:
        """
        Returns the optimizers that are used by thy pytorch-lightning trainer.
        Has to be either a single optimizer or a dict of optimizer and lr scheduler.
        """
        raise NotImplementedError("No optimizers. Agent does not support training.")

    def get_training_callbacks(self) -> List[pl.Callback]:
        """
        Returns a list of pytorch-lightning callbacks that are used during training.
        See navsim.planning.training.callbacks for examples.
        """
        return []
