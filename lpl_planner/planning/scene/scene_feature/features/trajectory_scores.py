from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from lpl_planner.planning.scene.scene_feature.features.trajectory_score import TrajectoryScore


@dataclass
class TrajectoryScores(AbstractModelFeature):
    """
    A feature that contains multiple trajectories
    """

    trajectory_scores: List[TrajectoryScore]

    @classmethod
    def get_feature_unique_name(cls) -> str:
        return "trajectory_scores"

    def to_feature_tensor(self) -> TrajectoryScores:
        """Implemented. See interface."""
        return TrajectoryScores(trajectory_scores=[trajectory_score.to_feature_tensor() for trajectory_score in self.trajectory_scores])

    def to_device(self, device: torch.device) -> TrajectoryScores:
        """Implemented. See interface."""
        return TrajectoryScores(trajectory_scores=[trajectory_score.to_device(device) for trajectory_score in self.trajectory_scores])

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> TrajectoryScores:
        """Implemented. See interface."""
        return TrajectoryScores(trajectory_scores=[TrajectoryScore.deserialize(trajectory_score) for trajectory_score in data["trajectory_scores"]])

    @property
    def number_of_trajectories(self) -> int:
        """
        :return: number of trajectories in this feature.
        """
        return len(self.trajectory_scores)

    def unpack(self) -> List[TrajectoryScores]:
        """Implemented. See interface."""
        return [TrajectoryScores([trajectory_scores]) for trajectory_scores in self.trajectory_scores]
    
    def collate(self, batch: List[TrajectoryScores]) -> torch.Tensor:
        """Implemented. See interface."""
        # Convert each item in batch to a tensor of shape [M_i, S]
        dtype = batch[0].trajectory_scores[0].feasible_score.dtype if batch[0].trajectory_scores else torch.float32
        B = len(batch)
        max_M = max([trajs.number_of_trajectories for trajs in batch])
        # Allocate and pad to [B, M_max, S]
        out = torch.zeros((B, max_M,), dtype=dtype)
        for i, t in enumerate(batch):
            M_i = t.number_of_trajectories
            out[i, :M_i] = TrajectoryScore.collate(t.trajectory_scores).aggregate()

        return out
