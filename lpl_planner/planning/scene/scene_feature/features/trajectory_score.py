from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrajectoryScore(AbstractModelFeature):
    """
    Class that holds the scores for each trajectory.
    """
    feasible_score: FeatureDataType  # shape: ((B),3, )
    progress_score: FeatureDataType  # shape: ((B),T, )
    comfort_score: FeatureDataType  # shape: ((B),T, )
    safety_score: FeatureDataType  # shape: ((B),T, )
    # timed_progress_score: FeatureDataType # shape: (T, )
    # timed_comfort_score: FeatureDataType  # shape: (T, )
    # timed_ttc_score: FeatureDataType  # shape: (T, )

    
    @classmethod
    def get_feature_unique_name(cls) -> str:
        return "trajectory_score"

    def to_feature_tensor(self, dtype: torch.dtype = torch.float32) -> TrajectoryScore:
        """
        :return object which will be collated into a batch
        """
        feasible_score_tensor = to_tensor(self.feasible_score)
        progress_score_tensor = to_tensor(self.progress_score)
        comfort_score_tensor = to_tensor(self.comfort_score)
        safety_score_tensor = to_tensor(self.safety_score)
        
        return TrajectoryScore(
            feasible_score=feasible_score_tensor,
            progress_score=progress_score_tensor,
            comfort_score=comfort_score_tensor,
            safety_score=safety_score_tensor,
        )

    def to_device(self, device: torch.device) -> TrajectoryScore:
        """
        :param device: desired device to move feature to
        :return feature type that was moved to a device
        """
        return TrajectoryScore(
            feasible_score=self.feasible_score.to(device=device), 
            progress_score=self.progress_score.to(device=device),
            comfort_score=self.comfort_score.to(device=device),
            safety_score=self.safety_score.to(device=device)
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> TrajectoryScore:
        """
        :return: Return dictionary of data that can be serialized
        """
        return cls(
            feasible_score=data["feasible_score"],
            progress_score=data["progress_score"],
            comfort_score=data["comfort_score"],
            safety_score=data["safety_score"]
        )

    def unpack(self) -> List[AbstractModelFeature]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        raise NotImplementedError(
            "Unpacking is not implemented for SceneFeature. Please implement it if needed."
        )

    def aggregate(self, progress_weight = 5.0, comfort_weight = 5.0, safety_weight = 2.0, discount_factor = 1.0) -> torch.Tensor:
        """
        :return: Aggregate a list of features to a single feature.
        """
        horizon = self.progress_score.shape[-1]
        discounts = discount_factor ** (torch.arange(horizon, device=self.progress_score.device) * 0.1)
        aggregate_score = self.feasible_score.prod(dim=-1) * (
            (self.progress_score * discounts).mean(dim=-1) * progress_weight +
            (self.safety_score * discounts).mean(dim=-1) * safety_weight +
            (self.comfort_score * discounts).mean(dim=-1) * comfort_weight
        )  # [(batch_size), 1]

        return aggregate_score
