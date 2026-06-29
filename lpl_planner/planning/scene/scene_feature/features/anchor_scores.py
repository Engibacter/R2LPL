from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

import numpy as np

from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType

def to_tensor(data: FeatureDataType, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert data to tensor
    :param data which is either numpy or Tensor
    :return torch.Tensor
    """
    if isinstance(data, torch.Tensor):
        return data.to(dtype)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype)
    else:
        raise ValueError(f"Unknown type: {type(data)}")
    
@dataclass
class AnchorScores(AbstractModelFeature):
    """
    A feature that contains scores for multiple anchor trajectories
    """

    aggregated_scores: FeatureDataType # [(B), num_anchors]
    # multi_scores: FeatureDataType # [(B), num_metrics, num_anchors]
    # weighted_scores: FeatureDataType # [(B), num_metrics, num_anchors]

    @classmethod
    def get_feature_unique_name(cls) -> str:
        return "anchor_scores"

    def to_feature_tensor(self) -> AnchorScores:
        """Implemented. See interface."""
        return AnchorScores(aggregated_scores=to_tensor(self.aggregated_scores),
                            # multi_scores=to_tensor(self.multi_scores),
                            # weighted_scores=to_tensor(self.weighted_scores)
                            )
    
    def to_device(self, device: torch.device) -> AnchorScores:
        """Implemented. See interface."""
        return AnchorScores(aggregated_scores=self.aggregated_scores.to(device),
                            # multi_scores=self.multi_scores.to(device),
                            # weighted_scores=self.weighted_scores.to(device)
                            )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AnchorScores:
        """Implemented. See interface."""
        return AnchorScores(aggregated_scores=data["aggregated_scores"],
                            # multi_scores=data["multi_scores"],
                            # weighted_scores=data["weighted_scores"]
                            )


    def unpack(self) -> List[AnchorScores]:
        """Implemented. See interface."""
        batch_size = self.aggregated_scores.shape[0]
        unpacked_scores = []
        for i in range(batch_size):
            unpacked_scores.append(
                AnchorScores(
                    aggregated_scores=self.aggregated_scores[i],
                    # multi_scores=self.multi_scores[i],
                    # weighted_scores=self.weighted_scores[i]
                )
            )
        return unpacked_scores
    
    
    def collate(self, batch: List[AnchorScores]) -> torch.Tensor:
        """Implemented. See interface."""
        # Convert each item in batch to a tensor of shape [(B), ...]
        aggregated_scores = torch.stack([item.aggregated_scores for item in batch], dim=0)
        # multi_scores = torch.stack([item.multi_scores for item in batch], dim=0)
        # weighted_scores = torch.stack([item.weighted_scores for item in batch], dim=0)
        return AnchorScores(aggregated_scores=aggregated_scores,
                            # multi_scores=multi_scores,
                            # weighted_scores=weighted_scores
                            )
