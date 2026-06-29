from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch

from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature


@dataclass
class SceneToken(AbstractModelFeature):
    """
    A feature that contains multiple trajectories
    """

    token: Union[List[str], str]

    @classmethod
    def get_feature_unique_name(cls) -> str:
        return "scene_token"

    
    @property
    def number_of_trajectories(self) -> int:
        """
        :return: number of trajectories in this feature.
        """
        return len(self.token) if isinstance(self.token, list) else 1
    
    def to_feature_tensor(self) -> SceneToken:
        """Implemented. See interface."""
        return SceneToken(token=self.token)

    def to_device(self, device: torch.device) -> SceneToken:
        """Implemented. See interface."""
        return SceneToken(token=self.token)

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> SceneToken:
        """Implemented. See interface."""
        return SceneToken(token=data["token"])

    def unpack(self) -> List[SceneToken]:
        """Implemented. See interface."""
        # If already a tensor (e.g., from collate), remove padded rows and split batch
        if isinstance(self.token, list):
            return [SceneToken(token=t) for t in self.token]
        else:
            return [SceneToken(token=self.token)]

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> SceneToken:
        """Implemented. See interface."""
        return SceneToken(token=data["token"])

    
    
        
