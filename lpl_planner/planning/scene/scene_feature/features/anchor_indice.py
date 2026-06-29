from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import numpy as np

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
class AnchorIndice(AbstractModelFeature):
    """
    Class that holds the scores for each trajectory.
    """
    indice: FeatureDataType  # shape: ((B),K, )

    
    @classmethod
    def get_feature_unique_name(cls) -> str:
        return "anchor_indice"

    def to_device(self, device: torch.device) -> AnchorIndice:
        """Implemented. See interface."""
        validate_type(self.indice, torch.Tensor)
        return AnchorIndice(indice=self.indice.to(device=device))

    def to_feature_tensor(self, dtype: torch.dtype = torch.int32) -> AnchorIndice:
        """Inherited, see superclass."""
        return AnchorIndice(indice=to_tensor(self.indice, dtype=dtype))
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AnchorIndice:
        """Implemented. See interface."""
        return AnchorIndice(indice=data["indice"])
    def unpack(self) -> List[AnchorIndice]:
        """Implemented. See interface."""
        return [AnchorIndice(indice=data[None]) for data in self.indice]
