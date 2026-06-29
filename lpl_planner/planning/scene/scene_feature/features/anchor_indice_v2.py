from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType


def to_tensor(data: FeatureDataType, dtype: torch.dtype = torch.int32) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(dtype)
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype)
    raise ValueError(f"Unknown type: {type(data)}")


@dataclass
class AnchorIndiceV2(AbstractModelFeature):
    path_anchor_indice: FeatureDataType
    vel_anchor_indice: FeatureDataType

    @classmethod
    def get_feature_unique_name(cls) -> str:
        return "anchor_indice_v2"

    def to_device(self, device: torch.device) -> AnchorIndiceV2:
        validate_type(self.path_anchor_indice, torch.Tensor)
        validate_type(self.vel_anchor_indice, torch.Tensor)
        return AnchorIndiceV2(
            path_anchor_indice=self.path_anchor_indice.to(device=device),
            vel_anchor_indice=self.vel_anchor_indice.to(device=device),
        )

    def to_feature_tensor(self, dtype: torch.dtype = torch.int32) -> AnchorIndiceV2:
        return AnchorIndiceV2(
            path_anchor_indice=to_tensor(self.path_anchor_indice, dtype=dtype),
            vel_anchor_indice=to_tensor(self.vel_anchor_indice, dtype=dtype),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AnchorIndiceV2:
        return AnchorIndiceV2(
            path_anchor_indice=data["path_anchor_indice"],
            vel_anchor_indice=data["vel_anchor_indice"],
        )

    def unpack(self) -> List[AnchorIndiceV2]:
        return [
            AnchorIndiceV2(
                path_anchor_indice=path_data[None],
                vel_anchor_indice=vel_data[None],
            )
            for path_data, vel_data in zip(self.path_anchor_indice, self.vel_anchor_indice)
        ]