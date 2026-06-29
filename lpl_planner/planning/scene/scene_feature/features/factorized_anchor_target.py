from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType


def to_tensor(data: FeatureDataType, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(dtype)
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype)
    raise ValueError(f"Unknown type: {type(data)}")


@dataclass
class FactorizedAnchorTarget(AbstractModelFeature):
    path_target: FeatureDataType
    path_target_mask: FeatureDataType
    velocity_target: FeatureDataType
    velocity_target_mask: FeatureDataType
    path_anchor_indice: FeatureDataType
    vel_anchor_indice: FeatureDataType

    @classmethod
    def get_feature_unique_name(cls) -> str:
        return "factorized_anchor_target"

    def to_feature_tensor(self) -> FactorizedAnchorTarget:
        return FactorizedAnchorTarget(
            path_target=to_tensor(self.path_target, dtype=torch.float32),
            path_target_mask=to_tensor(self.path_target_mask, dtype=torch.bool),
            velocity_target=to_tensor(self.velocity_target, dtype=torch.float32),
            velocity_target_mask=to_tensor(self.velocity_target_mask, dtype=torch.bool),
            path_anchor_indice=to_tensor(self.path_anchor_indice, dtype=torch.int32),
            vel_anchor_indice=to_tensor(self.vel_anchor_indice, dtype=torch.int32),
        )

    def to_device(self, device: torch.device) -> FactorizedAnchorTarget:
        validate_type(self.path_target, torch.Tensor)
        validate_type(self.path_target_mask, torch.Tensor)
        validate_type(self.velocity_target, torch.Tensor)
        validate_type(self.velocity_target_mask, torch.Tensor)
        validate_type(self.path_anchor_indice, torch.Tensor)
        validate_type(self.vel_anchor_indice, torch.Tensor)
        return FactorizedAnchorTarget(
            path_target=self.path_target.to(device=device),
            path_target_mask=self.path_target_mask.to(device=device),
            velocity_target=self.velocity_target.to(device=device),
            velocity_target_mask=self.velocity_target_mask.to(device=device),
            path_anchor_indice=self.path_anchor_indice.to(device=device),
            vel_anchor_indice=self.vel_anchor_indice.to(device=device),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> FactorizedAnchorTarget:
        return FactorizedAnchorTarget(
            path_target=data["path_target"],
            path_target_mask=data["path_target_mask"],
            velocity_target=data["velocity_target"],
            velocity_target_mask=data["velocity_target_mask"],
            path_anchor_indice=data["path_anchor_indice"],
            vel_anchor_indice=data["vel_anchor_indice"],
        )

    def unpack(self) -> List[FactorizedAnchorTarget]:
        return [
            FactorizedAnchorTarget(
                path_target=path_target[None],
                path_target_mask=path_mask[None],
                velocity_target=velocity_target[None],
                velocity_target_mask=velocity_mask[None],
                path_anchor_indice=path_indice[None],
                vel_anchor_indice=vel_indice[None],
            )
            for path_target, path_mask, velocity_target, velocity_mask, path_indice, vel_indice in zip(
                self.path_target,
                self.path_target_mask,
                self.velocity_target,
                self.velocity_target_mask,
                self.path_anchor_indice,
                self.vel_anchor_indice,
            )
        ]

    def collate(self, batch: List[FactorizedAnchorTarget]) -> FactorizedAnchorTarget:
        return FactorizedAnchorTarget(
            path_target=torch.stack([item.path_target for item in batch], dim=0),
            path_target_mask=torch.stack([item.path_target_mask for item in batch], dim=0),
            velocity_target=torch.stack([item.velocity_target for item in batch], dim=0),
            velocity_target_mask=torch.stack([item.velocity_target_mask for item in batch], dim=0),
            path_anchor_indice=torch.stack([item.path_anchor_indice for item in batch], dim=0),
            vel_anchor_indice=torch.stack([item.vel_anchor_indice for item in batch], dim=0),
        )