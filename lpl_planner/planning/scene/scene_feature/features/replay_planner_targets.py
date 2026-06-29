from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType


def to_tensor(data: FeatureDataType, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(dtype)
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype)
    raise ValueError(f"Unknown type: {type(data)}")


@dataclass
class ReplayPlannerTargets(AbstractModelFeature):
    anchor_indices: FeatureDataType
    teacher_logits: FeatureDataType

    @classmethod
    def get_feature_unique_name(cls) -> str:
        return "replay_planner_targets"

    def to_feature_tensor(self) -> ReplayPlannerTargets:
        return ReplayPlannerTargets(
            anchor_indices=to_tensor(self.anchor_indices, dtype=torch.int32),
            teacher_logits=to_tensor(self.teacher_logits, dtype=torch.float16),
        )

    def to_device(self, device: torch.device) -> ReplayPlannerTargets:
        return ReplayPlannerTargets(
            anchor_indices=self.anchor_indices.to(device=device),
            teacher_logits=self.teacher_logits.to(device=device),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> ReplayPlannerTargets:
        return ReplayPlannerTargets(
            anchor_indices=data["anchor_indices"],
            teacher_logits=data["teacher_logits"],
        )

    def unpack(self) -> List[ReplayPlannerTargets]:
        batch_size = self.anchor_indices.shape[0]
        unpacked_targets: List[ReplayPlannerTargets] = []
        for idx in range(batch_size):
            unpacked_targets.append(
                ReplayPlannerTargets(
                    anchor_indices=self.anchor_indices[idx],
                    teacher_logits=self.teacher_logits[idx],
                )
            )
        return unpacked_targets

    def collate(self, batch: List[ReplayPlannerTargets]) -> ReplayPlannerTargets:
        max_count = max(int(item.anchor_indices.numel()) for item in batch)
        padded_indices: List[torch.Tensor] = []
        padded_logits: List[torch.Tensor] = []
        for item in batch:
            indices = item.anchor_indices.reshape(-1)
            logits = item.teacher_logits.reshape(-1)
            if indices.numel() != logits.numel():
                raise ValueError(
                    f"ReplayPlannerTargets anchor_indices/logits length mismatch: "
                    f"{indices.numel()} vs {logits.numel()}"
                )
            if indices.numel() == 0:
                indices = torch.zeros((1,), dtype=torch.int32, device=item.anchor_indices.device)
                logits = torch.full((1,), -1.0e4, dtype=item.teacher_logits.dtype, device=item.teacher_logits.device)

            pad_count = max_count - int(indices.numel())
            if pad_count > 0:
                pad_indices = indices[-1:].repeat(pad_count)
                pad_logits = torch.full((pad_count,), -1.0e4, dtype=logits.dtype, device=logits.device)
                indices = torch.cat([indices, pad_indices], dim=0)
                logits = torch.cat([logits, pad_logits], dim=0)
            padded_indices.append(indices)
            padded_logits.append(logits)

        return ReplayPlannerTargets(
            anchor_indices=torch.stack(padded_indices, dim=0),
            teacher_logits=torch.stack(padded_logits, dim=0),
        )
