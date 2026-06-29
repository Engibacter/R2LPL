from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType


def to_tensor(data: FeatureDataType, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(dtype)
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype)
    if isinstance(data, (np.float32, np.float64, np.int32, np.int64, float, int)):
        return torch.tensor(data, dtype=dtype)
    raise ValueError(f"Unknown type: {type(data)}")


@dataclass
class RolloutTeacherMetadata(AbstractModelFeature):
    sampled_anchor_indices: FeatureDataType
    sampled_teacher_sources: FeatureDataType
    sampled_trajectories: FeatureDataType
    sampled_model_log_scores: FeatureDataType
    sampled_eval_scores: FeatureDataType
    sampled_selection_scores: FeatureDataType
    chosen_local_index: FeatureDataType
    chosen_anchor_index: FeatureDataType
    chosen_score: FeatureDataType
    planner_ref_path: FeatureDataType
    expert_path: FeatureDataType
    expert_ref_path: FeatureDataType
    timestamp_us: FeatureDataType
    iteration: FeatureDataType
    emergency_brake: FeatureDataType

    @classmethod
    def get_feature_unique_name(cls) -> str:
        return "rollout_teacher_metadata"

    def to_feature_tensor(self) -> RolloutTeacherMetadata:
        return RolloutTeacherMetadata(
            sampled_anchor_indices=to_tensor(self.sampled_anchor_indices, dtype=torch.int32),
            sampled_teacher_sources=to_tensor(self.sampled_teacher_sources, dtype=torch.int32),
            sampled_trajectories=to_tensor(self.sampled_trajectories, dtype=torch.float32),
            sampled_model_log_scores=to_tensor(self.sampled_model_log_scores, dtype=torch.float32),
            sampled_eval_scores=to_tensor(self.sampled_eval_scores, dtype=torch.float32),
            sampled_selection_scores=to_tensor(self.sampled_selection_scores, dtype=torch.float32),
            chosen_local_index=to_tensor(self.chosen_local_index, dtype=torch.int32),
            chosen_anchor_index=to_tensor(self.chosen_anchor_index, dtype=torch.int32),
            chosen_score=to_tensor(self.chosen_score, dtype=torch.float32),
            planner_ref_path=to_tensor(self.planner_ref_path, dtype=torch.float32),
            expert_path=to_tensor(self.expert_path, dtype=torch.float32),
            expert_ref_path=to_tensor(self.expert_ref_path, dtype=torch.float32),
            timestamp_us=to_tensor(self.timestamp_us, dtype=torch.int64),
            iteration=to_tensor(self.iteration, dtype=torch.int32),
            emergency_brake=to_tensor(self.emergency_brake, dtype=torch.int32),
        )

    def to_device(self, device: torch.device) -> RolloutTeacherMetadata:
        return RolloutTeacherMetadata(
            sampled_anchor_indices=self.sampled_anchor_indices.to(device=device),
            sampled_teacher_sources=self.sampled_teacher_sources.to(device=device),
            sampled_trajectories=self.sampled_trajectories.to(device=device),
            sampled_model_log_scores=self.sampled_model_log_scores.to(device=device),
            sampled_eval_scores=self.sampled_eval_scores.to(device=device),
            sampled_selection_scores=self.sampled_selection_scores.to(device=device),
            chosen_local_index=self.chosen_local_index.to(device=device),
            chosen_anchor_index=self.chosen_anchor_index.to(device=device),
            chosen_score=self.chosen_score.to(device=device),
            planner_ref_path=self.planner_ref_path.to(device=device),
            expert_path=self.expert_path.to(device=device),
            expert_ref_path=self.expert_ref_path.to(device=device),
            timestamp_us=self.timestamp_us.to(device=device),
            iteration=self.iteration.to(device=device),
            emergency_brake=self.emergency_brake.to(device=device),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> RolloutTeacherMetadata:
        return RolloutTeacherMetadata(
            sampled_anchor_indices=data["sampled_anchor_indices"],
            sampled_teacher_sources=data.get("sampled_teacher_sources", np.zeros_like(data["sampled_anchor_indices"])),
            sampled_trajectories=data["sampled_trajectories"],
            sampled_model_log_scores=data["sampled_model_log_scores"],
            sampled_eval_scores=data["sampled_eval_scores"],
            sampled_selection_scores=data["sampled_selection_scores"],
            chosen_local_index=data["chosen_local_index"],
            chosen_anchor_index=data["chosen_anchor_index"],
            chosen_score=data["chosen_score"],
            planner_ref_path=data["planner_ref_path"],
            expert_path=data["expert_path"],
            expert_ref_path=data["expert_ref_path"],
            timestamp_us=data["timestamp_us"],
            iteration=data["iteration"],
            emergency_brake=data["emergency_brake"],
        )

    def unpack(self) -> List[RolloutTeacherMetadata]:
        batch_size = self.sampled_anchor_indices.shape[0]
        unpacked: List[RolloutTeacherMetadata] = []
        for idx in range(batch_size):
            unpacked.append(
                RolloutTeacherMetadata(
                    sampled_anchor_indices=self.sampled_anchor_indices[idx],
                    sampled_teacher_sources=self.sampled_teacher_sources[idx],
                    sampled_trajectories=self.sampled_trajectories[idx],
                    sampled_model_log_scores=self.sampled_model_log_scores[idx],
                    sampled_eval_scores=self.sampled_eval_scores[idx],
                    sampled_selection_scores=self.sampled_selection_scores[idx],
                    chosen_local_index=self.chosen_local_index[idx],
                    chosen_anchor_index=self.chosen_anchor_index[idx],
                    chosen_score=self.chosen_score[idx],
                    planner_ref_path=self.planner_ref_path[idx],
                    expert_path=self.expert_path[idx],
                    expert_ref_path=self.expert_ref_path[idx],
                    timestamp_us=self.timestamp_us[idx],
                    iteration=self.iteration[idx],
                    emergency_brake=self.emergency_brake[idx],
                )
            )
        return unpacked

    def collate(self, batch: List[RolloutTeacherMetadata]) -> RolloutTeacherMetadata:
        return RolloutTeacherMetadata(
            sampled_anchor_indices=torch.stack([item.sampled_anchor_indices for item in batch], dim=0),
            sampled_teacher_sources=torch.stack([item.sampled_teacher_sources for item in batch], dim=0),
            sampled_trajectories=torch.stack([item.sampled_trajectories for item in batch], dim=0),
            sampled_model_log_scores=torch.stack([item.sampled_model_log_scores for item in batch], dim=0),
            sampled_eval_scores=torch.stack([item.sampled_eval_scores for item in batch], dim=0),
            sampled_selection_scores=torch.stack([item.sampled_selection_scores for item in batch], dim=0),
            chosen_local_index=torch.stack([item.chosen_local_index for item in batch], dim=0),
            chosen_anchor_index=torch.stack([item.chosen_anchor_index for item in batch], dim=0),
            chosen_score=torch.stack([item.chosen_score for item in batch], dim=0),
            planner_ref_path=torch.stack([item.planner_ref_path for item in batch], dim=0),
            expert_path=torch.stack([item.expert_path for item in batch], dim=0),
            expert_ref_path=torch.stack([item.expert_ref_path for item in batch], dim=0),
            timestamp_us=torch.stack([item.timestamp_us for item in batch], dim=0),
            iteration=torch.stack([item.iteration for item in batch], dim=0),
            emergency_brake=torch.stack([item.emergency_brake for item in batch], dim=0),
        )