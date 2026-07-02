from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor,
)
import dataclasses

@dataclass
class AgentPrediction(AbstractModelFeature):
    """
    Class that holds the agent future state prediction.
    """
    agent_future_state: FeatureDataType # [(B), n_agent, horizon, pos]
    agent_future_mask: FeatureDataType # [(B), n_agent, horizon]

    @classmethod
    def collate(cls, batch: List[AgentPrediction]) -> AgentPrediction:
        """
        Collate a batch of AgentPrediction into a single AgentPrediction with padded agent dimension.
        :param batch: features to be batched
        :return: batched features together
        """
        B = len(batch)
        agent_list = [item.agent_future_state for item in batch]  # [n_agent, horizon, (x, y, yaw, vx, vy)]
        mask_list = [item.agent_future_mask for item in batch]    # [n_agent, horizon]
        max_n_agent = max(a.shape[0] for a in agent_list)
        horizon = agent_list[0].shape[1]
        pos = agent_list[0].shape[2]

        dtype = agent_list[0].dtype
        device = agent_list[0].device

        batched_agent_future_state = torch.zeros((B, max_n_agent, horizon, pos), dtype=dtype, device=device)
        batched_agent_future_mask = torch.zeros((B, max_n_agent, horizon), dtype=dtype, device=device)

        for i, (a, m) in enumerate(zip(agent_list, mask_list)):
            n = a.shape[0]
            batched_agent_future_state[i, :n] = a
            batched_agent_future_mask[i, :n] = m

        return AgentPrediction(
            agent_future_state=batched_agent_future_state,
            agent_future_mask=batched_agent_future_mask,
        )
    
    def to_feature_tensor(self) -> AgentPrediction:
        """
        :return object which will be collated into a batch
        """
        agent_feature_state_tensor = to_tensor(self.agent_future_state)
        agent_feature_mask_tensor = to_tensor(self.agent_future_mask)
        return AgentPrediction(
            agent_future_state=agent_feature_state_tensor,
            agent_future_mask=agent_feature_mask_tensor,
        )

    def to_device(self, device: torch.device) -> AgentPrediction:
        """
        :param device: desired device to move feature to
        :return feature type that was moved to a device
        """
        return AgentPrediction(
            agent_future_state=self.agent_future_state.to(device),
            agent_future_mask=self.agent_future_mask.to(device),
        )


    def serialize(self) -> Dict[str, Any]:
        """
        :return: Return dictionary of data that can be serialized
        """
        return dataclasses.asdict(self)

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AgentPrediction:
        """
        :return: Return dictionary of data that can be serialized
        """
        return cls(
            agent_future_state=data['agent_future_state'],
            agent_future_mask=data['agent_future_mask'],
        )


    def unpack(self) -> List[AbstractModelFeature]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        features: List[AgentPrediction] = []
        B, N, H, D = self.agent_future_state.shape

        for b in range(B):
            agent_future_state = self.agent_future_state[b]      # [N,H,D]
            agent_future_mask = self.agent_future_mask[b]        # [N,H]

            any_valid = agent_future_mask.any(dim=1)             # [N]

            if any_valid.any():
                last_valid_idx = int(torch.nonzero(any_valid, as_tuple=False)[-1].item())
                kept_state = agent_future_state[:last_valid_idx + 1]
                kept_mask = agent_future_mask[:last_valid_idx + 1]
            else:
                kept_state = agent_future_state[:0]
                kept_mask = agent_future_mask[:0]

            features.append(
                AgentPrediction(
                    agent_future_state=kept_state,
                    agent_future_mask=kept_mask,
                )
            )
        return features

    @classmethod
    def get_feature_unique_name(cls) -> str:
        return "agent_prediction"
