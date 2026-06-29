from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import numpy as np
import torch

from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


@dataclass
class Trajectories(AbstractModelFeature):
    """
    A feature that contains multiple trajectories
    """

    trajectories: Union[List[Trajectory], torch.Tensor]

    @classmethod
    def get_feature_unique_name(cls) -> str:
        return "trajectories"

    def to_feature_tensor(self) -> Trajectories:
        """Implemented. See interface."""
        return Trajectories(trajectories=[trajectory.to_feature_tensor() for trajectory in self.trajectories])

    def to_device(self, device: torch.device) -> Trajectories:
        """Implemented. See interface."""
        if isinstance(self.trajectories, torch.Tensor):
            return Trajectories(trajectories=self.trajectories.to(device)) 
        
        return Trajectories(trajectories=[trajectory.to_device(device) for trajectory in self.trajectories])

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Trajectories:
        """Implemented. See interface."""
        return Trajectories(trajectories=[Trajectory.deserialize(trajectory) for trajectory in data["trajectories"]])

    def to_numpy_array(self) -> np.ndarray:
        """Convert trajectories to list of numpy arrays."""
        return np.array([trajectory for trajectory in self.trajectories])
    
    @property
    def number_of_trajectories(self) -> int:
        """
        :return: number of trajectories in this feature.
        """
        return len(self.trajectories)

    def unpack(self) -> List[Trajectories]:
        """Implemented. See interface."""
        # If already a tensor (e.g., from collate), remove padded rows and split batch
        if isinstance(self.trajectories, torch.Tensor):
            x = self.trajectories
            out = []
            for b in range(x.shape[0]):
                m = x[b]  # [M, T, S]
                if m.numel() == 0:
                    out.append(Trajectories(trajectories=m))
                    continue
                valid = (m != 0).any(dim=-1).any(dim=-1)  # [M]
                out.append(Trajectories(trajectories=m[valid]))
                return out
            
        # Fallback for list of Trajectory objects
        return [Trajectories([t]) for t in self.trajectories]

    
    def collate(self, batch: List[Trajectories]) -> torch.Tensor:
        """Implemented. See interface."""
        # Convert each item in batch to a tensor of shape [M_i, T, S]

        dtype = batch[0].trajectories[0].data.dtype if batch[0].trajectories else torch.float32
        T, S = batch[0].trajectories[0].data.shape[-2:] 
        max_M = max([trajs.number_of_trajectories for trajs in batch])
        B = len(batch)
        # Allocate and pad to [B, M_max, T, S]
        out = torch.zeros((B, max_M, T, S), dtype=dtype)
        for i, t in enumerate(batch):
            M_i = t.number_of_trajectories
            out[i, :M_i] = Trajectory.collate(t.trajectories).data

        return Trajectories(trajectories=out)
        
