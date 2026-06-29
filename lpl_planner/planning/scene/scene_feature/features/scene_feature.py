from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F 
import numpy as np
import pickle

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
class SceneFeature(AbstractModelFeature):
    """
    Class that holds the scene features.
    """
    ego_feature: EgoFeature
    agent_feature: AgentFeature
    static_obstacle_feature: StaticObstacleFeature
    road_feature: RoadFeature
    route_feature: RouteFeature
    ref_path_feature: Optional[FeatureDataType] = None  # [(B), n_waypoints, 6] (x, y, yaw, left_bound, right_bound, speed_limit)
    ref_path_feature_mask: Optional[FeatureDataType] = field(default_factory=lambda: np.array([], dtype=bool))  # [(B), n_waypoints]
    rasterized_feature: Optional[FeatureDataType] = None  # [(B), Channel, H, W]

    @classmethod
    def get_feature_unique_name(cls) -> str:
        return "scene_feature"

    @classmethod
    def collate(cls, batch: List[SceneFeature]) -> SceneFeature:
        """
        Batch features together with a default_collate function
        :param batch: features to be batched
        :return: batched features together
        """
        device = batch[0].ego_feature.ego_current_state.device if isinstance(batch[0].ego_feature.ego_current_state, torch.Tensor) else torch.device('cpu')
        
        ref_path_features = [feature.ref_path_feature for feature in batch]
        non_empty_ref_paths = [feature for feature in ref_path_features if feature is not None]
        if non_empty_ref_paths:
            sample_ref_path = to_tensor(non_empty_ref_paths[0]).to(device=device)
            max_n_waypoints = max(
                feature.shape[0] if feature is not None else 0 for feature in ref_path_features
            )
            batched_ref_path = torch.zeros(
                (len(batch), max_n_waypoints, sample_ref_path.shape[-1]),
                dtype=sample_ref_path.dtype,
                device=device,
            )
            ref_path_mask = torch.zeros(len(batch), max_n_waypoints, dtype=torch.bool, device=device)
            for batch_index, ref_path_feature in enumerate(ref_path_features):
                if ref_path_feature is None:
                    continue
                ref_path_tensor = to_tensor(ref_path_feature).to(device=device)
                valid_length = ref_path_tensor.shape[0]
                batched_ref_path[batch_index, :valid_length] = ref_path_tensor
                ref_path_mask[batch_index, :valid_length] = True
        else:
            batched_ref_path = torch.zeros((len(batch), 0, 6), dtype=torch.float32, device=device)
            ref_path_mask = torch.zeros((len(batch), 0), dtype=torch.bool, device=device)

        # batch ego features
        batched_ego_feature = EgoFeature.collate([f.ego_feature for f in batch])
        # batch agent features
        batched_agent_feature = AgentFeature.collate([f.agent_feature for f in batch])
        # batch static obstacle features
        batched_static_obstacle_feature = StaticObstacleFeature.collate([f.static_obstacle_feature for f in batch])
        # batch road features
        batched_road_feature = RoadFeature.collate([f.road_feature for f in batch])
        # batch route features
        batched_route_feature = RouteFeature.collate([f.route_feature for f in batch])
        # Create the SceneFeature object
        return cls(
            ego_feature=batched_ego_feature,
            agent_feature=batched_agent_feature,
            static_obstacle_feature=batched_static_obstacle_feature,
            road_feature=batched_road_feature,
            route_feature=batched_route_feature,
            ref_path_feature=batched_ref_path,
            ref_path_feature_mask=ref_path_mask,
        )

    def to_feature_tensor(self, dtype: torch.dtype = torch.float32) -> SceneFeature:
        """
        :return object which will be collated into a batch
        """
        if self.ref_path_feature is None:
            ref_path_feature_tensor = torch.zeros((0, 6), dtype=dtype)
            ref_path_feature_mask_tensor = torch.zeros((0,), dtype=torch.bool)
        else:
            ref_path_feature_tensor = to_tensor(self.ref_path_feature, dtype=dtype)
            if self.ref_path_feature_mask is None:
                ref_path_feature_mask_tensor = torch.ones(
                    ref_path_feature_tensor.shape[0], dtype=torch.bool, device=ref_path_feature_tensor.device
                )
            elif isinstance(self.ref_path_feature_mask, np.ndarray) and self.ref_path_feature_mask.size == 0:
                ref_path_feature_mask_tensor = torch.ones(
                    ref_path_feature_tensor.shape[0], dtype=torch.bool, device=ref_path_feature_tensor.device
                )
            elif isinstance(self.ref_path_feature_mask, torch.Tensor) and self.ref_path_feature_mask.numel() == 0:
                ref_path_feature_mask_tensor = torch.ones(
                    ref_path_feature_tensor.shape[0], dtype=torch.bool, device=ref_path_feature_tensor.device
                )
            else:
                ref_path_feature_mask_tensor = to_tensor(self.ref_path_feature_mask, dtype=torch.bool)
        ego_feature_tensor = self.ego_feature.to_feature_tensor(dtype=dtype)
        agent_feature_tensor = self.agent_feature.to_feature_tensor(dtype=dtype)
        static_obstacle_feature_tensor = self.static_obstacle_feature.to_feature_tensor(dtype=dtype)
        road_feature_tensor = self.road_feature.to_feature_tensor(dtype=dtype)
        route_feature_tensor = self.route_feature.to_feature_tensor(dtype=dtype)

        return SceneFeature(
            ego_feature=ego_feature_tensor,
            agent_feature=agent_feature_tensor,
            static_obstacle_feature=static_obstacle_feature_tensor,
            road_feature=road_feature_tensor,
            route_feature=route_feature_tensor,
            ref_path_feature=ref_path_feature_tensor,
            ref_path_feature_mask=ref_path_feature_mask_tensor,
        )

    def to_device(self, device: torch.device) -> SceneFeature:
        """
        :param device: desired device to move feature to
        :return feature type that was moved to a device
        """
        return SceneFeature(
            ego_feature=self.ego_feature.to_device(device),
            agent_feature=self.agent_feature.to_device(device),
            static_obstacle_feature=self.static_obstacle_feature.to_device(device),
            road_feature=self.road_feature.to_device(device),
            route_feature=self.route_feature.to_device(device),
            ref_path_feature=self.ref_path_feature.to(device) if isinstance(self.ref_path_feature, torch.Tensor) else self.ref_path_feature,
            ref_path_feature_mask=self.ref_path_feature_mask.to(device) if isinstance(self.ref_path_feature_mask, torch.Tensor) else self.ref_path_feature_mask,
        )

    def serialize(self) -> Dict[str, Any]:
        """
        :return: Return dictionary of data that can be serialized
        """
        data = { 
            'road_feature': self.road_feature.serialize(),
            'ref_path_feature': self.ref_path_feature,
            'ref_path_feature_mask': self.ref_path_feature_mask,
            'ego_feature': self.ego_feature.serialize(),
            'agent_feature': self.agent_feature.serialize(),
            'static_obstacle_feature': self.static_obstacle_feature.serialize(),
            'route_feature': self.route_feature.serialize(),
        }
        return data

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> SceneFeature:
        """
        :return: Return dictionary of data that can be serialized
        """
        agent_feature=AgentFeature.deserialize(data['agent_feature'])
        static_obstacle_feature=StaticObstacleFeature.deserialize(data['static_obstacle_feature'])
        road_feature=RoadFeature.deserialize(data['road_feature'])
        route_feature=RouteFeature.deserialize(data['route_feature'])
        ego_feature=EgoFeature.deserialize(data['ego_feature'])
        return cls(
            ego_feature=ego_feature,
            agent_feature=agent_feature,
            static_obstacle_feature=static_obstacle_feature,
            road_feature=road_feature,
            route_feature=route_feature,
            ref_path_feature=data.get('ref_path_feature'),
            ref_path_feature_mask=data.get('ref_path_feature_mask', np.array([], dtype=bool)),
        )

    def unpack(self) -> List[AbstractModelFeature]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        features = []
        batch_size = self.ego_feature.ego_current_state.shape[0]

        agent_feature_list = self.agent_feature.unpack()
        static_obstacle_feature_list = self.static_obstacle_feature.unpack()
        road_feature_list = self.road_feature.unpack()
        route_feature_list = self.route_feature.unpack()
        ego_feature_list = self.ego_feature.unpack()
        for i in range(batch_size):
            if self.ref_path_feature is not None:
                mask = self.ref_path_feature_mask[i] if self.ref_path_feature_mask is not None else None
                if mask is not None and isinstance(mask, torch.Tensor) and mask.numel() > 0:
                    num_waypoints = int(mask.sum().item())
                    ref_path_feature = self.ref_path_feature[i][:num_waypoints]
                    ref_path_feature_mask = mask[:num_waypoints]
                else:
                    ref_path_feature = self.ref_path_feature[i]
                    ref_path_feature_mask = None
            else:
                ref_path_feature = None
                ref_path_feature_mask = None

            ego_feature = ego_feature_list[i]
            agent_feature = agent_feature_list[i]
            static_obstacle_feature = static_obstacle_feature_list[i]
            road_feature = road_feature_list[i]
            route_feature = route_feature_list[i]

            features.append(
                SceneFeature(
                    ego_feature=ego_feature,
                    agent_feature=agent_feature,
                    static_obstacle_feature=static_obstacle_feature,
                    road_feature=road_feature,
                    route_feature=route_feature,
                    ref_path_feature=ref_path_feature,
                    ref_path_feature_mask=ref_path_feature_mask,
                )
            )
        return features

@dataclass
class EgoFeature(AbstractModelFeature):
    """
    Class that holds the ego features.
    """

    ego_current_state: FeatureDataType    # [(B), state_dim(x, y, yaw, v, a, r)]
    ego_history_state: FeatureDataType    # [(B), history_length, state_dim]
    ego_geometry: FeatureDataType         # [(B), geometry_dim(half_width, half_length, rear_to_center_dist)]

    @classmethod
    def get_feature_unique_name(cls) -> str:
        return "ego_feature"
    
    def to_feature_tensor(self, dtype: torch.dtype = torch.float32) -> EgoFeature:
        """
        :return object which will be collated into a batch
        """
        ego_current_state_tensor = to_tensor(self.ego_current_state, dtype=dtype)
        ego_history_state_tensor = to_tensor(self.ego_history_state, dtype=dtype)
        ego_geometry_tensor = to_tensor(self.ego_geometry, dtype=dtype)

        return EgoFeature(
            ego_current_state=ego_current_state_tensor,
            ego_history_state=ego_history_state_tensor,
            ego_geometry=ego_geometry_tensor
        )
    
    def to_device(self, device: torch.device) -> EgoFeature:
        """
        :param device: desired device to move feature to
        :return feature type that was moved to a device
        """
        return EgoFeature(
            ego_current_state=self.ego_current_state.to(device),
            ego_history_state=self.ego_history_state.to(device),
            ego_geometry=self.ego_geometry.to(device)
        )
    
    def serialize(self) -> Dict[str, Any]:
        """
        :return: Return dictionary of data that can be serialized
        """
        data = {
            'ego_current_state': self.ego_current_state,
            'ego_history_state': self.ego_history_state,
            'ego_geometry': self.ego_geometry
        }
        return data
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> EgoFeature:
        """
        :param data: Dictionary containing serialized data
        :return: Deserialized EgoFeature object
        """
        return cls(
            ego_current_state=data['ego_current_state'],
            ego_history_state=data['ego_history_state'],
            ego_geometry=data['ego_geometry']
        )
    
    @classmethod
    def collate(cls, batch: List[EgoFeature]) -> EgoFeature:
        """
        Batch features together with a default_collate function
        :param batch: features to be batched
        :return: batched features together
        """
        ego_current_states = [f.ego_current_state for f in batch]
        ego_history_states = [f.ego_history_state for f in batch]
        ego_geometries = [f.ego_geometry for f in batch]

        return cls(
            ego_current_state=torch.stack(ego_current_states),
            ego_history_state=torch.stack(ego_history_states),
            ego_geometry=torch.stack(ego_geometries)
        )
    
    def unpack(self) -> List[EgoFeature]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        features = []
        batch_size = self.ego_current_state.shape[0]
        for i in range(batch_size):
            features.append(
                EgoFeature(
                    ego_current_state=self.ego_current_state[i],
                    ego_history_state=self.ego_history_state[i],
                    ego_geometry=self.ego_geometry[i]
                )
            )
        return features

@dataclass
class AgentFeature(AbstractModelFeature):
    """
    Class that holds the agent features.
    """

    agent_current_state: FeatureDataType    # [(B), n_agents, state_dim]
    agent_history_state: FeatureDataType    # [(B), n_agents, history_length, state_dim]
    agent_history_mask: FeatureDataType     # [(B), n_agents, history_length]
    agent_type: FeatureDataType             # [(B), n_agents, ]
    agent_geometry: FeatureDataType         # [(B), n_agents, geometry_dim]
    agent_mask : Optional[FeatureDataType] = field(default_factory=lambda: np.array([]))

    @classmethod
    def collate(cls, batch: List[AgentFeature]) -> AgentFeature:
        """
        Batch features together with a default_collate function
        :param batch: features to be batched
        :return: batched features together
        """
        device = batch[0].agent_current_state.device if isinstance(batch[0].agent_current_state, torch.Tensor) else torch.device('cpu')
        
        # Find the maximum number of agents in the batch
        max_n_agents = max(f.agent_current_state.shape[0] for f in batch)
        
        padded = {
            'agent_current_state': [],
            'agent_history_state': [],
            'agent_geometry': [],
            'agent_history_mask': [],
            'agent_type': [],
            'agent_mask': torch.zeros(len(batch), max_n_agents, dtype=torch.bool, device=device)
        }

        for b_idx, sample in enumerate(batch):
            num_agents = sample.agent_current_state.shape[0]
            # Pad the agent features to the maximum number of agents
            if num_agents == 0:
                # If there are no agents, create zero tensors with the correct shape
                padded['agent_current_state'].append(torch.zeros((max_n_agents, sample.agent_current_state.shape[-1]), dtype=sample.agent_current_state.dtype, device=device))
                padded['agent_history_state'].append(torch.zeros((max_n_agents, sample.agent_history_state.shape[1], sample.agent_history_state.shape[2]), dtype=sample.agent_history_state.dtype, device=device))
                padded['agent_geometry'].append(torch.zeros((max_n_agents, sample.agent_geometry.shape[-1]), dtype=sample.agent_geometry.dtype, device=device))
                padded['agent_history_mask'].append(torch.zeros((max_n_agents, sample.agent_history_mask.shape[-1]), dtype=sample.agent_history_mask.dtype, device=device))
                padded['agent_type'].append(torch.zeros((max_n_agents,), dtype=sample.agent_type.dtype, device=device))
                # agent_mask remains all False (already initialized)
                continue
            padded['agent_current_state'].append(F.pad(sample.agent_current_state, (0, 0, 0, max_n_agents - num_agents)))
            padded['agent_history_state'].append(F.pad(sample.agent_history_state, (0,0,0,0,0,max_n_agents - num_agents)))
            padded['agent_geometry'].append(F.pad(sample.agent_geometry, (0, 0, 0, max_n_agents - num_agents)))
            padded['agent_history_mask'].append(F.pad(sample.agent_history_mask, (0, 0, 0, max_n_agents - num_agents)))
            padded['agent_type'].append(F.pad(sample.agent_type, (0, max_n_agents - num_agents)))
            padded['agent_mask'][b_idx, :num_agents] = True

        return cls(
            agent_current_state=torch.stack(padded['agent_current_state']),
            agent_history_state=torch.stack(padded['agent_history_state']),
            agent_geometry=torch.stack(padded['agent_geometry']),
            agent_history_mask=torch.stack(padded['agent_history_mask']),
            agent_type=torch.stack(padded['agent_type']),
            agent_mask=padded['agent_mask'],
        )

    def to_feature_tensor(self, dtype: torch.dtype = torch.float32) -> AgentFeature:
        """
        :return object which will be collated into a batch
        """
        agent_current_state_tensor = to_tensor(self.agent_current_state, dtype=dtype)
        agent_history_state_tensor = to_tensor(self.agent_history_state, dtype=dtype)
        agent_history_mask_tensor = to_tensor(self.agent_history_mask, dtype=torch.bool)
        agent_type_tensor = to_tensor(self.agent_type, dtype=torch.int)
        agent_geometry_tensor = to_tensor(self.agent_geometry, dtype=dtype)
        agent_mask_tensor = (
            torch.ones(agent_current_state_tensor.shape[0], dtype=torch.bool) 
            if self.agent_mask.size == 0 
            else to_tensor(self.agent_mask, dtype=torch.bool)
        )

        return AgentFeature(
            agent_current_state=agent_current_state_tensor,
            agent_history_state=agent_history_state_tensor,
            agent_history_mask=agent_history_mask_tensor,
            agent_type=agent_type_tensor,
            agent_geometry=agent_geometry_tensor,
            agent_mask=agent_mask_tensor
        )

    def to_device(self, device: torch.device) -> AgentFeature:
        """
        :param device: desired device to move feature to
        :return feature type that was moved to a device
        """
        return AgentFeature(
            agent_current_state=self.agent_current_state.to(device),
            agent_history_state=self.agent_history_state.to(device),
            agent_history_mask=self.agent_history_mask.to(device),
            agent_type=self.agent_type.to(device),
            agent_geometry=self.agent_geometry.to(device),
            agent_mask=self.agent_mask.to(device)
        )

    def serialize(self) -> Dict[str, Any]:
        """
        :return: Return dictionary of data that can be serialized
        """
        data = {
            'agent_current_state': self.agent_current_state,
            'agent_history_state': self.agent_history_state,
            'agent_history_mask': self.agent_history_mask,
            'agent_type': self.agent_type,
            'agent_geometry': self.agent_geometry,
            'agent_mask': self.agent_mask
        }
        return data

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AgentFeature:
        """
        :return: Return dictionary of data that can be serialized
        """
        return cls(
            agent_current_state=data['agent_current_state'],
            agent_history_state=data['agent_history_state'],
            agent_history_mask=data['agent_history_mask'],
            agent_type=data['agent_type'],
            agent_geometry=data['agent_geometry'],
            # agent_mask=data['agent_mask'],
        )

    def unpack(self) -> List[AgentFeature]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        # Unpack batched AgentFeature into a list of single AgentFeature, removing padding according to agent_mask
        features = []
        batch_size = self.agent_current_state.shape[0]
        for i in range(batch_size):
            if self.agent_mask is not None and self.agent_mask.numel() > 0:
                mask = self.agent_mask[i]
                num_agents = mask.sum().item()
                agent_current_state = self.agent_current_state[i][:num_agents]
                agent_history_state = self.agent_history_state[i][:num_agents]
                agent_history_mask = self.agent_history_mask[i][:num_agents]
                agent_type = self.agent_type[i][:num_agents]
                agent_geometry = self.agent_geometry[i][:num_agents]
                agent_mask = mask[:num_agents]
            else:
                # No mask, use all
                agent_current_state = self.agent_current_state[i]
                agent_history_state = self.agent_history_state[i]
                agent_history_mask = self.agent_history_mask[i]
                agent_type = self.agent_type[i]
                agent_geometry = self.agent_geometry[i]
                agent_mask = torch.ones(agent_current_state.shape[0], dtype=torch.bool)
            features.append(
                AgentFeature(
                    agent_current_state=agent_current_state,
                    agent_history_state=agent_history_state,
                    agent_history_mask=agent_history_mask,
                    agent_type=agent_type,
                    agent_geometry=agent_geometry,
                    agent_mask=agent_mask
                )
            )
        return features


@dataclass
class RoadFeature(AbstractModelFeature):
    """
    Class that holds the road features.
    """

    center_line: FeatureDataType              # [(B), n_roads, n_points, 3]
    road_geometry: FeatureDataType            # [(B), n_roads, n_edge, 2]
    road_type: FeatureDataType                # [(B), n_roads]
    road_speed_limit: FeatureDataType         # [(B), n_roads]
    road_traffic_light: FeatureDataType       # [(B), n_roads]
    road_mask: Optional[FeatureDataType] = field(default_factory=lambda: np.array([]))

    @classmethod
    def collate(cls, batch: List[RoadFeature]) -> RoadFeature:
        """
        Batch features together with a default_collate function
        :param batch: features to be batched
        :return: batched features together
        """
        device = batch[0].center_line.device if isinstance(batch[0].center_line, torch.Tensor) else torch.device('cpu')
        
        
        max_n_roads = max(f.road_geometry.shape[0] for f in batch)
        padded = {
            'road_geometry': [],
            'road_mask': torch.zeros(len(batch), max_n_roads, dtype=torch.bool),
            'center_line': [],
            'road_type': [],
            'road_speed_limit': [],
            'road_traffic_light': []
        }
        for b_idx, sample in enumerate(batch):
            try:
                logger.debug(f"[DEBUG] batch_idx={b_idx}, road_geometry.shape={sample.road_geometry.shape}")
                logger.debug(f"[DEBUG] batch_idx={b_idx}, center_line.shape={sample.center_line.shape}")
                num_roads = sample.road_geometry.shape[0]
                logger.debug(f"[DEBUG] num_roads={num_roads}, max_n_roads={max_n_roads}")
                # todo: Handle the case where there are no roads in the sample, avoid hard coding
                if num_roads == 0:
                    padded['road_geometry'].append(torch.zeros((max_n_roads, 20, 2), dtype=sample.road_geometry.dtype, device=device))
                    padded['center_line'].append(torch.zeros((max_n_roads, 20, 3), dtype=sample.center_line.dtype, device=device))
                    padded['road_type'].append(torch.zeros((max_n_roads,), dtype=sample.road_type.dtype, device=device))
                    padded['road_speed_limit'].append(torch.full((max_n_roads,), -1, dtype=sample.road_speed_limit.dtype, device=device))
                    padded['road_traffic_light'].append(torch.zeros((max_n_roads,), dtype=sample.road_traffic_light.dtype, device=device))
                    continue
                padded['road_geometry'].append(F.pad(sample.road_geometry, (0, 0, 0, 0, 0, max_n_roads - num_roads)))
                padded['road_mask'][b_idx, :num_roads] = True
                padded['center_line'].append(F.pad(sample.center_line, (0, 0, 0, 0, 0, max_n_roads - num_roads)))
                padded['road_type'].append(F.pad(sample.road_type, (0, max_n_roads - num_roads)))  
                padded['road_speed_limit'].append(F.pad(sample.road_speed_limit, (0, max_n_roads - num_roads), value=-1))  # -1 for unknown speed limit
                padded['road_traffic_light'].append(F.pad(sample.road_traffic_light, (0, max_n_roads - num_roads), value=0)) # UNKNOWN traffic light state

            except Exception as e:
                logger.error(f"Error processing batch index {b_idx}: {e}")
                logger.debug(f"Sample road_geometry shape: {sample.road_geometry.shape}")
                with open(f"debug_batch_{b_idx}.pkl", "wb") as f:
                    pickle.dump(sample, f)
                raise

        return cls(
            road_geometry=torch.stack(padded['road_geometry']),
            road_mask=padded['road_mask'],
            center_line=torch.stack(padded['center_line']),
            road_type=torch.stack(padded['road_type']),
            road_speed_limit=torch.stack(padded['road_speed_limit']),
            road_traffic_light=torch.stack(padded['road_traffic_light'])
        )

    def to_feature_tensor(self, dtype: torch.dtype = torch.float32) -> RoadFeature:
        """
        :return object which will be collated into a batch
        """
        road_geometry_tensor = to_tensor(self.road_geometry, dtype=dtype)
        road_mask_tensor = (
            torch.ones(road_geometry_tensor.shape[0], dtype=torch.bool) 
            if self.road_mask.size == 0 
            else to_tensor(self.road_mask, dtype=torch.bool)
        )
        center_line_tensor = to_tensor(self.center_line, dtype=dtype)
        road_type_tensor = to_tensor(self.road_type, dtype=torch.int)
        road_speed_limit_tensor = to_tensor(self.road_speed_limit, dtype=dtype)
        road_traffic_light_tensor = to_tensor(self.road_traffic_light, dtype=torch.int)

        return RoadFeature(
            road_geometry=road_geometry_tensor,
            road_mask=road_mask_tensor,
            center_line=center_line_tensor,
            road_type=road_type_tensor,
            road_speed_limit=road_speed_limit_tensor,
            road_traffic_light=road_traffic_light_tensor
        )

    def to_device(self, device: torch.device) -> RoadFeature:
        """
        :param device: desired device to move feature to
        :return feature type that was moved to a device
        """
        return RoadFeature(
            road_geometry=self.road_geometry.to(device),
            road_mask=self.road_mask.to(device),
            center_line=self.center_line.to(device),
            road_type=self.road_type.to(device),
            road_speed_limit=self.road_speed_limit.to(device),
            road_traffic_light=self.road_traffic_light.to(device)
        )

    def serialize(self) -> Dict[str, Any]:
        """
        :return: Return dictionary of data that can be serialized
        """
        data = {
            'road_geometry': self.road_geometry,
            # 'road_mask': self.road_mask,
            'center_line': self.center_line,
            'road_type': self.road_type,
            'road_speed_limit': self.road_speed_limit,
            'road_traffic_light': self.road_traffic_light
        }
        return data

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> RoadFeature:
        """
        :return: Return dictionary of data that can be serialized
        """
        return cls(
            road_geometry=data['road_geometry'],
            # road_mask=data['road_mask'],
            center_line=data['center_line'],
            road_type=data['road_type'],
            road_speed_limit=data['road_speed_limit'],
            road_traffic_light=data['road_traffic_light']
        )
    
    def unpack(self) -> List[AbstractModelFeature]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        features = []
        batch_size = self.road_geometry.shape[0]
        for i in range(batch_size):
            if self.road_mask is not None and self.road_mask.numel() > 0:
                mask = self.road_mask[i]
                num_roads = mask.sum().item()
                road_geometry = self.road_geometry[i][:num_roads]
                center_line = self.center_line[i][:num_roads]
                road_type = self.road_type[i][:num_roads]
                road_speed_limit = self.road_speed_limit[i][:num_roads]
                road_traffic_light = self.road_traffic_light[i][:num_roads]
                road_mask = mask[:num_roads]
            else:
                road_geometry = self.road_geometry[i]
                center_line = self.center_line[i]
                road_type = self.road_type[i]
                road_speed_limit = self.road_speed_limit[i]
                road_traffic_light = self.road_traffic_light[i]
                road_mask = torch.ones(road_geometry.shape[0], dtype=torch.bool)
            features.append(
                RoadFeature(
                    road_geometry=road_geometry,
                    road_mask=road_mask,
                    center_line=center_line,
                    road_type=road_type,
                    road_speed_limit=road_speed_limit,
                    road_traffic_light=road_traffic_light
                )
            )
        return features

@dataclass
class RouteFeature(AbstractModelFeature):
    """
    Class that holds the route features.
    """

    route_geometry: FeatureDataType    # [(B), n_route, n_edge, 3] (x, y, yaw)
    route_mask: Optional[FeatureDataType] = field(default_factory=lambda: np.array([]))

    @classmethod
    def collate(cls, batch: List[RouteFeature]) -> RouteFeature:
        """
        Batch features together with a default_collate function
        :param batch: features to be batched
        :return: batched features together
        """
        device = batch[0].route_geometry.device if isinstance(batch[0].route_geometry, torch.Tensor) else torch.device('cpu')

        max_n_route_points = max(f.route_geometry.shape[0] for f in batch)
        padded = {
            'route_geometry': [],
            'route_mask': torch.zeros(len(batch), max_n_route_points, dtype=torch.bool, device=device)
        }

        for b_idx, sample in enumerate(batch):
            num_route_points = sample.route_geometry.shape[0]
            if num_route_points == 0:
                # If there are no route points, create zero tensors with the correct shape
                padded['route_geometry'].append(torch.zeros((max_n_route_points, 20, 2), dtype=sample.route_geometry.dtype, device=device))
                # route_mask remains all False (already initialized)
                continue
            # Pad the route features to the maximum number of route points
            padded['route_geometry'].append(F.pad(sample.route_geometry, (0, 0, 0, 0, 0, max_n_route_points - num_route_points)))
            padded['route_mask'][b_idx, :num_route_points] = True

        return cls(
            route_geometry=torch.stack(padded['route_geometry']),
            route_mask=padded['route_mask']
        )
    
    def to_feature_tensor(self, dtype: torch.dtype = torch.float32) -> RouteFeature:
        """
        :return object which will be collated into a batch
        """
        route_geometry_tensor = to_tensor(self.route_geometry, dtype=dtype)
        route_mask_tensor = (
            torch.ones(route_geometry_tensor.shape[0], dtype=torch.bool) 
            if self.route_mask.size == 0 
            else to_tensor(self.route_mask, dtype=torch.bool)
        )

        return RouteFeature(
            route_geometry=route_geometry_tensor,
            route_mask=route_mask_tensor
        )
    
    def to_device(self, device: torch.device) -> RouteFeature:
        """
        :param device: desired device to move feature to
        :return feature type that was moved to a device
        """
        return RouteFeature(
            route_geometry=self.route_geometry.to(device),
            route_mask=self.route_mask.to(device)
        )
    
    def serialize(self) -> Dict[str, Any]:
        """
        :return: Return dictionary of data that can be serialized
        """
        data = {
            'route_geometry': self.route_geometry,
            'route_mask': self.route_mask
        }
        return data
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> RouteFeature:
        """
        :return: Return dictionary of data that can be serialized
        """
        return cls(
            route_geometry=data['route_geometry'],
            # route_mask=data['route_mask']
        )
    
    def unpack(self) -> List[AbstractModelFeature]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        features = []
        batch_size = self.route_geometry.shape[0]
        for i in range(batch_size):
            # Check if route_mask exists and is non-empty
            if self.route_mask is not None and self.route_mask.numel() > 0:
                mask = self.route_mask[i]
                num_route_points = mask.sum().item()
                route_geometry = self.route_geometry[i][:num_route_points]
                route_mask = mask[:num_route_points]
            else:
                route_geometry = self.route_geometry[i]
                route_mask = torch.ones(route_geometry.shape[0], dtype=torch.bool)
            features.append(
                RouteFeature(
                    route_geometry=route_geometry,
                    route_mask=route_mask
                )
            )
        return features

@dataclass
class StaticObstacleFeature(AbstractModelFeature):
    """
    Class that holds the static obstacle features.
    """

    static_obstacle_position: FeatureDataType     # [(B), n_static_obstacles, 3] (x, y, yaw)
    static_object_dimension: FeatureDataType      # [(B), n_static_obstacles, 2] (half_length, half_width)
    static_object_type: FeatureDataType           # [(B), n_static_obstacles]
    static_obstacle_mask: Optional[FeatureDataType] = field(default_factory=lambda: np.array([]))

    @classmethod
    def collate(cls, batch: List[StaticObstacleFeature]) -> StaticObstacleFeature:
        """
        Batch features together with a default_collate function
        :param batch: features to be batched
        :return: batched features together
        """
        device = batch[0].static_obstacle_position.device if isinstance(batch[0].static_obstacle_position, torch.Tensor) else torch.device('cpu')

        max_n_static_obstacles = max(f.static_obstacle_position.shape[0] for f in batch)
        padded = {
            'position': [],
            'dimension': [],
            'static_object_type': [],
            'static_obstacle_mask': torch.zeros(len(batch), max_n_static_obstacles, dtype=torch.bool, device=device)
        }

        for b_idx, sample in enumerate(batch):
            num_static_obstacles = sample.static_obstacle_position.shape[0]
            if num_static_obstacles == 0:
                # If there are no static obstacles, create zero tensors with the correct shape
                padded['position'].append(torch.zeros((max_n_static_obstacles, 3), dtype=sample.static_obstacle_position.dtype, device=device))
                padded['dimension'].append(torch.zeros((max_n_static_obstacles, 2), dtype=sample.static_object_dimension.dtype, device=device))
                padded['static_object_type'].append(torch.zeros((max_n_static_obstacles,), dtype=sample.static_object_type.dtype, device=device))
                # static_obstacle_mask remains all False (already initialized)
                continue
            # Pad the static obstacle features to the maximum number of static obstacles
            padded['position'].append(F.pad(sample.static_obstacle_position, (0, 0, 0, max_n_static_obstacles - num_static_obstacles)))
            padded['dimension'].append(F.pad(sample.static_object_dimension, (0, 0, 0, max_n_static_obstacles - num_static_obstacles)))
            padded['static_object_type'].append(F.pad(sample.static_object_type, (0, max_n_static_obstacles - num_static_obstacles), value=0))  # 0 for unknown type
            padded['static_obstacle_mask'][b_idx, :num_static_obstacles] = True

        return cls(
            static_obstacle_position=torch.stack(padded['position']),
            static_object_dimension=torch.stack(padded['dimension']),
            static_object_type=torch.stack(padded['static_object_type']),
            static_obstacle_mask=padded['static_obstacle_mask']
        )

    def to_feature_tensor(self, dtype: torch.dtype = torch.float32) -> StaticObstacleFeature:
        """
        :return object which will be collated into a batch
        """
        position_tensor = to_tensor(self.static_obstacle_position, dtype=dtype)
        dimension_tensor = to_tensor(self.static_object_dimension, dtype=dtype)
        type_tensor = to_tensor(self.static_object_type, dtype=torch.int)
        mask_tensor = (
            torch.ones(position_tensor.shape[0], dtype=torch.bool) 
            if self.static_obstacle_mask.size == 0 
            else to_tensor(self.static_obstacle_mask, dtype=torch.bool)
        )

        return StaticObstacleFeature(
            static_obstacle_position=position_tensor,
            static_object_dimension=dimension_tensor,
            static_object_type=type_tensor,
            static_obstacle_mask=mask_tensor
        )

    def to_device(self, device: torch.device) -> StaticObstacleFeature:
        """
        :param device: desired device to move feature to
        :return feature type that was moved to a device
        """
        return StaticObstacleFeature(
            static_obstacle_position=self.static_obstacle_position.to(device),
            static_object_dimension=self.static_object_dimension.to(device),
            static_object_type=self.static_object_type.to(device),
            static_obstacle_mask=self.static_obstacle_mask.to(device)
        )
    def serialize(self) -> Dict[str, Any]:
        """
        :return: Return dictionary of data that can be serialized
        """
        data = {
            'static_obstacle_position': self.static_obstacle_position,
            'static_object_dimension': self.static_object_dimension,
            'static_object_type': self.static_object_type,
            'static_obstacle_mask': self.static_obstacle_mask
        }
        return data
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> StaticObstacleFeature:
        """
        :return: Return dictionary of data that can be serialized
        """
        return cls(
            static_obstacle_position=data['static_obstacle_position'],
            static_object_dimension=data['static_object_dimension'],
            static_object_type=data['static_object_type'],
            # static_obstacle_mask=data['static_obstacle_mask']
        )
    def unpack(self) -> List[AbstractModelFeature]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        features = []
        batch_size = self.static_obstacle_position.shape[0]
        for i in range(batch_size):
            if self.static_obstacle_mask is not None and self.static_obstacle_mask.numel() > 0:
                mask = self.static_obstacle_mask[i]
                num_static_obstacles = mask.sum().item()
                static_obstacle_position = self.static_obstacle_position[i][:num_static_obstacles]
                static_object_dimension = self.static_object_dimension[i][:num_static_obstacles]
                static_object_type = self.static_object_type[i][:num_static_obstacles]
                static_obstacle_mask = mask[:num_static_obstacles]
            else:
                static_obstacle_position = self.static_obstacle_position[i]
                static_object_dimension = self.static_object_dimension[i]
                static_object_type = self.static_object_type[i]
                static_obstacle_mask = torch.ones(static_obstacle_position.shape[0], dtype=torch.bool)
            features.append(
                StaticObstacleFeature(
                    static_obstacle_position=static_obstacle_position,
                    static_object_dimension=static_object_dimension,
                    static_object_type=static_object_type,
                    static_obstacle_mask=static_obstacle_mask
                )
            )
        return features
    
