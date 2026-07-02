from typing import List, Dict, Optional, Tuple
import numpy as np
import torch

from scipy.interpolate import interp1d

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType

from shapely.geometry import Polygon

from lpl_planner.planning.scene.map.lane_map import LaneMap
from lpl_planner.planning.scene.map.map_utils.roi_segement import ROIMap
from ..map.occupancy_map import OccupancyMap
from .utils import process_agent_states
import logging
logger = logging.getLogger(__name__)


class AgentManager:
    def __init__(self, 
                 predicton_step: int = 60, 
                 history_step: int = 30,
                 time_step = 0.1, 
                 use_local_coords: bool = False,
                 longitudinal_safe_margin: float = 0.5,
                 lateral_safe_margin: float = 0.2,
                 map_radius: float = 100,
                 tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}):

        
        self.occupancy_maps: List[OccupancyMap] = []
        self.time_step = time_step
        self.prediction_step = predicton_step
        self.longitudinal_safe_margin = longitudinal_safe_margin
        self.lateral_safe_margin = lateral_safe_margin
        self.use_local_coords = use_local_coords
        self._tenser_args = tensor_args
        self._map_radius = map_radius
        self.unique_objects: Dict[str, TrackedObject] = {}
        # self.genenrate_agent_point_set()

        # for step in np.arange(predicton_step+1):
        #     t = step * time_step
        #     occupancy_map = self.get_agent_map(t)
        #     self.occupancy_maps.append(occupancy_map)

    def __getitem__(self, step_idx: int):
        
        return self.occupancy_maps[step_idx]

    def init_with_nuplan_scene(self, 
                               scenario: AbstractScenario,
                               prediction_sampling: TrajectorySampling,
                               lane_map: LaneMap = None,
                               map_radius = 100,
                               use_agent_tensor = False,
                               ):
        
        self.scenario = scenario
        self.prediction_sampling = prediction_sampling
        self._map_radius = map_radius
        self.lane_map = lane_map
        if use_agent_tensor:
            self.agent_dict: Dict[str, TrackedObject] = {}
            self.static_object_dict: Dict[str, TrackedObject]= {}
            dt = self.scenario.get_tracked_objects_at_iteration(iteration=0,
                                                                future_trajectory_sampling=self.prediction_sampling
                                                                )
            tracked_objects = dt.tracked_objects.tracked_objects
            self._genenrate_agent_point_set_with_scene(tracked_objects=tracked_objects,
                                                    ego_state=scenario.initial_ego_state,
                                                    lane_map=lane_map)

            occupancy_maps = []
            occupancy_map = self.get_agent_map_from_scenario(ego_state=scenario.initial_ego_state)
            occupancy_maps.append(occupancy_map)
        else:
            occupancy_maps = self.get_occupancy_map(ego_state=scenario.initial_ego_state,
                                                     scenario=scenario,
                                                     roi_map=None,
                                                     lane_map=lane_map,)

        self.occupancy_maps = occupancy_maps

    def init_with_planner_input(self, 
                               prediction_sampling: TrajectorySampling,
                               lane_map: LaneMap = None,
                               map_radius = 100):
        
        self.prediction_sampling = prediction_sampling
        self._map_radius = map_radius
        self.lane_map = lane_map

        self.agent_dict: Dict[str, TrackedObject] = {}
        self.static_object_dict: Dict[str, TrackedObject]= {}


    def step_with_iteration(self,iteration,ego_state=None,lane_map: LaneMap = None):
        self.agent_dict: Dict[str, TrackedObject] = {}
        current_ego_state = ego_state if ego_state else self.scenario.get_ego_state_at_iteration(iteration=iteration)
        dt = self.scenario.get_tracked_objects_at_iteration(iteration=iteration,
                                                            future_trajectory_sampling=self.prediction_sampling
                                                            )
        tracked_objects = dt.tracked_objects.tracked_objects
        self._genenrate_agent_point_set_with_scene(tracked_objects=tracked_objects,
                                                  ego_state=current_ego_state,
                                                  lane_map=lane_map)
        occupancy_maps = []
        occupancy_map = self.get_agent_map_from_scenario(ego_state=current_ego_state)
        occupancy_maps.append(occupancy_map)
    
        self.occupancy_maps = occupancy_maps


    
    def get_agent_map_from_scenario(self,ego_state: EgoState = None,t=0):
        agent_polygons = []
        agent_tokens = []
        for agent in self.agent_dict.values():
            agent_polygon = agent.box.geometry
            if self.use_local_coords:
                agent_polygon_coords = np.array(agent_polygon.exterior.coords)
                ego_yaw = - ego_state.rear_axle.heading
                agent_polygon_coords = agent_polygon_coords - ego_state.rear_axle.array
                agent_polygon_coords = np.matmul(agent_polygon_coords, np.array([[np.cos(ego_yaw),np.sin(ego_yaw)],
                                                        [-np.sin(ego_yaw),np.cos(ego_yaw)]]))
                agent_polygon = Polygon(agent_polygon_coords)
            agent_polygons.append(agent_polygon)
            agent_tokens.append(agent.token)

        for static_obj in self.static_object_dict.values():
            static_obj_polygon = static_obj.box.geometry
            if self.use_local_coords:
                static_obj_polygon_coords = np.array(static_obj_polygon.exterior.coords)
                ego_yaw = - ego_state.rear_axle.heading
                static_obj_polygon_coords = static_obj_polygon_coords - ego_state.rear_axle.array
                static_obj_polygon_coords = np.matmul(static_obj_polygon_coords, np.array([[np.cos(ego_yaw),np.sin(ego_yaw)],
                                                        [-np.sin(ego_yaw),np.cos(ego_yaw)]]))
                static_obj_polygon = Polygon(static_obj_polygon_coords)
            agent_polygons.append(static_obj_polygon)
            agent_tokens.append(static_obj.token)

        return OccupancyMap(agent_tokens,agent_polygons)

    def get_occupancy_map(self, 
                      ego_state: EgoState = None,
                      scenario: AbstractScenario = None,
                      roi_map: Optional[ROIMap] = None,
                      lane_map: LaneMap = None,
                      t: float = 0) -> List[OccupancyMap]:
        
        future_samples = int(self.prediction_step / self.time_step)

        ego_yaw = - ego_state.rear_axle.heading
        rotation_matrix = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)],
                                    [-np.sin(ego_yaw), np.cos(ego_yaw)]])
        ego_position = ego_state.rear_axle.point.array

        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects

        future_dts_gen = scenario.get_future_tracked_objects(
                iteration=0,
                time_horizon=self.prediction_step,
                num_samples=future_samples,
        )
        future_dts = [dt for dt in future_dts_gen]

        # Pad future_dts: if dt is None, use previous non-None dt
        padded_future_dts = []
        last_valid_dt = None
        for dt in future_dts:
            if dt is not None:
                padded_future_dts.append(dt)
                last_valid_dt = dt
            else:
                # If dt is None, pad with last valid dt
                if last_valid_dt is not None:
                    padded_future_dts.append(last_valid_dt)
                else:
                    # If no valid dt yet, pad with the first non-None dt in future_dts
                    first_valid_dt = next((x for x in future_dts if x is not None), None)
                    padded_future_dts.append(first_valid_dt)
        
        future_tracked_objects = [dt.tracked_objects for dt in padded_future_dts]

        future_tl_gen = scenario.get_future_traffic_light_status_history(
                iteration=0,
                time_horizon=self.prediction_step,
                num_samples=future_samples,
        )
        future_tl_dts = [tl_data for tl_data in future_tl_gen if tl_data is not None]
        future_tl_statuses = [tl_data.traffic_lights for tl_data in future_tl_dts]

        obj_dict : Dict[str: int] = {}
        unique_objects: Dict[str, TrackedObject] = {}
        for idx, obj in enumerate(present_tracked_objects.tracked_objects):
            tracked_object_type = obj.tracked_object_type
            tracked_object_token = obj.track_token
            tracked_object_distance = np.linalg.norm(
                ego_state.rear_axle.array - obj.center.array
            )

            if tracked_object_distance > self._map_radius:
                continue

            # filter out agents not in the route    
            if roi_map is not None:

                polygon_obj = obj.box.geometry
                polygon_coords = np.array(polygon_obj.exterior.coords)
                local_polygon_coords = (polygon_coords - ego_position) @ rotation_matrix
                polygon_obj = Polygon(local_polygon_coords)

                if not roi_map.polygon_in_roi(polygon_obj):
                    continue

            obj_dict[tracked_object_token] = idx
            unique_objects[tracked_object_token] = obj
        
        self.unique_objects = unique_objects

        # Sort agent tokens to ensure stable order in agent_dict
        obj_dict = {token: i for i, token in enumerate(sorted(obj_dict.keys()))}
        num_obj = len(obj_dict)
        occupancy_maps : List[OccupancyMap] = []

        for t_idx, tracked_objects in enumerate(future_tracked_objects):
            
            tokens = []
            polygons = []

            for obj in tracked_objects.tracked_objects:
                if obj.track_token not in obj_dict:
                    continue
                obj_idx = obj_dict[obj.track_token]
                if t_idx >= future_samples:
                    break

                obj_polygon = obj.box.geometry
                if self.use_local_coords:
                    obj_polygon_coords = np.array(obj_polygon.exterior.coords)
                    obj_polygon_coords = obj_polygon_coords - ego_position
                    obj_polygon_coords = np.matmul(obj_polygon_coords, rotation_matrix)
                    obj_polygon = Polygon(obj_polygon_coords)
                tokens.append(obj.track_token)
                polygons.append(obj_polygon)

            for tl_status in future_tl_statuses[t_idx]:
                lane_connector_id = str(tl_status.lane_connector_id)
                if (tl_status.status == TrafficLightStatusType.RED) and (
                    lane_connector_id in lane_map._route_lane_dict.keys()
                ):
                    lane_connector = lane_map._route_lane_dict[lane_connector_id]
                    stop_lines = []

                    if bool(stop_lines): 
                        for stop_line in stop_lines:
                            tokens.append(f"red_light_{stop_line.id}")
                            stop_line_polygon_coords = np.array(stop_line.polygon.exterior.coords)
                            if self.use_local_coords:
                                stop_line_polygon_coords = stop_line_polygon_coords - ego_state.rear_axle.array
                                stop_line_polygon_coords = np.matmul(stop_line_polygon_coords, rotation_matrix)
                            stop_line_polygon = Polygon(stop_line_polygon_coords)
                            polygons.append(stop_line_polygon)
                    
                    tokens.append(f"red_light_{lane_connector_id}")
                    lane_connector_polygon_coords = np.array(lane_connector.polygon.exterior.coords)
                    if self.use_local_coords:
                        lane_connector_polygon_coords = lane_connector_polygon_coords - ego_state.rear_axle.array
                        lane_connector_polygon_coords = np.matmul(lane_connector_polygon_coords, rotation_matrix)
                    lane_connector_polygon = Polygon(lane_connector_polygon_coords)
                    polygons.append(lane_connector_polygon)

            occupancy_map = OccupancyMap(
                tokens,
                np.array(polygons),
            )
            occupancy_maps.append(occupancy_map)

        return occupancy_maps
    


    def get_static_obstacles_feature(self,
                                    present_tracked_objects: TrackedObjects, 
                                    ego_state: EgoState,
                                    roi_map: ROIMap = None,
                                    num_static_objects: int = 20,
                                    ):

        ego_yaw = - ego_state.rear_axle.heading
        rotation_matrix = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)],
                                    [-np.sin(ego_yaw), np.cos(ego_yaw)]])
        ego_position = ego_state.rear_axle.point.array
        
    
        static_object_dict : Dict[str, TrackedObject] = {}

        static_object_distance_dict : Dict[str, float] = {}
        for static_object in present_tracked_objects.get_static_objects():
            tracked_object_token = static_object.track_token
            tracked_object_distance = np.linalg.norm(
                ego_state.rear_axle.array - static_object.center.array
            )
            
            if tracked_object_distance > self._map_radius:
                continue

            if roi_map is not None:
                polygon_obj = static_object.box.geometry
                polygon_coords = np.array(polygon_obj.exterior.coords)
                local_polyon_coords = (polygon_coords - ego_position) @ rotation_matrix
                polygon_obj = Polygon(local_polyon_coords)
                if not roi_map.polygon_in_roi(polygon_obj):
                    continue

            static_object_distance_dict[tracked_object_token] = tracked_object_distance
            static_object_dict[tracked_object_token] = static_object
        
        #limit the number of static objects
        if len(static_object_distance_dict) > num_static_objects:
            limited_static_object_dict = {}
            sorted_tokens = sorted(static_object_distance_dict.keys(), key=lambda x: static_object_distance_dict[x])
            for token in sorted_tokens[:num_static_objects]:
                limited_static_object_dict[token] = static_object_dict[token]
            static_object_dict = limited_static_object_dict 
        
        # extract static object features
        static_obj_num = len(static_object_dict)
        static_object_position = np.zeros((static_obj_num, 3))
        static_object_dimension = np.zeros((static_obj_num, 2))
        static_object_type = np.zeros((static_obj_num,))
        for i, static_object in enumerate(static_object_dict.values()):
            # convert static object position to local coordinate
            static_object_local_position = static_object.center.array - ego_state.rear_axle.array
            static_object_local_position = np.matmul(static_object_local_position, rotation_matrix)
            static_object_local_heading = static_object.center.heading + ego_yaw
            static_object_local_heading = (static_object_local_heading + np.pi) % (2 * np.pi) - np.pi
            static_object_position[i, :2] = static_object_local_position
            static_object_position[i, 2] = static_object_local_heading
            static_object_dimension[i] = np.array([static_object.box.half_length, static_object.box.half_width])
            static_object_type[i] = static_object.tracked_object_type.value - 2 

        static_object_position = np.array(static_object_position, dtype=np.float32)
        static_object_dimension = np.array(static_object_dimension, dtype=np.float32)
        static_object_type = np.array(static_object_type, dtype=np.int32)

        static_object_feature = {
            "static_obstacle_position": static_object_position,
            "static_object_dimension": static_object_dimension,
            "static_object_type": static_object_type,
        }

        # Check static_object_feature for invalid values and report their locations.
        for key, value in static_object_feature.items():
            if isinstance(value, np.ndarray):
                if np.isnan(value).any() or np.isinf(value).any():
                    nan_indices = np.argwhere(np.isnan(value))
                    inf_indices = np.argwhere(np.isinf(value))
                    if nan_indices.size > 0:
                        logging.warning(f"NaN detected in '{key}' at positions: {nan_indices.tolist()}")
                    if inf_indices.size > 0:
                        logging.warning(f"Inf detected in '{key}' at positions: {inf_indices.tolist()}")

        return static_object_feature
    
    def get_agent_feature(self, 
                        present_tracked_objects: TrackedObjects,
                        past_tracked_objects: List[TrackedObjects],
                        ego_state: EgoState,
                        history_horizon: float = 3,
                        time_step: float = 0.1,
                        roi_map: Optional[ROIMap] = None,
                        max_vehicle_num: int = 30,
                        max_vru_num: int = 20,
                        ):
        
        history_samples = int(history_horizon / time_step)

        ego_yaw = - ego_state.rear_axle.heading
        rotation_matrix = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)],
                                    [-np.sin(ego_yaw), np.cos(ego_yaw)]])
        ego_position = ego_state.rear_axle.point.array

        present_tracked_objects = present_tracked_objects
        past_dts = past_tracked_objects
        # Pad past_dts: if dt is None, use previous non-None dt (backward padding)
        padded_past_dts = []
        last_valid_dt = None
        for dt in reversed(past_dts):
            if dt is not None:
                padded_past_dts.append(dt)
                last_valid_dt = dt
            else:
                # If dt is None, pad with last valid dt
                if last_valid_dt is not None:
                    padded_past_dts.append(last_valid_dt)
                else:
                    # If no valid dt yet, pad with the first non-None dt in past_dts
                    first_valid_dt = next((x for x in past_dts if x is not None), None)
                    padded_past_dts.append(first_valid_dt)
        padded_past_dts = list(reversed(padded_past_dts))
        past_tracked_objects = padded_past_dts

        agent_id_dict: Dict[str: int] = {}
        agent_dist_dict: Dict[str: float] = {}
        agent_type_dict: Dict[str: TrackedObjectType] = {}
        for idx, agent in enumerate(present_tracked_objects.get_agents()):

            tracked_object_token = agent.track_token

            if agent.tracked_object_type is TrackedObjectType.EGO:
                # Skip ego vehicle
                continue

            tracked_object_distance = np.linalg.norm(
                ego_state.rear_axle.array - agent.center.array
            )
            if tracked_object_distance < 1.0:
                # skip already collided agents
                continue

            if tracked_object_distance > self._map_radius:
                continue
            
            if roi_map is not None:
                polygon_obj = agent.box.geometry
                polygon_coords = np.array(polygon_obj.exterior.coords)
                local_polygon_coords = (polygon_coords - ego_position) @ rotation_matrix
                polygon_obj = Polygon(local_polygon_coords)

                if not roi_map.polygon_in_roi(polygon_obj):
                    continue

            agent_id_dict[tracked_object_token] = idx
            agent_dist_dict[tracked_object_token] = tracked_object_distance
            agent_type_dict[tracked_object_token] = agent.tracked_object_type

        # Sort agent tokens according to distance to ensure stable order in agent_id_dict
        agent_dict = {token: i for i, token in enumerate(sorted(agent_id_dict.keys(), key=lambda x: agent_dist_dict[x]))}

        # limit the number of agents
        if len(agent_dict) > (max_vehicle_num + max_vru_num):
            limited_agent_dict = {}
            vehicle_count = 0
            vru_count = 0
            for token in agent_dict.keys():
                agent_type = agent_type_dict[token]
                if agent_type == TrackedObjectType.VEHICLE and vehicle_count < max_vehicle_num:
                    limited_agent_dict[token] = len(limited_agent_dict)
                    vehicle_count += 1
                elif agent_type in [TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE] and vru_count < max_vru_num:
                    limited_agent_dict[token] = len(limited_agent_dict)
                    vru_count += 1
                if vehicle_count >= max_vehicle_num and vru_count >= max_vru_num:
                    break
            agent_dict = limited_agent_dict

        num_agents = len(agent_dict)

        # extract agent features
        agent_current_state = np.zeros((num_agents, 5), dtype=np.float32)
        agent_history_state = np.zeros((num_agents, history_samples, 5), dtype=np.float32)
        agent_history_mask = np.zeros((num_agents, history_samples), dtype=np.bool_)
        agent_type_ids = np.zeros((num_agents, ), dtype=np.int32)
        agent_geometry = np.zeros((num_agents, 2), dtype=np.float32)

        for agent in present_tracked_objects.get_agents():
            if agent.track_token not in agent_dict:
                continue
            agent_idx = agent_dict[agent.track_token]

            # translate the position to local coordinates
            agent_position = agent.center.array - ego_state.rear_axle.array
            agent_position = np.matmul(agent_position, rotation_matrix)
            agent_headings = agent.center.heading + ego_yaw
            agent_headings = (agent_headings + np.pi) % (2 * np.pi) - np.pi
            agent_velocity = np.array([agent.velocity.magnitude() * np.cos(agent_headings),
                                       agent.velocity.magnitude() * np.sin(agent_headings)])
            
            agent_current_state[agent_idx, :2] = agent_position
            agent_current_state[agent_idx, 2] = agent_headings
            agent_current_state[agent_idx, 3:5] = agent_velocity
            agent_type_ids[agent_idx] = agent.tracked_object_type.value + 1
            agent_geometry[agent_idx, 0] = agent.box.half_length
            agent_geometry[agent_idx, 1] = agent.box.half_width


        for t_idx, tracked_objects in enumerate(past_tracked_objects):
            for agent in tracked_objects.get_agents():

                if agent.track_token not in agent_dict:
                    continue

                agent_idx = agent_dict[agent.track_token]
                if t_idx >= history_samples:
                    break

                # translate the position to local coordinates
                agent_position = agent.center.array - ego_state.rear_axle.array
                agent_position = np.matmul(agent_position, rotation_matrix)
                agent_headings = agent.center.heading + ego_yaw
                agent_headings = (agent_headings + np.pi) % (2 * np.pi) - np.pi
                agent_velocity = np.array([agent.velocity.magnitude() * np.cos(agent_headings),
                                        agent.velocity.magnitude() * np.sin(agent_headings)])
                
                agent_history_state[agent_idx, t_idx, :2] = agent_position
                agent_history_state[agent_idx, t_idx, 2] = agent_headings
                agent_history_state[agent_idx, t_idx, 3:5] = agent_velocity
                agent_history_mask[agent_idx, t_idx] = 1.0

        # interpolate with histroy and current
        agent_all_state = np.concatenate((agent_history_state, 
                                          agent_current_state[:, np.newaxis, :],
                                          ), axis=1)
        agent_all_mask = np.concatenate((agent_history_mask, 
                                         np.ones((num_agents,1), dtype=np.bool_),
                                         ), axis=1)
        agent_all_state_interp = np.zeros((agent_all_state.shape[0], agent_all_state.shape[1], 6), 
                                          dtype=np.float32)
        for agent_idx in range(num_agents):
            valid_indices = np.where(agent_all_mask[agent_idx])[0]

            # if less than 2 valid points, set history to current state with zero velocity
            if len(valid_indices) < 2:
                agent_all_state_interp[agent_idx, :, :3] = agent_all_state[agent_idx, -1, :3]
                continue

            interp_func = interp1d(valid_indices,
                                   agent_all_state[agent_idx, valid_indices, :],
                                   kind='linear',
                                   axis=0,
                                   fill_value='extrapolate')
            
            interp_indices = np.arange(valid_indices[0], valid_indices[-1]+1)
            agent_all_state[agent_idx, interp_indices] = interp_func(interp_indices)
            agent_all_state_interp[agent_idx, interp_indices] = process_agent_states(
                agent_all_state[agent_idx, interp_indices], dt=time_step)[0]

            agent_all_mask[agent_idx, interp_indices] = True
        
        agent_current_state = agent_all_state_interp[:, history_samples, :]
        agent_history_state = agent_all_state_interp[:, :history_samples, :]
        agent_history_mask = agent_all_mask[:, :history_samples]

        # print(f'agent_current_state: {np.array2string(agent_current_state, formatter={"float_kind":lambda x: f"{x:.2f}"})}')

        dynamic_agent_feature = {
            "agent_current_state": agent_current_state,
            "agent_history_state": agent_history_state,
            "agent_geometry": agent_geometry,
            "agent_history_mask": agent_history_mask,
            "agent_type": agent_type_ids,
        }
        # Check for invalid values in dynamic_agent_feature
        for key, value in dynamic_agent_feature.items():
            if isinstance(value, np.ndarray):
                if np.isnan(value).any() or np.isinf(value).any():
                    nan_indices = np.argwhere(np.isnan(value))
                    inf_indices = np.argwhere(np.isinf(value))
                    if nan_indices.size > 0:
                        logging.warning(f"NaN detected in '{key}' at positions: {nan_indices.tolist()}")
                    if inf_indices.size > 0:
                        logging.warning(f"Inf detected in '{key}' at positions: {inf_indices.tolist()}")
        return dynamic_agent_feature
    
    def get_agent_target_with_scene(self,
                                    scenario: AbstractScenario,
                                    ego_state: EgoState,
                                    future_horizon: float = 6,
                                    time_step: float = 0.1,
                                    roi_map: Optional[ROIMap] = None,
                                    max_vehicle_num: int = 30,
                                    max_vru_num: int = 10,
                                    ) -> Tuple[np.ndarray, np.ndarray]:

        future_samples = int(future_horizon / time_step)

        ego_yaw = - ego_state.rear_axle.heading
        rotation_matrix = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)],
                                    [-np.sin(ego_yaw), np.cos(ego_yaw)]])
        ego_position = ego_state.rear_axle.point.array

        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects

        future_dts_gen = scenario.get_future_tracked_objects(
                iteration=0,
                time_horizon=future_horizon,
                num_samples=future_samples,
        )
        future_dts = [dt for dt in future_dts_gen]

        # Pad future_dts: if dt is None, use previous non-None dt
        padded_future_dts = []
        last_valid_dt = None
        for dt in future_dts:
            if dt is not None:
                padded_future_dts.append(dt)
                last_valid_dt = dt
            else:
                # If dt is None, pad with last valid dt
                if last_valid_dt is not None:
                    padded_future_dts.append(last_valid_dt)
                else:
                    # If no valid dt yet, pad with the first non-None dt in future_dts
                    first_valid_dt = next((x for x in future_dts if x is not None), None)
                    padded_future_dts.append(first_valid_dt)
        
        future_tracked_objects = [dt.tracked_objects for dt in padded_future_dts]


        agent_dict : Dict[str: int] = {}
        agent_dist_dict : Dict[str: float] = {}
        agent_type_dict : Dict[str: TrackedObjectType] = {}
        for idx, agent in enumerate(present_tracked_objects.get_agents()):
            tracked_object_type = agent.tracked_object_type

            if agent.tracked_object_type is TrackedObjectType.EGO:
                # Skip ego vehicle
                continue

            tracked_object_token = agent.track_token
            tracked_object_distance = np.linalg.norm(
                ego_state.rear_axle.array - agent.center.array
            )

            if tracked_object_distance > self._map_radius:
                continue

            if roi_map is not None:
                polygon_obj = agent.box.geometry
                polygon_coords = np.array(polygon_obj.exterior.coords)
                local_polyon_coords = (polygon_coords - ego_position) @ rotation_matrix
                polygon_obj = Polygon(local_polyon_coords)
                if not roi_map.polygon_in_roi(polygon_obj):
                    continue

            agent_dict[tracked_object_token] = idx
            agent_dist_dict[tracked_object_token] = tracked_object_distance
            agent_type_dict[tracked_object_token] = tracked_object_type

        # limit the number of agents
        agent_dict = {token: i for i, token in enumerate(sorted(agent_dict.keys(), key=lambda x: agent_dist_dict[x]))}
        if len(agent_dict) > (max_vehicle_num + max_vru_num):
            limited_agent_dict = {}
            vehicle_count = 0
            vru_count = 0
            for token in agent_dict.keys():
                agent_type = agent_type_dict[token]
                if agent_type == TrackedObjectType.VEHICLE and vehicle_count < max_vehicle_num:
                    limited_agent_dict[token] = len(limited_agent_dict)
                    vehicle_count += 1
                elif agent_type in [TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE] and vru_count < max_vru_num:
                    limited_agent_dict[token] = len(limited_agent_dict)
                    vru_count += 1
                if vehicle_count >= max_vehicle_num and vru_count >= max_vru_num:
                    break
            agent_dict = limited_agent_dict

        # Sort agent tokens to ensure stable order in agent_dict
        agent_dict = {token: i for i, token in enumerate(sorted(agent_dict.keys()))}

        num_agents = len(agent_dict)

        # extract agent features    
        agent_future_state = np.zeros((num_agents, future_samples, 5), dtype=np.float32)
        agent_future_mask = np.zeros((num_agents, future_samples), dtype=np.bool_)


        for t_idx, tracked_objects in enumerate(future_tracked_objects):
            for agent in tracked_objects.get_agents():
                if agent.track_token not in agent_dict:
                    continue
                agent_idx = agent_dict[agent.track_token]
                if t_idx >= future_samples:
                    break

                # translate the position to local coordinates
                agent_position = agent.center.array - ego_state.rear_axle.array
                agent_position = np.matmul(agent_position, rotation_matrix)
                agent_headings = agent.center.heading + ego_yaw
                agent_headings = (agent_headings + np.pi) % (2 * np.pi) - np.pi
                agent_velocity = np.array([agent.velocity.magnitude() * np.cos(agent_headings),
                                        agent.velocity.magnitude() * np.sin(agent_headings)])
                
                agent_future_state[agent_idx, t_idx, :2] = agent_position
                agent_future_state[agent_idx, t_idx, 2] = agent_headings
                agent_future_state[agent_idx, t_idx, 3:5] = agent_velocity
                agent_future_mask[agent_idx, t_idx] = 1.0


        # Check for invalid values in agent_future_state and agent_future_mask
        if np.isnan(agent_future_state).any() or np.isinf(agent_future_state).any():
            nan_indices = np.argwhere(np.isnan(agent_future_state))
            inf_indices = np.argwhere(np.isinf(agent_future_state))
            if nan_indices.size > 0:
                logging.warning(f"NaN detected in 'agent_future_state' at positions: {nan_indices.tolist()}")
            if inf_indices.size > 0:
                logging.warning(f"Inf detected in 'agent_future_state' at positions: {inf_indices.tolist()}")
        if np.isnan(agent_future_mask).any() or np.isinf(agent_future_mask).any():
            nan_indices = np.argwhere(np.isnan(agent_future_mask))
            inf_indices = np.argwhere(np.isinf(agent_future_mask))
            if nan_indices.size > 0:
                logging.warning(f"NaN detected in 'agent_future_mask' at positions: {nan_indices.tolist()}")
            if inf_indices.size > 0:
                logging.warning(f"Inf detected in 'agent_future_mask' at positions: {inf_indices.tolist()}")

        return agent_future_state, agent_future_mask


    def get_agent_feature_target(self, 
                        present_tracked_objects: TrackedObjects,
                        past_tracked_objects: List[TrackedObjects],
                        future_tracked_objects: List[TrackedObjects],
                        ego_state: EgoState,
                        history_horizon: float = 3,
                        future_horizon: float = 6,
                        time_step: float = 0.1,
                        roi_map: Optional[ROIMap] = None,
                        max_vehicle_num: int = 30,
                        max_vru_num: int = 20,
                        ):
        
        history_samples = int(history_horizon / time_step)
        future_samples = int(future_horizon / time_step)

        ego_yaw = - ego_state.rear_axle.heading
        rotation_matrix = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)],
                                    [-np.sin(ego_yaw), np.cos(ego_yaw)]])
        ego_position = ego_state.rear_axle.point.array

        agent_id_dict: Dict[str: int] = {}
        agent_dist_dict: Dict[str: float] = {}
        agent_type_dict: Dict[str: TrackedObjectType] = {}
        for idx, agent in enumerate(present_tracked_objects.get_agents()):

            tracked_object_token = agent.track_token

            if agent.tracked_object_type is TrackedObjectType.EGO:
                # Skip ego vehicle
                continue

            tracked_object_distance = np.linalg.norm(
                ego_state.rear_axle.array - agent.center.array
            )

            if tracked_object_distance > self._map_radius:
                continue
            
            if roi_map is not None:
                polygon_obj = agent.box.geometry
                polygon_coords = np.array(polygon_obj.exterior.coords)
                local_polygon_coords = (polygon_coords - ego_position) @ rotation_matrix
                polygon_obj = Polygon(local_polygon_coords)

                if not roi_map.polygon_in_roi(polygon_obj):
                    continue

            agent_id_dict[tracked_object_token] = idx
            agent_dist_dict[tracked_object_token] = tracked_object_distance
            agent_type_dict[tracked_object_token] = agent.tracked_object_type

        # Sort agent tokens according to distance to ensure stable order in agent_id_dict
        agent_dict = {token: i for i, token in enumerate(sorted(agent_id_dict.keys(), key=lambda x: agent_dist_dict[x]))}

        # limit the number of agents
        if len(agent_dict) > (max_vehicle_num + max_vru_num):
            limited_agent_dict = {}
            vehicle_count = 0
            vru_count = 0
            for token in agent_dict.keys():
                agent_type = agent_type_dict[token]
                if agent_type == TrackedObjectType.VEHICLE and vehicle_count < max_vehicle_num:
                    limited_agent_dict[token] = len(limited_agent_dict)
                    vehicle_count += 1
                elif agent_type in [TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE] and vru_count < max_vru_num:
                    limited_agent_dict[token] = len(limited_agent_dict)
                    vru_count += 1
                if vehicle_count >= max_vehicle_num and vru_count >= max_vru_num:
                    break
            agent_dict = limited_agent_dict

        num_agents = len(agent_dict)

        # extract agent features
        agent_current_state = np.zeros((num_agents, 5), dtype=np.float32)
        agent_history_state = np.zeros((num_agents, history_samples, 5), dtype=np.float32)
        agent_history_mask = np.zeros((num_agents, history_samples), dtype=np.bool_)
        agent_type_ids = np.zeros((num_agents, ), dtype=np.int32)
        agent_geometry = np.zeros((num_agents, 2), dtype=np.float32)

        for agent in present_tracked_objects.get_agents():
            if agent.track_token not in agent_dict:
                continue
            agent_idx = agent_dict[agent.track_token]

            # translate the position to local coordinates
            agent_position = agent.center.array - ego_state.rear_axle.array
            agent_position = np.matmul(agent_position, rotation_matrix)
            agent_headings = agent.center.heading + ego_yaw
            agent_headings = (agent_headings + np.pi) % (2 * np.pi) - np.pi
            agent_velocity = np.array([agent.velocity.magnitude() * np.cos(agent_headings),
                                       agent.velocity.magnitude() * np.sin(agent_headings)])
            
            agent_current_state[agent_idx, :2] = agent_position
            agent_current_state[agent_idx, 2] = agent_headings
            agent_current_state[agent_idx, 3:5] = agent_velocity
            agent_type_ids[agent_idx] = agent.tracked_object_type.value + 1
            agent_geometry[agent_idx, 0] = agent.box.half_length
            agent_geometry[agent_idx, 1] = agent.box.half_width
        
        del present_tracked_objects
        
        # extract agent features
        for t_idx, tracked_objects in enumerate(past_tracked_objects):

            if tracked_objects is None:
                continue

            for agent in tracked_objects.get_agents():

                if agent.track_token not in agent_dict:
                    continue

                agent_idx = agent_dict[agent.track_token]
                if t_idx >= history_samples:
                    break

                # translate the position to local coordinates
                agent_position = agent.center.array - ego_state.rear_axle.array
                agent_position = np.matmul(agent_position, rotation_matrix)
                agent_headings = agent.center.heading + ego_yaw
                agent_headings = (agent_headings + np.pi) % (2 * np.pi) - np.pi
                agent_velocity = np.array([agent.velocity.magnitude() * np.cos(agent_headings),
                                        agent.velocity.magnitude() * np.sin(agent_headings)])
                
                agent_history_state[agent_idx, t_idx, :2] = agent_position
                agent_history_state[agent_idx, t_idx, 2] = agent_headings
                agent_history_state[agent_idx, t_idx, 3:5] = agent_velocity
                agent_history_mask[agent_idx, t_idx] = True
        
        del past_tracked_objects
        
        # extract agent features    
        agent_future_state = np.zeros((num_agents, future_samples, 5), dtype=np.float32)
        agent_future_mask = np.zeros((num_agents, future_samples), dtype=np.bool_)

        for t_idx, tracked_objects in enumerate(future_tracked_objects):

            if tracked_objects is None:
                continue
            for agent in tracked_objects.get_agents():
                if agent.track_token not in agent_dict:
                    continue
                agent_idx = agent_dict[agent.track_token]
                if t_idx >= future_samples:
                    break

                # translate the position to local coordinates
                agent_position = agent.center.array - ego_state.rear_axle.array
                agent_position = np.matmul(agent_position, rotation_matrix)
                agent_headings = agent.center.heading + ego_yaw
                agent_headings = (agent_headings + np.pi) % (2 * np.pi) - np.pi
                agent_velocity = np.array([agent.velocity.magnitude() * np.cos(agent_headings),
                                        agent.velocity.magnitude() * np.sin(agent_headings)])
                
                agent_future_state[agent_idx, t_idx, :2] = agent_position
                agent_future_state[agent_idx, t_idx, 2] = agent_headings
                agent_future_state[agent_idx, t_idx, 3:5] = agent_velocity
                agent_future_mask[agent_idx, t_idx] = True

        del future_tracked_objects

        # interpolate with histroy and future mask
        agent_all_state = np.concatenate((agent_history_state, 
                                          agent_current_state[:, np.newaxis, :],
                                          agent_future_state), axis=1)
        agent_all_mask = np.concatenate((agent_history_mask, 
                                         np.ones((num_agents,1), dtype=np.bool_),
                                         agent_future_mask), axis=1)
        agent_all_state_interp = np.zeros((agent_all_state.shape[0], agent_all_state.shape[1], 6), dtype=np.float32)
        for agent_idx in range(num_agents):
            valid_indices = np.where(agent_all_mask[agent_idx])[0]

            if len(valid_indices) < 2:
                agent_all_state_interp[agent_idx, :, :3] = agent_all_state[agent_idx, -1, :3]
                continue

            interp_func = interp1d(valid_indices,
                                   agent_all_state[agent_idx, valid_indices, :],
                                   kind='linear',
                                   axis=0,
                                   fill_value='extrapolate')
            
            interp_indices = np.arange(valid_indices[0], valid_indices[-1]+1)
            agent_all_state[agent_idx, interp_indices] = interp_func(interp_indices)
            agent_all_state_interp[agent_idx, interp_indices] = process_agent_states(
                agent_all_state[agent_idx, interp_indices], dt=time_step)[0]

            agent_all_mask[agent_idx, interp_indices] = True

        agent_current_state = agent_all_state_interp[:, history_samples, :]
        agent_history_state = agent_all_state_interp[:, :history_samples, :]
        agent_future_state = agent_all_state_interp[:, history_samples+1:, :]

        agent_history_mask = agent_all_mask[:, :history_samples]
        agent_future_mask = agent_all_mask[:, history_samples+1:]
        
        agent_feature = {
            "agent_current_state": agent_current_state,
            "agent_history_state": agent_history_state,
            "agent_geometry": agent_geometry,
            "agent_history_mask": agent_history_mask,
            "agent_type": agent_type_ids,
        }

        agent_target = {
            "agent_future_state": agent_future_state,
            "agent_future_mask": agent_future_mask,
        }

        return agent_feature, agent_target
