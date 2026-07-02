import torch
import numpy as np
from scipy import signal
from typing import Any, Dict, List, Set, Tuple
from scipy.interpolate import interp1d

from nuplan.planning.simulation.planner.abstract_planner import Optional, PlannerInput, PlannerInitialization
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.common.actor_state.vehicle_parameters import (
    VehicleParameters,
    get_pacifica_parameters,
)
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from lpl_planner.planning.scene.agent.agent_manager import AgentManager
from lpl_planner.planning.scene.map.lane_map import LaneMap
from lpl_planner.planning.scene.map.occupancy_map import OccupancyMap
from lpl_planner.planning.scene.utils.scene_index import STATEINDEX, ACTIONINDEX
from lpl_planner.planning.scene.utils.simulator import Simulator
from lpl_planner.planning.scene.map.map_utils.roi_segement import ROIMap
from lpl_planner.planning.scene.scene_feature.features import SceneFeature, AgentPrediction

from scipy.signal import savgol_filter
from shapely import creation
from shapely.geometry import Polygon

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import collections
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


SCENE_MANAGER_COLLISION_DTYPE = np.float32
SCENE_MANAGER_STOPPED_SPEED_THRESHOLD = 5e-03
SCENE_MANAGER_AGENT_FUTURE_DEFAULT_DT = 0.2
COLLISION_KIND_AGENT = 0
COLLISION_KIND_STATIC = 1


class SceneManager:
    def __init__(self, start=None, goal=None,
                 visualize=False,
                 num_trajectories=5,
                 time_step = 0.1,
                 planning_step = 80,
                 map_radius = 100,
                 num_samples = 500,
                 use_local_coordinate = True,
                 simluate_expert_trajectory = True,
                 tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32},
                 use_dynamics = False,
                 use_ref_path = True):
        self.t = 0
        self.time_step = time_step
        self.state_ranges = [
            (-30, 50),
            (-40, 40)
        ]
        # print(f"SceneManager initialized with time step: {self.time_step}, planning step: {planning_step}, map radius: {map_radius}, num_samples: {num_samples}")
        self.num_trajectories = num_trajectories
        self.visualize = visualize
        self.num_samples = num_samples
        self.horizon = planning_step
        self.map_radius = map_radius
        history_sampling = TrajectorySampling(time_horizon=3.0, interval_length=time_step)
        future_sampling = TrajectorySampling(time_horizon=11.0, interval_length=time_step)
        self.history_sampling = history_sampling
        self.future_sampling = future_sampling
        self.use_ref_path = use_ref_path

        self.history_horizon = history_sampling.time_horizon
        self.future_horizon = future_sampling.time_horizon

        self.start = start or torch.tensor([0,1.5,0,3,0,0,0,0], **tensor_args)
        self.goal = goal or torch.tensor([2, 2], **tensor_args)
        self.state = self.start
        self.current_iteration = 0

        self.wheel_base = 2.8
        
        self.dynamics = None
            
        self.agent_manager = AgentManager(predicton_step=planning_step+30,
                                          use_local_coords=use_local_coordinate,
                                          map_radius=map_radius,
                                          tensor_args = tensor_args)
        
        self.lane_map = LaneMap(use_local_coords=use_local_coordinate,
                                use_ref_path=use_ref_path,
                                tensor_args=tensor_args)

        
        self.simluate_expert_trajectory = simluate_expert_trajectory

        self.half_length, self.half_width, self.rear_axle_to_center = (
        1.2,
        0.6,
        1.4,
        )   

        self.trajectory_artist = None
        self.tensor_args = tensor_args
        self.use_local_coordinate = use_local_coordinate        
        self.ego_history_buffer = []
        self.ego_history_is_sim_buffer = []
        self._current_planner_input: Optional[PlannerInput] = None
        self._future_collision_map: Optional[List[OccupancyMap]] = None
        self._future_collision_meta: Optional[List[Dict[str, np.ndarray]]] = None
        self._cached_static_obstacle_position: Optional[np.ndarray] = None
        self._prediction_cache_sampling: Optional[TrajectorySampling] = None
        self.collieded_track_tokens: Set[str] = set()

        if self.visualize:
            self.start_visualization()

    @staticmethod
    def _resample_agent_future_states(
        agent_current_state: np.ndarray,
        agent_future_state: np.ndarray,
        agent_future_mask: np.ndarray,
        target_num_steps: int,
        target_dt: float,
        source_dt: float = SCENE_MANAGER_AGENT_FUTURE_DEFAULT_DT,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if target_num_steps <= 0:
            num_agents = min(agent_current_state.shape[0], agent_future_state.shape[0], agent_future_mask.shape[0])
            state_dim = agent_future_state.shape[-1] if agent_future_state.ndim == 3 else agent_current_state.shape[-1]
            return (
                np.zeros((num_agents, 0, state_dim), dtype=SCENE_MANAGER_COLLISION_DTYPE),
                np.zeros((num_agents, 0), dtype=np.bool_),
            )

        num_agents = min(agent_current_state.shape[0], agent_future_state.shape[0], agent_future_mask.shape[0])
        state_dim = agent_future_state.shape[-1]
        if num_agents == 0:
            return (
                np.zeros((0, target_num_steps, state_dim), dtype=SCENE_MANAGER_COLLISION_DTYPE),
                np.zeros((0, target_num_steps), dtype=np.bool_),
            )

        target_times = np.arange(1, target_num_steps + 1, dtype=SCENE_MANAGER_COLLISION_DTYPE) * target_dt
        future_times = np.arange(1, agent_future_state.shape[1] + 1, dtype=SCENE_MANAGER_COLLISION_DTYPE) * source_dt
        source_times = np.concatenate((np.array([0.0], dtype=SCENE_MANAGER_COLLISION_DTYPE), future_times), axis=0)

        resampled_states = np.zeros((num_agents, target_num_steps, state_dim), dtype=SCENE_MANAGER_COLLISION_DTYPE)
        resampled_mask = np.zeros((num_agents, target_num_steps), dtype=np.bool_)

        for agent_idx in range(num_agents):
            source_states = np.concatenate(
                (
                    agent_current_state[agent_idx : agent_idx + 1, :state_dim],
                    agent_future_state[agent_idx],
                ),
                axis=0,
            )
            source_mask = np.concatenate(
                (np.array([True], dtype=np.bool_), agent_future_mask[agent_idx].astype(np.bool_)),
                axis=0,
            )
            source_valid = source_mask & np.isfinite(source_states).all(axis=-1)
            if not np.any(source_valid):
                continue

            valid_times = source_times[source_valid]
            valid_states = source_states[source_valid]

            if valid_times.size == 1:
                resampled_states[agent_idx] = valid_states[0]
                resampled_mask[agent_idx] = True
                continue

            in_range = (target_times >= valid_times[0] - 1e-6) & (target_times <= valid_times[-1] + 1e-6)
            if not np.any(in_range):
                continue

            interp_times = target_times[in_range]
            interp_state = np.empty((interp_times.shape[0], state_dim), dtype=SCENE_MANAGER_COLLISION_DTYPE)
            for dim_idx in range(state_dim):
                values = valid_states[:, dim_idx]
                if dim_idx == 2:
                    values = np.unwrap(values)
                interp_fn = interp1d(
                    valid_times,
                    values,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(values[0], values[-1]),
                    assume_sorted=True,
                )
                interp_state[:, dim_idx] = interp_fn(interp_times)

            if state_dim > 2:
                interp_state[:, 2] = np.arctan2(np.sin(interp_state[:, 2]), np.cos(interp_state[:, 2]))

            resampled_states[agent_idx, in_range] = interp_state
            resampled_mask[agent_idx, in_range] = True

        return resampled_states, resampled_mask


    def init_from_nuplan_scenario(self, 
                                  scenario: AbstractScenario,
                                  use_occupancy_map: bool = False,
                                  use_vehicle_dynamics: bool = False,
                                  iteration: int = 0,
                                  use_route_correction: bool = False,
                                  use_scenario_for_route_correction: bool = False):

        vehicle_parameter = scenario.ego_vehicle_parameters
        self.half_width = vehicle_parameter.half_width
        self.half_length = vehicle_parameter.half_length
        self.rear_axle_to_center = vehicle_parameter.rear_axle_to_center
        self.start_time = scenario.start_time.time_s
        self.scenario = scenario
        self.wheel_base = vehicle_parameter.wheel_base

        
        # always update map first before agent manager
        ego_state = scenario.get_ego_state_at_iteration(iteration)

        self.lane_map.init_from_scenario(ego_state=ego_state,
                             map_api=scenario.map_api,
                             route_roadblock_ids=scenario.get_route_roadblock_ids(),
                             scenario=scenario,
                             use_route_correction=use_route_correction,
                             use_scenario_for_route_correction=use_scenario_for_route_correction,
                             )
        
        if use_occupancy_map:
            prediction_sampling = TrajectorySampling(num_poses=self.horizon+10,
                                                 interval_length=self.time_step)
            self.agent_manager.init_with_nuplan_scene(scenario,
                                                    prediction_sampling,
                                                    self.lane_map)
        
        
        self.x0,self.y0,self.heading0 = ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading
        self.ego_state = ego_state
        self._update_state_from_scenario(ego_state)
        self.current_iteration = 0

        if self.visualize:
            self.clear_artist(self.road_artists)
            self.draw_road_lines()

    def init_from_planner_init(self, 
                               planner_input: PlannerInitialization,
                               scenario: Optional[AbstractScenario] = None ):

        self.map_api = planner_input.map_api

        self.lane_map.init_from_planner_init(planner_input.map_api,
                                             planner_input.route_roadblock_ids,
                                             mission_goal=planner_input.mission_goal,
                                             scenario=scenario)

        prediction_sampling = TrajectorySampling(num_poses=self.horizon+10,
                                                 interval_length=self.time_step)
        
        self.agent_manager.init_with_planner_input(prediction_sampling,
                                                   self.lane_map)
        self.current_iteration = 0

    
    def extract_feature_from_simulation(self, 
                                      planner_input: PlannerInput,
                                      use_roi_map: bool = False,
                                      ):
        history = planner_input.history
        ego_state = history.ego_states[-1]
        tl_data = planner_input.traffic_light_data
        # logger.info(f"Extracting feature from scenario: {scenario_id}")
        # print(f"extract feature from scenario: {scenario.token}, ego_state: {ego_state.car_footprint.center.array}, tl_data: {tl_data}")
        # extract ref_path feature
        ref_path_feature = self.lane_map.get_ref_path_feature()
        extract_step = round(self.time_step / history.sample_interval)
        
        # ! make sure history has enough length (adjust simulation history horizon in config if needed)
        extract_indices = list(np.arange(-1-extract_step, -len(history.ego_states)-1, -extract_step))[::-1]

        # build ROIMap
        if use_roi_map:
            roi_map = ROIMap(ref_path=ref_path_feature,
                            s_min = -30,
                            s_max = self.map_radius,
                            width = 40,
                            extrapolate_back=True)
        else:
            roi_map = None
        # extract road feature
        route_feature = self.lane_map.get_route_feature(ego_state=ego_state,
                                                        tl_data=tl_data,
                                                        roi_map=roi_map)
        road_feature = self.lane_map.get_road_feature(ego_state=ego_state, 
                                                      tl_data=tl_data,
                                                      roi_map=roi_map)
        tracked_objects_list = [
            observation.tracked_objects for observation in history.observations
        ]

        # extract agent feature
        present_tracked_objects = tracked_objects_list[-1]
        history_pose_num =int(self.history_horizon / self.time_step)
        past_dts = [tracked_objects_list[idx] for idx in extract_indices[-history_pose_num:]]
        # print(f'past dt length: {len(past_dts)}, expected: {history_pose_num}')
        
        agent_feature = self.agent_manager.get_agent_feature(present_tracked_objects,
                                                             past_tracked_objects=past_dts,
                                                            ego_state=ego_state,
                                                            history_horizon=self.history_horizon,
                                                            time_step=self.time_step,
                                                            roi_map=roi_map)
        static_obstacle_feature = self.agent_manager.get_static_obstacles_feature(present_tracked_objects,
                                                                                ego_state=ego_state,
                                                                                roi_map=roi_map)
        ego_feature = self._build_ego_feature_from_history(
            ego_state=ego_state,
            ego_history_states=self._sample_ego_history_states(
                ego_history_states=planner_input.history.ego_states,
                current_state=ego_state,
            ),
        )
        
        return {
            'road_feature': road_feature,
            'ref_path_feature': ref_path_feature,
            'agent_feature': agent_feature,
            'static_obstacle_feature': static_obstacle_feature,
            'ego_feature': ego_feature,
            'route_feature': route_feature
        }
    
    def extract_feature_target_from_scenario(self, 
                                             scenario: AbstractScenario, 
                                             use_roi_map: bool = False,
                                             iteration: int = 0,
                                             ego_state: Optional[EgoState] = None,
                                             ego_history: Optional[List[EgoState]] = None,
                                             use_route_correction: bool = False):
        scenario_id = scenario.token
        
        # logger.info(f"Extracting feature from scenario: {scenario_id}")
        if ego_state is None:
            ego_state = scenario.get_ego_state_at_iteration(iteration)

        if not self.lane_map.initialized:
            self.lane_map.init_from_scenario(ego_state=ego_state,
                                map_api=scenario.map_api,
                                route_roadblock_ids=scenario.get_route_roadblock_ids(),
                                scenario=scenario,
                                use_route_correction=use_route_correction,
                                use_scenario_for_route_correction=use_route_correction,
                                )
        
        tl_data_gen = scenario.get_traffic_light_status_at_iteration(iteration)
        tl_data = [tl_data for tl_data in tl_data_gen if tl_data is not None]
        
        if self.use_ref_path:
            ref_path_feature = self.lane_map.get_ref_path_feature()
        else:
            ref_path_feature = None

        # build ROIMap
        if use_roi_map:
            roi_map = ROIMap(ref_path=ref_path_feature,
                            s_min = -30,
                            s_max = self.map_radius,
                            width = 40,
                            extrapolate_back=True)
        else:
            roi_map = None
        # extract road feature
        route_feature = self.lane_map.get_route_feature(ego_state=ego_state,
                                                        tl_data=tl_data,
                                                        roi_map=roi_map)
        road_feature = self.lane_map.get_road_feature(ego_state=ego_state, 
                                                      tl_data=tl_data,
                                                      roi_map=roi_map)
        # extract agent feature
        present_tracked_objects = scenario.get_tracked_objects_at_iteration(iteration).tracked_objects
        past_dts_gen = scenario.get_past_tracked_objects(
                iteration=iteration,
                time_horizon=self.history_horizon,
                num_samples=self.history_sampling.num_poses,
            )
        past_dts = [dt.tracked_objects for dt in past_dts_gen]
        
        future_dts_gen = scenario.get_future_tracked_objects(
                iteration=iteration,
                time_horizon=self.future_horizon,
                num_samples=int(self.future_horizon / self.time_step),
        )
        future_dts = [dt.tracked_objects for dt in future_dts_gen]
        
        agent_feature, agent_target = self.agent_manager.get_agent_feature_target(
            present_tracked_objects,
            past_tracked_objects=past_dts,
            future_tracked_objects=future_dts,
            future_horizon=self.future_horizon,
            ego_state=ego_state,
            history_horizon=self.history_horizon,
            time_step=self.time_step,
            roi_map=roi_map
        )
        static_obstacle_feature = self.agent_manager.get_static_obstacles_feature(present_tracked_objects,
                                                                                ego_state=ego_state,
                                                                                roi_map=roi_map)
        if ego_history is not None and len(ego_history) > 0:
            ego_history_states = self._sample_ego_history_states(
                ego_history_states=ego_history,
                current_state=ego_state,
            )
        else:
            scenario_ego_history = scenario.get_ego_past_trajectory(
                iteration=iteration,
                time_horizon=self.history_horizon,
                num_samples=self.history_sampling.num_poses,
            )
            ego_history_states = [state for state in scenario_ego_history]
            ego_history_states.append(ego_state)

        ego_feature = self._build_ego_feature_from_history(
            ego_state=ego_state,
            ego_history_states=ego_history_states,
        )
        
        feature = {
            'scenario_id': scenario_id,
            'road_feature': road_feature,
            'ref_path_feature': ref_path_feature,
            'agent_feature': agent_feature,
            'static_obstacle_feature': static_obstacle_feature,
            'ego_feature': ego_feature,
            'route_feature': route_feature
        }

        target = {'agent_future_state': agent_target['agent_future_state'],
                  'agent_future_mask': agent_target['agent_future_mask']}

        return feature, target

    def _sample_ego_history_states(
        self,
        ego_history_states: List[EgoState],
        current_state: EgoState,
    ) -> List[EgoState]:
        if len(ego_history_states) == 0:
            return [current_state]

        if len(ego_history_states) == 1:
            sampled_history = [ego_history_states[0]]
        else:
            timestamps_s = np.asarray(
                [state.time_point.time_s for state in ego_history_states],
                dtype=np.float64,
            )
            sample_diffs = np.diff(timestamps_s)
            positive_diffs = sample_diffs[sample_diffs > 1e-6]
            sample_interval = float(np.median(positive_diffs)) if positive_diffs.size > 0 else self.time_step
            extract_step = max(int(round(self.time_step / max(sample_interval, 1e-6))), 1)
            extract_indices = list(np.arange(-1 - extract_step, -len(ego_history_states) - 1, -extract_step))[::-1]
            history_pose_num = int(self.history_horizon / self.time_step)
            sampled_history = [ego_history_states[idx] for idx in extract_indices[-history_pose_num:]]

        sampled_history.append(current_state)
        return sampled_history

    def _build_ego_feature_from_history(
        self,
        ego_state: EgoState,
        ego_history_states: List[EgoState],
    ) -> Dict[str, np.ndarray]:
        ego_history_traj = np.zeros((len(ego_history_states), 6))
        for idx, history_state in enumerate(ego_history_states):
            ego_history_traj[idx, 0] = history_state.car_footprint.rear_axle.x
            ego_history_traj[idx, 1] = history_state.car_footprint.rear_axle.y
            ego_history_traj[idx, 2] = history_state.car_footprint.rear_axle.heading
            ego_history_traj[idx, 3] = history_state.dynamic_car_state.speed
            ego_history_traj[idx, 4] = history_state.dynamic_car_state.acceleration
            ego_history_traj[idx, 5] = history_state.dynamic_car_state.angular_velocity

        dt = self.time_step
        window_length = min(len(ego_history_traj), 8)
        if window_length % 2 == 0:
            window_length -= 1
        if window_length >= 3:
            yaw_rate = savgol_filter(
                ego_history_traj[:, 2],
                polyorder=min(2, window_length - 1),
                window_length=window_length,
                deriv=1,
                delta=dt,
                axis=0,
            )
            ego_history_traj[:, 5] = yaw_rate

        init_yaw = -ego_history_traj[-1, 2]
        rotation_matrix = np.array([
            [np.cos(init_yaw), -np.sin(init_yaw)],
            [np.sin(init_yaw), np.cos(init_yaw)]
        ])
        ego_history_local = ego_history_traj.copy()
        ego_history_local[:, 2] = ego_history_traj[:, 2] + init_yaw
        ego_history_local[:, 2] = (np.mod(ego_history_local[:, 2] + np.pi, 2 * np.pi)) - np.pi
        ego_history_local[:, :2] = np.dot(ego_history_traj[:, :2] - ego_history_traj[-1, :2], rotation_matrix.T)

        ego_geometry = np.array([
            ego_state.car_footprint.half_width,
            ego_state.car_footprint.half_length,
            ego_state.car_footprint.rear_axle_to_center_dist,
        ])

        ego_history_local = ego_history_local.astype(np.float32, copy=False)
        ego_geometry = ego_geometry.astype(np.float32, copy=False)

        return {
            'ego_current_state': ego_history_local[-1, :],
            'ego_history_state': ego_history_local[:-1, :],
            'ego_geometry': ego_geometry,
        }

    

    def reset(self):
        # self.state = self.start
        if self.visualize:
            self.clear_artist(self.rollout_artist)
            self.rollout_artist = None
            self.clear_trajectory()
            self.trajectory_artist = None

    def step(self, action):
        self.t += self.time_step
        self.state = self.dynamics(self.state, action)
        self.agent_manager.step(self.t)
        self.state_ranges = [
            (self.state[0].cpu().numpy()-5, self.state[0].cpu().numpy() + 45),
            (-1.75, 8.75)
        ]
        return self.state
    

    def step_with_planner_input(self, 
                                ego_state: EgoState=None,
                                planner_input: PlannerInput = None,
                                scenario: AbstractScenario=None,
                                iteration=None):
        
        self.ego_state = ego_state
        self._current_planner_input = planner_input
        self._update_state_from_scenario(ego_state)

        # update vehicle parameters
        vehicle_parameter = ego_state.car_footprint.vehicle_parameters
        self.half_width = vehicle_parameter.half_width
        self.half_length = vehicle_parameter.half_length

        # update lane map
        self.lane_map.step_with_planner_init(ego_state, scenario=scenario, iteration=iteration)

        # update agent manager
        # self.agent_manager.step_with_planner_input(planner_input, self.lane_map)


    def _update_state_from_scenario(self, ego_state: EgoState):
        
        state = torch.zeros(size=(len(STATEINDEX),),**self.tensor_args)
        state[STATEINDEX.X] = ego_state.rear_axle.x
        state[STATEINDEX.Y] = ego_state.rear_axle.y
        state[STATEINDEX.YAW] = ego_state.rear_axle.heading
        state[STATEINDEX.Vx] = ego_state.dynamic_car_state.speed
        state[STATEINDEX.Vy] = 0
        state[STATEINDEX.YAW_RATE] = ego_state.dynamic_car_state.angular_velocity
        state[STATEINDEX.ACC_X] = ego_state.dynamic_car_state.acceleration
        state[STATEINDEX.STEERING_ANGLE] = ego_state.tire_steering_angle

        if self.use_local_coordinate:
            state[STATEINDEX.X] = 0
            state[STATEINDEX.Y] = 0
            state[STATEINDEX.YAW] = 0

            self.x0 = ego_state.rear_axle.x
            self.y0 = ego_state.rear_axle.y
            self.yaw0 = ego_state.rear_axle.heading
            

        self.state = state


    def start_visualization(self):
        if self.visualize:
            plt.ion()

        else:
            plt.ioff()

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        # self.fig2, self.ax2 = plt.subplots(figsize=(5, 5))
        self.ax.set_aspect('equal')
        self.ax.set(xlim=self.state_ranges[0])
        self.ax.set(ylim=self.state_ranges[1])

        self.cmap = "Greys"
        # artists for clearing / redrawing
        self.start_artist = None
        self.goal_artist = None
        self.cost_artist = None
        self.rollout_artist = None
        self.agent_artists = None
        self.expert_artist = None
        self.road_artists = None
        self.custom_trajectory_artist = None

        self.score_artist = None
            


    def draw_expert_trajectory(self,expert_traj):
        self.clear_artist(self.expert_artist)
        expert_artist = []
        traj = expert_traj
        expert_artist += self.ax.plot(traj[:, 0], traj[:, 1],color='r',linewidth=1.5)
        self.expert_artist = expert_artist

    def draw_custom_trajectory_with_score(self, custom_traj, score):
        self.clear_artist(self.custom_trajectory_artist)
        assert custom_traj.shape[0] == score.shape[0], "custom trajectory and score must have the same number of samples"
        custom_trajectory_artist = []
        if len(custom_traj) == 2:
            custom_traj = custom_traj.reshape(-1, 2)
        score_norm = self._normalize_visual_scores(score)
        for idx, traj_score in enumerate(score_norm):
            # 颜色按score从低到高：浅绿 -> 深绿
            cmap = plt.get_cmap('Greens')
            color_rgba = cmap(0.10 + 0.75 * traj_score)
            alpha = 0.08 + 0.82 * traj_score
            custom_trajectory_artist += self.ax.plot(
                custom_traj[idx][:, 0],
                custom_traj[idx][:, 1],
                color=color_rgba,
                linewidth=1.5,
                alpha=alpha,
            )
        self.custom_trajectory_artist = custom_trajectory_artist

    @staticmethod
    def _normalize_visual_scores(scores, lower_percentile: float = 5.0, upper_percentile: float = 95.0, gamma: float = 1.5):
        """Map arbitrary scores to [0, 1] for visualization, robust to log-scores and outliers."""
        scores = np.asarray(scores, dtype=np.float64)
        if scores.size == 0:
            return scores

        finite_mask = np.isfinite(scores)
        if not finite_mask.any():
            return np.zeros_like(scores, dtype=np.float64)

        finite_scores = scores[finite_mask]
        low = np.percentile(finite_scores, lower_percentile)
        high = np.percentile(finite_scores, upper_percentile)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low + 1e-8:
            low = np.min(finite_scores)
            high = np.max(finite_scores)

        if high <= low + 1e-8:
            normalized = np.ones_like(scores, dtype=np.float64)
            normalized[~finite_mask] = 0.0
            return normalized

        clipped = np.clip(scores, low, high)
        normalized = (clipped - low) / (high - low + 1e-8)
        normalized = np.power(normalized, gamma)
        normalized[~finite_mask] = 0.0
        return normalized

    def draw_score(self, weighted_score, aggregate_scores):
        self.clear_artist(self.score_artist)
        # weighted_score: shape [N, 3], aggregate_scores: shape [N]
        weighted_score = np.array(weighted_score)
        aggregate_scores = np.array(aggregate_scores)
        score_norm = (aggregate_scores - np.min(aggregate_scores)) / (np.ptp(aggregate_scores) + 1e-8)
        cmap = plt.get_cmap('viridis')
        colors = cmap(score_norm)

        # 画三维点
        self.ax2.clear()
        self.ax2 = self.fig2.add_subplot(111, projection='3d')
        xs = weighted_score[0, :]
        ys = weighted_score[1, :]
        zs = weighted_score[2, :]
        self.score_artist = [self.ax2.scatter(xs, ys, zs, c=colors, s=40)]
        self.ax2.set_xlabel('Progress')
        self.ax2.set_ylabel('TTC')
        self.ax2.set_zlabel('Comfort')
        plt.draw()
        plt.pause(0.001)
        
    def draw_trajectory_step(self, prev_state, cur_state, color="tab:blue"):
        if not self.visualize:
            return
        if self.trajectory_artist is None:
            self.trajectory_artist = []
        artists = self.trajectory_artist
        artists += self.ax.plot([prev_state[0].cpu(), cur_state[0].cpu()],
                                [prev_state[1].cpu(), cur_state[1].cpu()], color=color)
        self.ax.set(xlim=self.state_ranges[0])
        self.ax.set(ylim=self.state_ranges[1])
        self.ax.set(title=f"current scene: {self.scenario.token}, step: {self.current_iteration} ")

        # draw agents
        self.clear_artist(self.agent_artists)
        artists = []
        for polygon in self.agent_manager[0].polygons_for_plot():
            artists += self.ax.plot(polygon[0],polygon[1], color='b',linestyle='solid',linewidth=0.6)
            # print(f"draw agent polygon: {polygon}")
        ego_box = self.get_ego_polygon()
        artists += self.ax.plot(ego_box[:,0],ego_box[:,1], linewidth=1.0, color='red')
        # plot agent point tensor
        agent_points = self.agent_manager.agent_points[0,:].cpu().numpy()
        point_radius = self.agent_manager.point_radius.cpu().numpy()
        for point_idx in np.arange(agent_points.shape[0]):
            circle = plt.Circle((agent_points[point_idx,0],agent_points[point_idx,1]),point_radius[point_idx], ec='r',fill=True,alpha=0.2)
            artists += [self.ax.add_patch(circle)]
        self.agent_artists = artists
        # plt.draw()
        # plt.pause(0.001)

    def draw_agents(self):
        # draw agents
        self.clear_artist(self.agent_artists)
        artists = []
        for polygon in self.agent_manager[0].polygons_for_plot():
            artists += self.ax.plot(polygon[0],polygon[1], color='b',linestyle='solid',linewidth=0.6)
            # print(f"draw agent polygon: {polygon}")
        ego_box = self.get_ego_polygon()
        artists += self.ax.plot(ego_box[:,0],ego_box[:,1], linewidth=1.0, color='red')
        self.agent_artists = artists
        # plt.draw()
        # plt.pause(0.001)

    def get_ego_polygon(self):
        half_width = self.half_width
        half_length = self.half_length
        top_left    = [-half_width , +half_length]
        top_right   = [+half_width , +half_length]
        buttom_left = [-half_width , -half_length]
        buttom_right= [+half_width , -half_length]
        ego_box = np.array([top_left,top_right,buttom_right,buttom_left,top_left])
        yaw = self.state[2].cpu() + np.pi/2
        rotate_metric = np.array([[np.cos(yaw), -np.sin(yaw)],
                                  [np.sin(yaw), np.cos(yaw)]])
        ego_box_rotate = np.dot(rotate_metric, ego_box.T)
        ego_box_xy = ego_box_rotate.T + self.state[:2].cpu().numpy()
        return ego_box_xy

    def draw_road_lines(self):

        road_artists = []
        for lane in self.lane_map._lane_map_dict.values():
            if self.use_local_coordinate:
                path = lane.baseline_path.discrete_path
                discrete_lane = np.array([state.array for state in path])
                discrete_lane_local = discrete_lane - self.ego_state.center.array
                ego_yaw = -self.ego_state.center.heading
                discrete_lane_rotate = np.matmul(discrete_lane_local, 
                                                 np.array([[np.cos(ego_yaw),np.sin(ego_yaw)],
                                                           [-np.sin(ego_yaw),np.cos(ego_yaw)]]
                                                           )
                                                )
                road_artists += self.ax.plot(discrete_lane_rotate[:,0],discrete_lane_rotate[:,1], color='k', linestyle='dashed', linewidth= 1)
            else:
                line = lane.baseline_path.linestring
                x,y = line.xy
                road_artists += self.ax.plot(x,y, color='k', linestyle='dashed', linewidth= 1)
        ref_path = self.lane_map.ref_path
        road_artists += self.ax.plot(ref_path[:,0],ref_path[:,1], color='yellow', linestyle='solid', linewidth= 2,alpha = 0.8)
        lane_bound = self.lane_map.lane_boundaries
        left_bound = np.expand_dims(lane_bound[:,0],axis=1)*np.vstack((np.cos(ref_path[:,2]+np.pi/2),np.sin(ref_path[:,2]+np.pi/2))).T + ref_path[:,:2]
        right_bound = np.expand_dims(lane_bound[:,1],axis=1)*np.vstack((np.cos(ref_path[:,2]+np.pi/2),np.sin(ref_path[:,2]+np.pi/2))).T + ref_path[:,:2]
        # fill the area between left_bound and right_bound
        # try:
        #     lane_polygon_pts = np.vstack([left_bound, right_bound[::-1]])
        #     lane_patch = matplotlib.patches.Polygon(
        #     lane_polygon_pts,
        #     closed=True,
        #     facecolor='yellow',
        #     edgecolor='none',
        #     alpha=0.15,
        #     )
        #     road_artists += [self.ax.add_patch(lane_patch)]
        # except Exception as e:
        #     logger.warning(f"Failed to draw lane area: {e}")
        road_artists += self.ax.plot(left_bound[:,0],left_bound[:,1], color='yellow', linestyle='solid', linewidth= 1.5,alpha = 0.5)
        road_artists += self.ax.plot(right_bound[:,0],right_bound[:,1], color='yellow', linestyle='solid', linewidth= 1.5,alpha = 0.5)
        
        for lane in self.lane_map._route_lane_dict.values():
            if self.use_local_coordinate:
                path = lane.baseline_path.discrete_path
                discrete_lane = np.array([state.array for state in path])
                discrete_lane_local = discrete_lane - self.ego_state.center.array
                ego_yaw = -self.ego_state.center.heading
                discrete_lane_rotate = np.matmul(discrete_lane_local, 
                                                 np.array([[np.cos(ego_yaw),np.sin(ego_yaw)],
                                                           [-np.sin(ego_yaw),np.cos(ego_yaw)]]
                                                           )
                                                )
                road_artists += self.ax.plot(discrete_lane_rotate[:,0],discrete_lane_rotate[:,1], color='purple', linestyle='solid', linewidth= 1)
            else:
                line = lane.baseline_path.linestring
                x,y = line.xy
                road_artists += self.ax.plot(x,y, color='purple', linestyle='dashed', linewidth= 1)
        
        
        self.road_artists = road_artists



        plt.pause(0.001)

    def clear_trajectory(self):
        self.clear_artist(self.trajectory_artist)

    def draw_current_scene(self, 
                           chosen_trajectory=None,
                           other_trajectory=None,
                           trajectory_score=None):
        self.ax.cla()
        self.ax.set_aspect('equal')
        self.ax.set(xlim=self.state_ranges[0])
        self.ax.set(ylim=self.state_ranges[1])

        self.draw_road_lines()
        self.draw_agents()
        if other_trajectory is not None:
            self.draw_custom_trajectory_with_score(other_trajectory, trajectory_score)
        else:
            print("no other trajectory to draw")
        if chosen_trajectory is not None:
            self.draw_expert_trajectory(chosen_trajectory)
        else:
            print("no chosen trajectory to draw")
        # draw img from canvas
        self.fig.canvas.draw()
        width, height = self.fig.get_size_inches() * self.fig.get_dpi()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
            int(height), int(width), 3
        )

        
        # self.clear_artist(self.agent_artists)
        # self.clear_artist(self.road_artists)
        # self.clear_artist(self.custom_trajectory_artist)
        # self.clear_artist(self.expert_artist)
            
        # plt.close(self.fig)
        return img

    def draw_model_in_out(self,
                          scene_feature: SceneFeature,
                          chosen_trajectory = None,
                          all_trajectories = None,
                          all_trajectory_scores = None,
                          agent_prediction: AgentPrediction = None,
                          prediction_mode: str = 'prediction',
                          expert_trajectory = None,
                          rollout_subset_trajectories = None,
                          rollout_subset_scores = None,
                          rollout_anchor_indices = None,
                          rollout_teacher_sources = None,
                          expert_path_local = None,
                          expert_route_ref_path = None):
        plt.ioff()
        fig, axes = plt.subplots(figsize=(10,10))
        axes.set_aspect('equal')
        axes.set(xlim=self.state_ranges[0])
        axes.set(ylim=self.state_ranges[1])

        # load features
        road_feature = scene_feature.road_feature
        # ref_path_feature = scene_feature.ref_path_feature
        route_feature = scene_feature.route_feature
        ego_feature = scene_feature.ego_feature
        static_obstacle_feature = scene_feature.static_obstacle_feature
        agent_feature = scene_feature.agent_feature
        
        def rotate_poly(poly, yaw, center):
            """"""
            c, s = np.cos(yaw), np.sin(yaw)
            rot_mat = np.array([[c, -s], [s, c]])
            return (poly - center) @ rot_mat.T + center


        # 绘制道路中心线
        for idx, center_line in enumerate(road_feature.center_line):
            center_line = np.array(center_line)
            tl_date = road_feature.road_traffic_light[idx]
            if center_line.shape[0] > 1:
                if tl_date == 3:  # 红灯
                    axes.plot(center_line[:, 0], center_line[:, 1], color='red', linestyle='--', linewidth=1, alpha=0.7)
                else:
                    axes.plot(center_line[:, 0], center_line[:, 1], color='gray', linestyle='--', linewidth=1, alpha=0.7)
        # 绘制道路多边形
        for polygon in road_feature.road_geometry:
            polygon = np.array(polygon)
            if polygon.shape[0] > 2:
                axes.fill(polygon[:, 0], polygon[:, 1], color='lightgray', alpha=0.3, edgecolor='gray')

        # 绘制参考路径
        for route_polygon in route_feature.route_geometry:
            route_polygon = np.array(route_polygon)
            if route_polygon.shape[0] > 2:
                axes.fill(route_polygon[:, 0], route_polygon[:, 1], color='green', alpha=0.2, edgecolor='green')


        # 绘制本车多边形
        ego_pose = np.array([0,0,0])
        ego_half_width = ego_feature.ego_geometry[0]
        ego_half_length = ego_feature.ego_geometry[1]
        headings = ego_pose[2]
        cos, sin = np.cos(headings), np.sin(headings)

        # calculate ego center from rear axle
        rear_axle_to_center = 1.461
        ego_pose[:2] = np.array(
            [rear_axle_to_center * cos, rear_axle_to_center * sin]
        )

        ego_poly = np.array([
            [ego_pose[0] - ego_half_length, ego_pose[1] - ego_half_width],
            [ego_pose[0] - ego_half_length, ego_pose[1] + ego_half_width],
            [ego_pose[0] + ego_half_length, ego_pose[1] + ego_half_width],
            [ego_pose[0] + ego_half_length, ego_pose[1] - ego_half_width]
        ])
        axes.fill(ego_poly[:, 0], ego_poly[:, 1], color='blue', alpha=0.5, label='Ego Vehicle', edgecolor='blue')

        # 绘制静态障碍物多边形
        if len(static_obstacle_feature.static_obstacle_position) > 0:
            static_obj_pos = np.asarray(static_obstacle_feature.static_obstacle_position) # [N, (x,y,yaw)]
            static_obj_dim = np.asarray(static_obstacle_feature.static_object_dimension) # [N, (half_length, half_width)]
            static_obj_poly = [
            np.array([
                [static_obj_pos[i][0] - static_obj_dim[i][0], static_obj_pos[i][1] - static_obj_dim[i][1]],
                [static_obj_pos[i][0] - static_obj_dim[i][0], static_obj_pos[i][1] + static_obj_dim[i][1]],
                [static_obj_pos[i][0] + static_obj_dim[i][0], static_obj_pos[i][1] + static_obj_dim[i][1]],
                [static_obj_pos[i][0] + static_obj_dim[i][0], static_obj_pos[i][1] - static_obj_dim[i][1]]
            ])
            for i in range(len(static_obj_pos))
            ]

            static_obj_poly_rotated = []
            for i, poly in enumerate(static_obj_poly):
                yaw = static_obj_pos[i][2]  # yaw in radians
                center = static_obj_pos[i][:2]
                poly_rot = rotate_poly(poly, yaw, center)
                static_obj_poly_rotated.append(poly_rot)
            static_obj_poly = np.array(static_obj_poly_rotated)
            static_obj_poly = np.array(static_obj_poly)
            static_obj_poly = static_obj_poly.reshape(-1, 4, 2)
            # 绘制静态障碍物多边形
            for static_obj_idx, poly in enumerate(static_obj_poly):
                pose = static_obj_pos[static_obj_idx]
                if poly.shape[0] > 2:
                    if static_obj_idx == 0:
                        axes.fill(poly[:, 0], poly[:, 1], color='red', alpha=0.3, label='Static Obstacle', edgecolor='red')
                    else:
                        axes.fill(poly[:, 0], poly[:, 1], color='red', alpha=0.3, edgecolor='red')


        # 绘制周围车辆多边形与历史轨迹
        if len(agent_feature.agent_current_state) > 0:

            # print(f"Agent vehicles found: {len(agent_feature.agent_current_state)}")
            agent_current = np.asarray(agent_feature.agent_current_state)  # [N, (x,y,yaw,v,a,yaw_rate)]
            agent_type = np.asarray(agent_feature.agent_type)
            agent_geo = agent_feature.agent_geometry  # [N, (half_length, half_width)]
            agent_hist = agent_feature.agent_history_state  # [N, T, (x,y,yaw,vx,vy)]
            agent_hist_mask = agent_feature.agent_history_mask  # [N, T]
            agent_poly = [
                np.array([
                [agent_current[i][0] - agent_geo[i][0], agent_current[i][1] - agent_geo[i][1]],
                [agent_current[i][0] - agent_geo[i][0], agent_current[i][1] + agent_geo[i][1]],
                [agent_current[i][0] + agent_geo[i][0], agent_current[i][1] + agent_geo[i][1]],
                [agent_current[i][0] + agent_geo[i][0], agent_current[i][1] - agent_geo[i][1]]
            ])
            for i in range(len(agent_current))
            ]
            agent_poly_rotated = []
            for i, poly in enumerate(agent_poly):
                yaw = agent_current[i][2]  # yaw in radians
                center = agent_current[i][:2]
                poly_rot = rotate_poly(poly, yaw, center)
                agent_poly_rotated.append(poly_rot)
            agent_poly = np.array(agent_poly_rotated)
            agent_poly = np.array(agent_poly)
            agent_poly = agent_poly.reshape(-1, 4, 2)
            # 绘制周围车辆多边形与历史轨迹
            for i, _ in enumerate(agent_current):
                pose = agent_current[i]
                poly = agent_poly[i]
                if poly.shape[0] > 2:
                    if i == 0:
                        axes.fill(poly[:, 0], poly[:, 1], color='orange', alpha=0.3, label='Agent Vehicle', edgecolor='orange')
                    else:
                        axes.fill(poly[:, 0], poly[:, 1], color='orange', alpha=0.3, edgecolor='orange')
                    # 绘制历史轨迹（去除padding，并用不同颜色区分）
                    history = agent_hist[i]
                    hist_mask = agent_hist_mask[i]
                    # 只保留有效的历史轨迹点
                    valid_idx = np.where(hist_mask)[0]
                    if len(valid_idx) > 1:
                        valid_history = history[valid_idx]
                        axes.plot(valid_history[:, 0], valid_history[:, 1], color='purple', linestyle='--', linewidth=2, alpha=0.5)
            if agent_prediction is not None and prediction_mode == 'prediction':
            
            # 绘制预测轨迹
                agent_future = np.array(agent_prediction.agent_future_state)  # [N, T, (x,y,yaw,vx,vy)]
                agent_future_mask = np.array(agent_prediction.agent_future_mask)
                # 去掉padding的future轨迹
                for i, future in enumerate(agent_future):
                    mask = agent_future_mask[i]
                    valid_idx = np.where(mask)[0]
                    if len(valid_idx) > 1:
                        valid_future = future[valid_idx]
                        axes.plot(valid_future[:, 0], valid_future[:, 1], color='orange', linestyle='-', linewidth=2, alpha=0.6)
            elif prediction_mode == 'CV' or prediction_mode == 'CA' or prediction_mode == 'CYAW':
                # 绘制基于当前速度的匀速直线预测轨迹//基于当前加速度的匀加速直线预测轨迹//基于当前偏航率的匀偏航率圆弧预测轨迹
                for i, current in enumerate(agent_current):
                    x, y = current[0], current[1]
                    yaw = current[2]
                    speed = current[3]
                    acceleration = current[4]
                    yaw_rate = current[5]
                    if agent_type[i] == 2:
                        yaw_rate = 0.0
                    num_future_steps = 20
                    dt = 0.2
                    future_positions = []
                    for t in range(1, num_future_steps + 1):
                        x = x + speed * np.cos(yaw) * dt
                        y = y + speed * np.sin(yaw) * dt
                        if prediction_mode == 'CA' or prediction_mode == 'CYAW':
                            speed = np.clip(speed + acceleration * dt, 0, None)
                        if prediction_mode == 'CYAW':
                            yaw = yaw + yaw_rate * dt
                        future_positions.append([x, y])
                    future_positions = np.array(future_positions)
                    axes.plot(future_positions[:, 0], future_positions[:, 1], color='orange', linestyle=':', linewidth=2, alpha=0.6)
            
        # 绘制参考路径
        # ref_path = ref_path_feature.numpy()
        # lane_bound = ref_path[:,3:5]
        # left_bound = np.expand_dims(lane_bound[:,0],axis=1)*np.vstack((np.cos(ref_path[:,2]+np.pi/2),np.sin(ref_path[:,2]+np.pi/2))).T + ref_path[:,:2]
        # right_bound = np.expand_dims(lane_bound[:,1],axis=1)*np.vstack((np.cos(ref_path[:,2]+np.pi/2),np.sin(ref_path[:,2]+np.pi/2))).T + ref_path[:,:2]
        # axes.plot(ref_path[:, 0], ref_path[:, 1], color='green', linestyle='-', linewidth=2, alpha=0.5)
        # # print(f'left_bound.shape: {left_bound.shape}')
        # # 绘制参考路径的左右边界多边形
        # ref_path_poly = np.vstack([left_bound, right_bound[::-1]])
        # axes.fill(ref_path_poly[:, 0], ref_path_poly[:, 1], color='green', alpha=0.2)
        # # ax.fill_between(ref_path[:, 0], left_bound, right_bound, color='green', alpha=0.2, label='Reference Path Boundary')

        # 绘制所有候选轨迹
        if all_trajectories is not None and all_trajectory_scores is not None:
            # print(f"All candidate trajectories found: {len(all_trajectories)}")
            # print(f"All candidate trajectory scores: {all_trajectory_scores}")
            # print(f"All candidate trajectory : {all_trajectories}")
            score_norm = self._normalize_visual_scores(all_trajectory_scores)
            for idx, traj_score in enumerate(score_norm):
                # 颜色按score从低到高：浅蓝 -> 深蓝
                cmap = plt.get_cmap('Blues')
                color_rgba = cmap(0.10 + 0.75 * traj_score)
                alpha = 0.06 + 0.88 * traj_score
                axes.plot(
                    all_trajectories[idx][:, 0],
                    all_trajectories[idx][:, 1],
                    color=color_rgba,
                    linewidth=1.5,
                    alpha=alpha,
                    zorder=2,
                    label='Candidate Trajectory' if idx==0 else None,
                )

        if rollout_subset_trajectories is not None and rollout_subset_scores is not None:
            rollout_subset_trajectories = np.asarray(rollout_subset_trajectories, dtype=np.float32)
            rollout_subset_scores = np.asarray(rollout_subset_scores, dtype=np.float32).reshape(-1)
            rollout_anchor_indices = None if rollout_anchor_indices is None else np.asarray(rollout_anchor_indices, dtype=np.int32).reshape(-1)
            rollout_teacher_sources = None if rollout_teacher_sources is None else np.asarray(rollout_teacher_sources, dtype=np.int32).reshape(-1)
            score_norm = self._normalize_visual_scores(rollout_subset_scores)
            best_rollout_idx = int(np.argmax(rollout_subset_scores)) if rollout_subset_scores.size > 0 else -1
            teacher_styles = {
                0: {'color': '#7f1d1d', 'linestyle': '-', 'label': 'Expert Teacher'},
                1: {'color': '#1f4e5f', 'linestyle': '-', 'label': 'Expert Route Teacher'},
                2: {'color': '#5b4b1a', 'linestyle': '-', 'label': 'Policy Teacher'},
                3: {'color': '#5c2a72', 'linestyle': '-', 'label': 'Chosen Policy Anchor'},
                10: {'color': '#b45309', 'linestyle': '-', 'label': 'Expert Teacher Cruise'},
                11: {'color': '#0f766e', 'linestyle': '-', 'label': 'Expert Route Cruise'},
                12: {'color': '#6b7280', 'linestyle': '-', 'label': 'Policy Teacher Cruise'},
                13: {'color': '#7c3aed', 'linestyle': '-', 'label': 'Chosen Policy Cruise'},
                20: {'color': '#ea580c', 'linestyle': '-', 'label': 'Expert Teacher Mild Accel'},
                21: {'color': '#0891b2', 'linestyle': '-', 'label': 'Expert Route Mild Accel'},
                22: {'color': '#4b5563', 'linestyle': '-', 'label': 'Policy Teacher Mild Accel'},
                23: {'color': '#9333ea', 'linestyle': '-', 'label': 'Chosen Policy Mild Accel'},
            }
            used_labels = set()
            for idx, traj_score in enumerate(score_norm):
                teacher_source = -1 if rollout_teacher_sources is None or idx >= rollout_teacher_sources.shape[0] else int(rollout_teacher_sources[idx])
                style = teacher_styles.get(teacher_source, {'color': '#3f3f46', 'linestyle': '-', 'label': 'Rollout Anchor'})
                alpha = 0.28 + 0.44 * traj_score
                trajectory = rollout_subset_trajectories[idx]
                is_best_rollout = idx == best_rollout_idx
                label = None if style['label'] in used_labels else style['label']
                used_labels.add(style['label'])
                axes.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    color='#111827' if is_best_rollout else style['color'],
                    linewidth=2.2 if is_best_rollout else 1.35,
                    linestyle=(0, (8, 4)) if is_best_rollout else style['linestyle'],
                    alpha=1.0 if is_best_rollout else alpha,
                    zorder=3.7 if is_best_rollout else 2.4,
                    label='Best Rollout Anchor' if is_best_rollout else label,
                )
                if trajectory.shape[0] > 0:
                    end_x, end_y = trajectory[-1, 0], trajectory[-1, 1]
                    source_text = style['label'].replace(' Teacher', '').replace(' Anchor', '').replace(' Rollout', '')
                    axes.text(
                        end_x,
                        end_y,
                        source_text,
                        color='#111827' if is_best_rollout else style['color'],
                        fontsize=6.5,
                        alpha=0.9 if is_best_rollout else min(alpha + 0.15, 0.9),
                        zorder=4.0 if is_best_rollout else 3.0,
                    )
                    if is_best_rollout:
                        axes.scatter(
                            [end_x],
                            [end_y],
                            s=52,
                            color='#111827',
                            marker='D',
                            edgecolors='#f9fafb',
                            linewidths=0.9,
                            zorder=3.9,
                        )

        if expert_path_local is not None:
            expert_path_local = np.asarray(expert_path_local, dtype=np.float32)
            if expert_path_local.shape[0] > 1:
                axes.plot(
                    expert_path_local[:, 0],
                    expert_path_local[:, 1],
                    color='#6b5b2a',
                    linewidth=0.95,
                    linestyle='-',
                    alpha=0.95,
                    zorder=3.2,
                    label='Expert Path' if expert_trajectory is None else None,
                )

        if expert_route_ref_path is not None:
            expert_route_ref_path = np.asarray(expert_route_ref_path, dtype=np.float32)
            if expert_route_ref_path.shape[0] > 1:
                axes.plot(
                    expert_route_ref_path[:, 0],
                    expert_route_ref_path[:, 1],
                    color='#355c7d',
                    linewidth=0.95,
                    linestyle='-',
                    alpha=0.95,
                    zorder=3.1,
                    label='Expert Route Ref Path',
                )

        # 绘制专家轨迹
        if expert_trajectory is not None:
            axes.plot(expert_trajectory[:, 0], expert_trajectory[:, 1], color='orange', linewidth=2, alpha=0.85, zorder=4.0, label='Expert Trajectory')

        # 绘制选中轨迹
        if chosen_trajectory is not None:
            axes.plot(chosen_trajectory[:, 0], chosen_trajectory[:, 1], color='#dc2626', linewidth=2.8, alpha=0.95, zorder=4.4, label='Chosen Trajectory')

        handles, labels = axes.get_legend_handles_labels()
        if labels:
            dedup = dict(zip(labels, handles))
            axes.legend(dedup.values(), dedup.keys(), loc='upper right')
        
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
            int(height), int(width), 3
        )
        plt.close(fig)
        return img

    @staticmethod
    def clear_artist(artist):
        """
        Safely remove matplotlib artists.
        Handles None, single artist, lists/tuples and nested lists.
        Ignores errors when an artist is already removed.
        """
        if artist is None:
            return

        def _safe_remove(a):
            try:
                if a is None:
                    return
                # some entries might themselves be lists of artists
                if isinstance(a, (list, tuple)):
                    for it in a:
                        _safe_remove(it)
                    return
                # try to call remove(), ignore if not present
                a.remove()
            except Exception:
                # ignore ValueError from artist already removed and other remove errors
                return

        # top-level may be a single artist or an iterable
        if isinstance(artist, (list, tuple)):
            for a in artist:
                _safe_remove(a)
        else:
            _safe_remove(artist)

def is_sequence(obj):
    if isinstance(obj, str):
        return False
    return isinstance(obj, collections.abc.Sequence)