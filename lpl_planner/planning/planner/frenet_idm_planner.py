from typing import List, Optional, Tuple

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from lpl_planner.planning.scene.scene_feature.features import SceneFeature, AgentPrediction
from lpl_planner.planning.scene.map.occupancy_map import OccupancyMap
from lpl_planner.planning.scene.scene_manager import SceneManager
from lpl_planner.planning.planner.frenet_utils import (
    QuinticPolynomial,
)

from shapely import LineString, Polygon, creation, Point

def normalize_angle(angle):
    """
    Map a angle in range [-π, π]
    :param angle: any angle as float
    :return: normalized angle
    """
    return np.arctan2(np.sin(angle), np.cos(angle))

class FrenetIDMPlanner:
    """Frenet + IDM 轨迹生成器。

    横向：基于 frenet 生成 1~3 个横向方案（保持车道 + 左/右换道）。
    纵向：基于 IDM + 期望速度（0.2/0.4/0.6/0.8/1.0 * 限速）生成 5 个纵向方案。
    最终组合得到 5~15 条全局轨迹。
    """

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        max_acceleration: float = 2.4,
        max_deceleration: float = -4.0,
        debug: bool = False,
    ) -> None:
        super().__init__()

        # planner parameters
        self.trajectory_sampling = trajectory_sampling
        self.time_step = trajectory_sampling.interval_length
        if debug:
            print(f"FrenetIDMPlanner time_step: {self.time_step}")
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self.speed_factors = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)
        self.debug = debug

        self._future_collision_map: List[OccupancyMap] = []
        self.agent_position: np.ndarray = np.array([])
        self.static_obstacle_position: np.ndarray = np.array([])

    def _update_future_collision_map(
        self,
        scene_feature: SceneFeature,
        agent_prediction_gt: Optional[AgentPrediction] = None,
        agent_prediction: Optional[AgentPrediction] = None,
        prediction_mode: str = "CYAW",
        longitudinal_buffer: float = 0.1,
        lateral_buffer: float = 0.1,
    ) -> None:
        """
        Docstring for _update_future_collision_map
        
        :param self: Description
        :param scene_feature: Description
        :type scene_feature: SceneFeature
        """
        # 2. extract static obstacles' polygons
        static_obstacle_polygons = []
        static_obstacle_tokens = []
        static_obstacle_feature = scene_feature.static_obstacle_feature
        self.static_obstacle_position = np.array(static_obstacle_feature.static_obstacle_position)
        if len(static_obstacle_feature.static_obstacle_position) > 0:
            static_obstacle_geo = np.array(static_obstacle_feature.static_object_dimension)
            static_obstacle_position = np.array(static_obstacle_feature.static_obstacle_position)
            hx, hy = static_obstacle_geo[:, 0] , static_obstacle_geo[:, 1] 
            offs = np.stack([
                np.stack([-hx, -hy], axis=-1),
                np.stack([-hx,  hy], axis=-1),
                np.stack([ hx,  hy], axis=-1),
                np.stack([ hx, -hy], axis=-1),
            ], axis=1) 
            cosyaw = np.cos(static_obstacle_position[..., 2])[:, None]                                              # [N,1]
            sinyaw = np.sin(static_obstacle_position[..., 2])[:, None]
            x = offs[..., 0]; y = offs[..., 1]          # [N,4]
            xr = cosyaw * x - sinyaw * y
            yr = sinyaw * x + cosyaw * y
            offs = np.stack([xr, yr], axis=-1) 
            obs_poly = offs + static_obstacle_position[:, None, :2]  # [N,4,(x,y)]
            obs_polygons = creation.polygons(obs_poly)
            static_obstacle_polygons.extend(obs_polygons)
            static_obstacle_tokens.extend([f"obstacle_{i}" for i in range(len(obs_polygons))])
        
        # 2.5. construct red-light polygons as additional static obstacles (only on-route lanes)
        road_feature = scene_feature.road_feature
        road_polygon = np.array(road_feature.road_geometry)
        tl_status = np.array(road_feature.road_traffic_light)
        centerlines = np.array(road_feature.center_line)

        route_feature = scene_feature.route_feature
        route_polygon = np.array(route_feature.route_geometry)
        route_tokens = [f"route_roadblock_{i}" for i in range(len(route_polygon))]
        route_polygons = [Polygon(polygon) for polygon in route_polygon]
        route_map = OccupancyMap(route_tokens, route_polygons)

        red_light_polygons: List[Polygon] = []
        red_light_tokens: List[str] = []
        for polygon, t_stat, centerline in zip(road_polygon, tl_status, centerlines):
            # t_stat == 3 is used as red-light in scene_scorer
            if t_stat == 3:
                red_light_poly = Polygon(polygon)
                # only consider red lights that belong to on-route lanes
                in_route = route_map.points_in_polygons(centerline)
                if np.all(in_route.any(axis=-1), axis=-1):
                    red_light_polygons.append(red_light_poly)
                    red_light_tokens.append(f"red_light_lane_{len(red_light_tokens)}")

        # decide whether to use red-light constraints: skip if ego is already inside a red-light area
        use_red_light = True
        if len(red_light_polygons) > 0:
            ego_state_arr = np.asarray(scene_feature.ego_feature["ego_current_state"], dtype=np.float64)
            ego_geom = np.asarray(scene_feature.ego_feature["ego_geometry"], dtype=np.float64)
            half_width, half_length, rear_axle_to_center = ego_geom

            ego_x, ego_y, ego_yaw = ego_state_arr[:3]
            cosyaw = np.cos(ego_yaw)
            sinyaw = np.sin(ego_yaw)

            # ego center from rear-axle pose
            center_x = ego_x + rear_axle_to_center * cosyaw
            center_y = ego_y + rear_axle_to_center * sinyaw

            hx, hy = half_length, half_width
            offs = np.stack([
                np.stack([-hx, -hy], axis=-1),
                np.stack([-hx,  hy], axis=-1),
                np.stack([ hx,  hy], axis=-1),
                np.stack([ hx, -hy], axis=-1),
            ], axis=0)
            ego_corners_x = offs[..., 0]
            ego_corners_y = offs[..., 1]
            xr = cosyaw * ego_corners_x - sinyaw * ego_corners_y
            yr = sinyaw * ego_corners_x + cosyaw * ego_corners_y
            offs = np.stack([xr, yr], axis=-1)
            ego_polygon_points = offs + np.array([center_x, center_y])[None, :]
            ego_polygon = Polygon(ego_polygon_points)

            # for red_poly in red_light_polygons:
            #     if red_poly.intersects(ego_polygon):
            #         use_red_light = False
            #         break
        else:
            use_red_light = False
            

        # 3. construct future collision map
        self.agent_position = np.array(scene_feature.agent_feature.agent_current_state)
        self._future_collision_map = []
        future_collision_map_len = self.trajectory_sampling.num_poses + 1  # including t=0

        # Interpolate missing (all-zero) future agent states in agent_prediction_gt
        num_pred_agents = None
        agent_feature = scene_feature.agent_feature
        agent_current_state = np.asarray(agent_feature.agent_current_state, dtype=np.float64)
        num_agents = len(agent_current_state)
        
        for time_idx in range(future_collision_map_len):  
            polygons, tokens = [], []

            # draw surrounding agents' polygons
            
            num_agents = len(agent_feature.agent_current_state) if num_pred_agents is None else min(num_pred_agents, len(agent_feature.agent_current_state))
            constant_velocity_buffer_time = 0  # number of time steps to use constant velocity 
            valid_agent_idx = np.arange(num_agents)  # initialize valid agent indices
            agent_type = np.array(agent_feature.agent_type)[:num_agents]
            pedestrain_buffer = 0.5 * (agent_type == 2)  # [m]
            if num_agents > 0:
                # 获取当前时刻的 agent 位姿，确保为 float64 ndarray
                if time_idx == 0:
                    agent_current = np.asarray(agent_feature.agent_current_state, dtype=np.float64)[:num_agents, :3]
                    agent_previous = np.asarray(agent_feature.agent_current_state, dtype=np.float64)[:num_agents, :]
                elif agent_prediction_gt is not None and prediction_mode == "prediction":
                    agent_future_states = np.asarray(agent_prediction_gt.agent_future_state, dtype=np.float64)  # [N, T, state_size]
                    agent_current = agent_future_states[:num_agents, time_idx, :3]  # [N, (x,y,yaw)]
                    agent_future_mask = np.asarray(agent_prediction_gt.agent_future_mask, dtype=bool)  # [N, T]
                    agent_current_mask = agent_future_mask[:num_agents, time_idx]  # [N,]
                    valid_agent_idx = valid_agent_idx[agent_current_mask]
                    agent_current = agent_current[valid_agent_idx]
                elif agent_prediction is not None and prediction_mode == "prediction":
                    # assume prediction at same frequency as proposal sampling
                    agent_future_states = np.asarray(agent_prediction.agent_future_state, dtype=np.float64)  # [N, T, state_size]
                    fut_idx = int(time_idx-1)
                    fut_idx = np.clip(fut_idx, 0, agent_future_states.shape[1]-1)
                    agent_current = np.asarray(agent_future_states, dtype=np.float64)[:num_agents, fut_idx, :3]
                    agent_stop_idx = np.where(np.asarray(agent_feature.agent_current_state, dtype=np.float64)[:num_agents, 3] < 0.5)[0]
                    agent_current[agent_stop_idx, :3] = np.asarray(agent_feature.agent_current_state, dtype=np.float64)[agent_stop_idx, :3]  # for stopped agents, keep them at initial position
                    # valid_agent_idx = np.where(np.all(np.abs(agent_current) > 2e-1, axis=1))[0]
                    # agent_current = agent_current[valid_agent_idx]
                else:
                    # propagate using constant yawrate + acceleration model
                    agent_previous_pose = agent_previous[:, :3]
                    agent_current = agent_previous.copy()
                    agent_speed = agent_previous[:, 3]
                    agent_acceleration = agent_previous[:, 4]
                    agent_yawrate = agent_previous[:, 5]    
                    effective_yawrate = agent_yawrate.copy()
                    effective_yawrate[agent_type == 2] = 0.0
                    agent_current[:, :2] = agent_previous_pose[:, :2] + np.stack([
                        agent_speed * np.cos(agent_previous_pose[:, 2]),
                        agent_speed * np.sin(agent_previous_pose[:, 2])
                    ], axis=-1) * self.trajectory_sampling.interval_length

                    if prediction_mode == "CA" or prediction_mode == "CYAW":
                        agent_current[:, 3] = np.clip(agent_speed + agent_acceleration * self.trajectory_sampling.interval_length, 0.0, 15.0)  # no reverse
                    if prediction_mode == "CYAW":
                        agent_current[:, 2] = agent_previous_pose[:, 2] + effective_yawrate * self.trajectory_sampling.interval_length
                    
                    agent_previous = agent_current.copy()

                # construct agent polygons
                agent_geo = np.asarray(agent_feature.agent_geometry, dtype=np.float64)[valid_agent_idx]  # [N,2] -> (hx, hy)
                # agent_longtitudial_buffer = longitudinal_buffer + time_idx * self._proposal_sampling.interval_length * longitudinal_velocity_buffer * np.linalg.norm(agent_current[:, :2], axis=1)  # [N,]
                agent_longitudial_buffer = longitudinal_buffer
                hx, hy = agent_geo[:, 0] + agent_longitudial_buffer + pedestrain_buffer, agent_geo[:, 1] + lateral_buffer + pedestrain_buffer
                offs = np.stack([
                     np.stack([-hx, -hy], axis=-1),
                     np.stack([-hx,  hy], axis=-1),
                     np.stack([ hx,  hy], axis=-1),
                     np.stack([ hx, -hy], axis=-1),
                 ], axis=1) 
                cosyaw = np.cos(agent_current[..., 2])[:, None]                          # [N,1]
                sinyaw = np.sin(agent_current[..., 2])[:, None]
                x = offs[..., 0]; y = offs[..., 1]                                    # [N,4]
                xr = cosyaw * x - sinyaw * y
                yr = sinyaw * x + cosyaw * y
                offs = np.stack([xr, yr], axis=-1) 
                agent_poly = offs + agent_current[:, None, :2]                          # [N,4,(x,y)]
                agent_polygons = creation.polygons(agent_poly)                           # Vectorized shapely Polygons
                polygons.extend(list(agent_polygons))
                tokens.extend([f"agent_{i}" for i in range(len(agent_polygons))])    
                    
            polygons.extend(static_obstacle_polygons)
            tokens.extend(static_obstacle_tokens)
            if use_red_light:
                polygons.extend(red_light_polygons)
                tokens.extend(red_light_tokens)

            self._future_collision_map.append(OccupancyMap(tokens, polygons))


    def _generate_path_plans(
        self,
        trajectories: np.ndarray,
        scenario_manager: SceneManager,
        ego_state: EgoState,
    ) -> List[LineString]:
        """
        Docstring for _generate_frenet_path_plans
        
        :param self: Description
        :param scene_feature: Description
        :type scene_feature: SceneFeature
        :return: Description
        :rtype: List[LineString]
        """

        path_plans: List[LineString] = []

        # 1. refpath as base path
        current_ego = np.array([0.0, 0.0, 0.0])
        current_sd = scenario_manager.lane_map.cartesian_to_frenet(
            points=current_ego.reshape(1, 3)
        )[0]  # (2,)
        current_s = current_sd[0]
        current_d = current_sd[1] if np.abs(current_sd[1]) > 0.5 else 0.0
        ref_path_max_s = scenario_manager.lane_map.frenet_path_util.cumulative_s[-1]
        ref_path_frenet_s = np.arange(current_s, ref_path_max_s + 0.1, 0.1)
        ref_path_cartesian = scenario_manager.lane_map.frenet_to_cartesian(
            frenet_points=np.stack((ref_path_frenet_s, np.ones_like(ref_path_frenet_s)*current_d), axis=1)
        )[:, :2]
        ref_path_ls = LineString(ref_path_cartesian)
        path_plans.append(ref_path_ls)

        # 2. other path plans from trajectories
        for traj_idx in range(trajectories.shape[0]):
            traj = trajectories[traj_idx]  # (T, 3)
            traj_ls = LineString(traj[:, :2])
            # skip duplicate paths
            if traj_ls.equals(ref_path_ls):
                continue
            # skip stop paths
            if traj_ls.length < 1.0:
                continue

            path_plans.append(traj_ls)

        # 3. lane_keep 和 lane_change 的 Frenet path（带速度相关的换道时间）

        # 3.0 当前纵向速度（近似 s_dot）
        v_long = float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x)
        if v_long < 0.5:
            v_long = 0.5  # 避免几乎静止导致 s 变化太小

        T_total = float(self.trajectory_sampling.time_horizon)
        dt = float(self.trajectory_sampling.interval_length)
        ts = np.arange(0.0, T_total + 1e-6, dt)

        def _compute_lc_duration(delta_d: float, v_s: float) -> float:
            """
            根据横向位移和当前车速计算换道时间窗：
            - 以最大横向速度 v_lat_max 为基础控制
            - 车速越快，时间窗适当放大，避免横向速度过大
            """
            v_lat_max = 1.0  # [m/s] 允许的最大横向速度
            base_T = abs(delta_d) / max(v_lat_max, 1e-2)
            speed_factor = 1.0 + max(0.0, v_s - 5.0) / 20.0  # 高速时放大时间窗
            T = base_T * speed_factor
            return float(np.clip(T, 2.0, 6.0))

        # 3.1 计算 lane_keep 目标 d（参照 RuleBasePlanner 逻辑）
        d_options = scenario_manager.lane_map.get_nearest_lane_distances(position_sd=current_sd)
        if not d_options:
            d_options = [current_d]  # stay if no lanes
        d_options_arr = np.array(d_options, dtype=float)
        nearest_d_idx = int(np.argmin(np.abs(d_options_arr - current_d)))
        keep_d = float(d_options_arr[nearest_d_idx] if np.abs(current_d) > 1.0 else 0.0)

        # 3.2 构造 lane keep path：从 current_d 平滑切到 keep_d，然后保持 keep_d
        if np.abs(keep_d - current_d) > 0.1:
            T_keep = _compute_lc_duration(keep_d - current_d, v_long)
            lat_qp_keep = QuinticPolynomial(
                current_d, 0.0, 0.0,
                keep_d, 0.0, 0.0,
                T_keep,
            )
            t_eff = np.clip(ts, 0.0, T_keep)
            d_traj = np.array([lat_qp_keep.calc_point(ti) for ti in t_eff], dtype=float)
        else:
            d_traj = np.ones_like(ts, dtype=float) * keep_d

        s_traj = current_s + v_long * ts
        valid_mask = s_traj <= ref_path_max_s
        s_keep = s_traj[valid_mask]
        d_keep = d_traj[valid_mask]

        if len(s_keep) > 1:
            sd_keep = np.stack((s_keep, d_keep), axis=1)
            keep_xy = scenario_manager.lane_map.frenet_to_cartesian(sd_keep)[:, :2]
            lane_keep_ls = LineString(keep_xy)
            # 避免和 ref_path 完全重复
            if not lane_keep_ls.equals(ref_path_ls) and lane_keep_ls.length > 1.0:
                path_plans.append(lane_keep_ls)

        # 3.3 lane change 目标 d（参照 RuleBasePlanner）
        lane_change_ds: List[float] = []
        if nearest_d_idx - 1 >= 0:
            valid_indice = np.abs(d_options_arr[0: nearest_d_idx] - keep_d) > 1.0
            if np.any(valid_indice):
                valid_left_lc_d = d_options_arr[0: nearest_d_idx][valid_indice]
                lane_change_ds.append(float(valid_left_lc_d[-1]))  # left
        if nearest_d_idx + 1 < len(d_options_arr):
            valid_indice = np.abs(d_options_arr[nearest_d_idx + 1:] - keep_d) > 1.0
            if np.any(valid_indice):
                valid_right_lc_d = d_options_arr[nearest_d_idx + 1:][valid_indice]
                lane_change_ds.append(float(valid_right_lc_d[0]))  # right

        # 3.4 为每个 lane_change 目标生成 Frenet 换道 path
        for target_d in lane_change_ds:
            if np.abs(target_d - current_d) < 0.5:
                continue
            T_lc = _compute_lc_duration(target_d - current_d, v_long)
            lat_qp_lc = QuinticPolynomial(
                current_d, 0.0, 0.0,
                target_d, 0.0, 0.0,
                T_lc,
            )
            t_eff = np.clip(ts, 0.0, T_lc)
            d_traj_lc = np.array([lat_qp_lc.calc_point(ti) for ti in t_eff], dtype=float)
            s_traj_lc = current_s + v_long * ts

            valid_mask = s_traj_lc <= ref_path_max_s
            s_lc = s_traj_lc[valid_mask]
            d_lc = d_traj_lc[valid_mask]

            if len(s_lc) <= 1:
                continue

            sd_lc = np.stack((s_lc, d_lc), axis=1)
            lc_xy = scenario_manager.lane_map.frenet_to_cartesian(sd_lc)[:, :2]
            lc_ls = LineString(lc_xy)

            # 去掉过短或与已有 path 完全重复的换道 path
            if lc_ls.length < 1.0:
                continue
            is_duplicate = any(lc_ls.equals(p) for p in path_plans)
            if not is_duplicate:
                path_plans.append(lc_ls)

        return path_plans
    
    def _get_leading_vehicle_on_path(
        self,
        path_plan: LineString,
        time_idx: int,
        ego_state: EgoState,
        ego_trajectory: np.ndarray,
    ) -> Optional[str]:
        """
        Docstring for _get_leading_vehicle_on_path
        
        :param self: Description
        :param path_plan: Description
        :type path_plan: LineString
        :return: Description
        :rtype: Optional[str]
        """

        buffed_path = path_plan.buffer(1.2, cap_style=2)  # 1.2m buffer

        intersecting_obj = self._future_collision_map[time_idx].intersects(buffed_path)
        path_length = path_plan.length
        leading_vehicle_progress = path_length
        leading_vehicle_speed = 0.0
        if len(intersecting_obj) == 0:
            return 100, 0.0  # no leading vehicle
        else:
            ego_prev_pos = ego_trajectory[time_idx-1, :2]
            ego_heading = ego_trajectory[time_idx-1, 2]
            cos, sin = np.cos(ego_heading), np.sin(ego_heading)
            # transform ego from rear axle position
            ego_center = ego_prev_pos + np.array([1.46 * cos, 1.46 * sin])  # 1.46m to center
            hx, hy = 2.59, 1.15 # half length and half width
            offs = np.stack([
                     np.stack([-hx, -hy], axis=-1),
                     np.stack([-hx,  hy], axis=-1),
                     np.stack([ hx,  hy], axis=-1),
                     np.stack([ hx, -hy], axis=-1),
                 ], axis=0) 
            ego_corners_x = offs[..., 0]; ego_corners_y = offs[..., 1]                                    # [4]
            xr = cos * ego_corners_x - sin * ego_corners_y
            yr = sin * ego_corners_x + cos * ego_corners_y
            offs = np.stack([xr, yr], axis=-1) 
            ego_polygon_points = offs + ego_center[None, :2]                          # [4,(x,y)]
            ego_polygon = Polygon(ego_polygon_points)

            ego_progress = float(path_plan.project(Point(ego_center[0], ego_center[1])))

            for token in intersecting_obj:
                obj_polygon = self._future_collision_map[time_idx][token]
                relative_rear_distance = obj_polygon.exterior.distance(ego_polygon)
                obj_progress = float(path_plan.project(Point(obj_polygon.centroid.x, obj_polygon.centroid.y)))
                
                if obj_progress < ego_progress:
                    # behind ego
                    continue

                if relative_rear_distance < leading_vehicle_progress:
                    leading_vehicle_progress = relative_rear_distance
                    if "agent" in token:
                        agent_idx = int(token.split('_')[1])
                        agent_speed = self.agent_position[agent_idx][3]
                        agent_heading = self.agent_position[agent_idx][2]
                        relative_heading = normalize_angle(agent_heading - ego_heading)
                        projected_speed = agent_speed * np.cos(relative_heading)
                        leading_vehicle_speed = projected_speed
                    else:
                        leading_vehicle_speed = 0.0  # static obstacle
        
        return leading_vehicle_progress, leading_vehicle_speed  
                    

    def _idm_step(
        self,
        current_speed: float,
        leading_vehicle_speed: float,
        leading_vehicle_progress: float,
        desired_speed: float = 15.0,
        min_spacing: float = 2.0,
        desired_time_headway: float = 1.5,
        acceleration: float = 1.5,
        comfortable_deceleration: float = 3.0,
        delta: float = 10.0,
        prev_acc: float = 0.0,
        max_jerk: float = 4.0,
    ) -> Tuple[float, float]:
        """
        Docstring for _idm_step
        
        :param self: Description
        :param current_speed: Description
        :type current_speed: float
        :param leading_vehicle_speed: Description
        :type leading_vehicle_speed: float
        :param leading_vehicle_progress: Description
        :type leading_vehicle_progress: float
        :return: Description
        :rtype: Tuple[float, float]
        """

        # compute desired gap
        speed_diff = current_speed - leading_vehicle_speed
        desired_gap = min_spacing + max(0.0, 
                                        current_speed * desired_time_headway + (current_speed * speed_diff) / (2.0 * np.sqrt(acceleration * comfortable_deceleration)))
        current_gap = max(leading_vehicle_progress, min_spacing) # avoid zero or negative gap

        # compute acceleration
        acc_raw = acceleration * (
            1
            - (current_speed / desired_speed) ** delta 
            - (desired_gap / current_gap) ** 2
            )

        # limit jerk
        max_da = max_jerk * self.time_step
        da = np.clip(acc_raw - prev_acc, -max_da, max_da)
        acc = prev_acc + da

        # clip acceleration
        acc = np.clip(acc, self.max_deceleration, self.max_acceleration)

        # update speed
        new_speed = current_speed + acc * self.time_step
        new_speed = max(0.0, new_speed)  # no reverse

        return new_speed, acc
    
    def _get_speed_limit(
        self,
        path_xy: np.ndarray,
        speed_limit: np.ndarray,
    ) -> float:
        """
        Docstring for _get_speed_limit_with_xy
        
        :param self: Description
        :param path_xy: Description
        :type path_xy: np.ndarray
        :param speed_limit: Description
        :type speed_limit: np.ndarray
        :return: Description
        :rtype: float
        """

        position = np.array([0.0, 0.0]) # ego at origin in frenet frame
        dists = np.linalg.norm(path_xy - position[None, :], axis=1)  # [P,]
        current_idx = np.argmin(dists)
        speed_limit_forward = speed_limit[current_idx:]

        forward_path_s = np.cumsum(
            np.concatenate(
                ([0.0], np.linalg.norm(np.diff(path_xy[current_idx:], axis=0), axis=1))
            )
        )  # [P',], P' <= P
        # 以80m内的最低限速作为限速
        max_distance = 80.0
        within_80m = forward_path_s <= max_distance
        closest_80m_idx = np.argmin(np.abs(forward_path_s - max_distance))
        path_xy_80m = min(current_idx + closest_80m_idx + 1, len(path_xy) - 1)
        base_speed_limit = np.min(speed_limit_forward[within_80m])


        pts = path_xy[current_idx: path_xy_80m]  # [P'',2]
        diff = np.diff(pts, axis=0, prepend=pts[0:1, :])
        yaw = np.arctan2(diff[:, 1], diff[:, 0])
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        dyaw = np.diff(yaw, axis=0, prepend=yaw[0:1])
        ds = np.linalg.norm(np.diff(pts, axis=0, prepend=pts[0:1, :]), axis=1) + 1e-3
        kappa = np.abs(dyaw / ds)
        a_lat_max = 2.0  # 你可以根据需要调
        curve_speed_limit = np.sqrt(np.maximum(a_lat_max / np.maximum(kappa, 1e-3), 0.0))
        curve_speed_limit = np.clip(curve_speed_limit, 0.0, 30.0)

        # 当前规划周期统一使用的限速
        horizon_speed_limit = float(
            min(base_speed_limit, np.min(curve_speed_limit))
        )
        
        return horizon_speed_limit
    
    def plan(
        self,
        ego_state: EgoState,
        scene_feature: SceneFeature,
        scenario_manager: SceneManager,
        trajectories: np.ndarray = None,
    ) -> np.ndarray:
        """
        Docstring for plan
        
        :param self: Description
        :param ego_state: Description
        :type ego_state: EgoState
        :param scene_feature: Description
        :type scene_feature: SceneFeature
        :param scenario_manager: Description
        :type scenario_manager: SceneManager
        :param trajectories: Description
        :type trajectories: np.ndarray
        :return: Description
        :rtype: np.ndarray
        """

        # 1. construct path plans
        path_plans = self._generate_path_plans(
            trajectories=trajectories,
            scenario_manager=scenario_manager,
            ego_state=ego_state,
        )

        # 2. update future collision map
        self._update_future_collision_map(
            scene_feature=scene_feature,
            prediction_mode="CYAW",
        )

        # 3. plan idm longitudinal behavior along each path plan
        ref_path = np.array(scene_feature.ref_path_feature)
        ref_path_xy = ref_path[:, :2]  # [P,2] ref path xy
        speed_limit_along_ref = ref_path[:, -1]  # [P,]
        horizon_speed_limit = self._get_speed_limit(
            path_xy=ref_path_xy,
            speed_limit=speed_limit_along_ref,
        )

        if self.debug:
            print(f"speed_limit_along_ref: {speed_limit_along_ref}")

        num_steps = self.trajectory_sampling.num_poses + 1  # T+1, 含 t=0
        num_factors = len(self.speed_factors)

        planned_trajectories: List[np.ndarray] = []

        for path_plan in path_plans:
            # generate 5 longitudinal IDM trajectories for this path at once
            # shape: (F, T, 6): (x, y, yaw, speed, acc, progress)
            idm_trajs = np.zeros((num_factors, num_steps, 6), dtype=float)

            # initialize with current speed/acceleration at t=0 (position/yaw still determined by first point on path)
            init_speed = float(ego_state.dynamic_car_state.speed)
            init_accel = float(ego_state.dynamic_car_state.acceleration)

            idm_trajs[..., 3] = init_speed   # all factors share the current speed
            idm_trajs[..., 4] = init_accel

            initial_progress = path_plan.project(Point(0,0))
            idm_trajs[..., 5] = initial_progress

            # rollout for t=1 ~ T
            for t_idx in range(1, num_steps):
                for f_idx, sf in enumerate(self.speed_factors):
                    
                    # extract previous step info
                    prev_speed = idm_trajs[f_idx, t_idx - 1, 3]
                    prev_acc = idm_trajs[f_idx, t_idx - 1, 4]

                    # query leading vehicle on path
                    leading_vehicle_progress, leading_vehicle_speed = self._get_leading_vehicle_on_path(
                        path_plan=path_plan,
                        time_idx=t_idx,
                        ego_state=ego_state,
                        ego_trajectory=idm_trajs[f_idx],
                    )

                    # calculate desired speed
                    desired_speed = sf * horizon_speed_limit

                    # update speed and acceleration using IDM
                    new_speed, new_acc = self._idm_step(
                        current_speed=prev_speed,
                        leading_vehicle_speed=leading_vehicle_speed,
                        leading_vehicle_progress=leading_vehicle_progress,
                        desired_speed=desired_speed,
                        prev_acc=prev_acc,
                    )
                    idm_trajs[f_idx, t_idx, 3] = new_speed
                    idm_trajs[f_idx, t_idx, 4] = new_acc

                    # accumulate progress
                    idm_trajs[f_idx, t_idx, 5] = (
                        idm_trajs[f_idx, t_idx - 1, 5]
                        + 0.5 * (prev_speed + new_speed) * self.time_step
                    )

                    # interpolate position along path
                    pos = np.array(
                        path_plan.interpolate(idm_trajs[f_idx, t_idx, 5]).coords
                    )[0]
                    idm_trajs[f_idx, t_idx, :2] = pos

                    # interpolate yaw
                    delta_pos = idm_trajs[f_idx, t_idx, :2] - idm_trajs[f_idx, t_idx - 1, :2]
                    if np.linalg.norm(delta_pos) > 1e-3:
                        idm_trajs[f_idx, t_idx, 2] = np.arctan2(delta_pos[1], delta_pos[0])
                    else:
                        idm_trajs[f_idx, t_idx, 2] = idm_trajs[f_idx, t_idx - 1, 2]

            # append the 5 trajectories for this path to the list (excluding t=0, only keep future T steps)
            planned_trajectories.append(idm_trajs[:, 1:, :])

        if len(planned_trajectories) == 0:
            return np.zeros((0, self.trajectory_sampling.num_poses, 6), dtype=float)

        planned_trajectories = np.concatenate(planned_trajectories, axis=0)
        # shape: (num_paths * 5, T, 6)
        return planned_trajectories
