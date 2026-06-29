from __future__ import annotations

import logging
import gzip
import pickle
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from scipy.signal import savgol_filter

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox, in_collision
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import find_new_collisions
from nuplan.planning.metrics.utils.collision_utils import CollisionType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.observation.idm.utils import is_agent_behind

from lpl_planner.planning.planner.utils.planner_utils import (
    hausdorff_xy,
    local_trajectory_to_abstract_trajectory,
)
from lpl_planner.planning.scene.evaluate.scene_scorer import BatchEvaluator
from lpl_planner.planning.scene.evaluate.simulator import DEFAULT_SIMULATION_DT, BatchSimulator
from lpl_planner.planning.scene.map.occupancy_map import OccupancyMap
from lpl_planner.planning.scene.scene_feature.features import (
    AgentPrediction,
    AnchorIndice,
    AnchorScores,
    ReplayPlannerTargets,
    SceneFeature,
    SceneToken,
    Trajectory,
)
from lpl_planner.planning.scene.scene_manager import SceneManager
from lpl_planner.rollout.oracle_labeler import (
    _build_dense_anchor_scores,
    _build_drivable_area_map,
    _prepend_current_state,
    _resolve_ref_path,
    _wrap_angle,
)
from lpl_planner.rollout.rollout_utils import (
    LightweightLocalDrivingEnv,
    extract_planner_state_dict,
)
from lpl_planner.training.dataset.dataset_utils import dump_feature_target_to_pickle


logger = logging.getLogger(__name__)


def ego_state_to_vector(ego_state: EgoState) -> List[float]:
    return [float(value) for value in ego_state]


def ego_state_from_vector(vector: Sequence[float]) -> EgoState:
    return EgoState.deserialize(list(vector), vehicle=get_pacifica_parameters())


def _scenario_scene_type(scenario: AbstractScenario) -> str:
    return str(getattr(scenario, "scenario_type", None) or getattr(scenario, "_scenario_type", None) or "__unknown__")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(value)).strip("_") or "sample"


def _normalize_scores(values: np.ndarray, invert: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    safe = values.copy()
    fill = float(np.max(safe[finite]) if invert else np.min(safe[finite]))
    safe[~finite] = fill
    lo = float(safe.min())
    hi = float(safe.max())
    if hi - lo <= 1e-6:
        norm = np.ones_like(safe, dtype=np.float32)
    else:
        norm = ((safe - lo) / (hi - lo)).astype(np.float32)
    return 1.0 - norm if invert else norm


def _phase_unwrap(angles: np.ndarray) -> np.ndarray:
    return np.unwrap(np.asarray(angles, dtype=np.float32))


def _safe_savgol(values: np.ndarray, polyorder: int, deriv: int = 0, delta: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] <= polyorder:
        return np.zeros_like(values, dtype=np.float32) if deriv > 0 else values.astype(np.float32, copy=False)
    window = min(9, values.shape[0] if values.shape[0] % 2 == 1 else values.shape[0] - 1)
    window = max(window, polyorder + 2 + ((polyorder + 2) % 2 == 0))
    window = min(window, values.shape[0] if values.shape[0] % 2 == 1 else values.shape[0] - 1)
    if window <= polyorder:
        return np.zeros_like(values, dtype=np.float32) if deriv > 0 else values.astype(np.float32, copy=False)
    return savgol_filter(values, polyorder=polyorder, window_length=window, deriv=deriv, delta=delta, axis=0).astype(np.float32)


def get_expert_trajectory_from_scenario_with_rollout_ego(
    scenario: AbstractScenario,
    rollout_ego_state: EgoState,
    run_step: int = 0,
    time_horizon: float = 8.0,
    time_interval: float = 0.1,
) -> np.ndarray:
    """Build the log expert's relative future motion in the rollout ego local frame.

    The future expert states are recorded in the log ego's world frame. During rollout,
    ego can already be offset from the log, so directly projecting log future world
    points into the rollout ego frame creates a discontinuous target. Instead, recover
    the expert's relative motion from the log current ego frame and use that as the
    rollout-local pseudo target.
    """
    future_count = int(time_horizon / time_interval)
    log_current_state = scenario.get_ego_state_at_iteration(run_step)
    future_states = list(
        scenario.get_ego_future_trajectory(
            iteration=run_step,
            time_horizon=time_horizon,
            num_samples=future_count,
        )
    )
    states = [log_current_state] + [state for state in future_states if state is not None]
    num_samples = len(states)
    if num_samples == 0:
        return np.zeros((0, 11), dtype=np.float32)

    expert_acc = np.zeros((num_samples, 2), dtype=np.float32)
    expert_velocity = np.zeros((num_samples, 2), dtype=np.float32)
    expert_headings = np.zeros((num_samples,), dtype=np.float32)
    expert_xy = np.zeros((num_samples, 2), dtype=np.float32)
    time_points = np.zeros((num_samples,), dtype=np.float64)

    for idx, state in enumerate(states):
        expert_acc[idx, 0] = float(state.dynamic_car_state.rear_axle_acceleration_2d.x)
        expert_acc[idx, 1] = float(state.dynamic_car_state.rear_axle_acceleration_2d.y)
        expert_velocity[idx, 0] = float(state.dynamic_car_state.rear_axle_velocity_2d.x)
        expert_velocity[idx, 1] = float(state.dynamic_car_state.rear_axle_velocity_2d.y)
        expert_headings[idx] = float(state.rear_axle.heading)
        expert_xy[idx, 0] = float(state.rear_axle.x)
        expert_xy[idx, 1] = float(state.rear_axle.y)
        time_points[idx] = float(state.time_point.time_us) / 1e6

    origin_xy = np.asarray([log_current_state.rear_axle.x, log_current_state.rear_axle.y], dtype=np.float32)
    origin_heading = float(log_current_state.rear_axle.heading)
    init_yaw = -origin_heading
    rotation_matrix = np.asarray(
        [
            [np.cos(init_yaw), -np.sin(init_yaw)],
            [np.sin(init_yaw), np.cos(init_yaw)],
        ],
        dtype=np.float32,
    )
    expert_local_xy = np.dot(expert_xy - origin_xy, rotation_matrix.T)
    expert_local_headings = _phase_unwrap(expert_headings - origin_heading)

    filtered_acceleration_x = np.round(_safe_savgol(expert_acc[:, 0], polyorder=2), decimals=8)
    filtered_acceleration_y = np.round(_safe_savgol(expert_acc[:, 1], polyorder=2), decimals=8)
    finite_dt = np.diff(time_points)
    dt = float(np.mean(finite_dt[finite_dt > 1e-6])) if np.any(finite_dt > 1e-6) else float(time_interval)
    yaw_rate = _safe_savgol(expert_local_headings, polyorder=2, deriv=1, delta=dt)
    yaw_acceleration = _safe_savgol(expert_local_headings, polyorder=3, deriv=2, delta=dt)
    jerk_x = _safe_savgol(filtered_acceleration_x, polyorder=2, deriv=1, delta=dt)
    jerk_y = _safe_savgol(filtered_acceleration_y, polyorder=2, deriv=1, delta=dt)

    trajectory = np.stack(
        [
            expert_local_xy[:, 0],
            expert_local_xy[:, 1],
            expert_local_headings,
            expert_velocity[:, 0],
            expert_velocity[:, 1],
            filtered_acceleration_x,
            filtered_acceleration_y,
            yaw_rate,
            yaw_acceleration,
            jerk_x,
            jerk_y,
        ],
        axis=-1,
    )
    return trajectory.astype(np.float32, copy=False)


def _nearest_ref_heading_and_distance(points_xy: np.ndarray, headings: np.ndarray, ref_path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points_xy = np.asarray(points_xy, dtype=np.float32)
    headings = np.asarray(headings, dtype=np.float32).reshape(-1)
    ref_path = np.asarray(ref_path, dtype=np.float32)
    if points_xy.size == 0 or ref_path.ndim != 2 or ref_path.shape[0] < 2:
        return np.full((points_xy.shape[0],), np.inf, dtype=np.float32), np.full((points_xy.shape[0],), np.pi, dtype=np.float32)

    ref_xy = ref_path[:, :2]
    if ref_path.shape[1] > 2:
        ref_heading = ref_path[:, 2].astype(np.float32, copy=False)
    else:
        diffs = np.diff(ref_xy, axis=0)
        seg_heading = np.arctan2(diffs[:, 1], diffs[:, 0]).astype(np.float32, copy=False)
        ref_heading = np.concatenate([seg_heading, seg_heading[-1:]], axis=0)

    delta = points_xy[:, None, :] - ref_xy[None, :, :]
    sq_dist = np.sum(delta * delta, axis=-1)
    nearest = np.argmin(sq_dist, axis=1)
    dist = np.sqrt(sq_dist[np.arange(points_xy.shape[0]), nearest]).astype(np.float32)
    heading_error = np.abs(_wrap_angle(headings - ref_heading[nearest])).astype(np.float32)
    return dist, heading_error


def _project_points_to_polyline_progress(points_xy: np.ndarray, path_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points_xy = np.asarray(points_xy, dtype=np.float32)
    path_xy = np.asarray(path_xy, dtype=np.float32)
    if points_xy.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    if path_xy.ndim != 2 or path_xy.shape[0] < 2:
        return np.zeros((points_xy.shape[0],), dtype=np.float32), np.full((points_xy.shape[0],), np.inf, dtype=np.float32)

    segment_start = path_xy[:-1]
    segment_vec = path_xy[1:] - path_xy[:-1]
    segment_len = np.linalg.norm(segment_vec, axis=1).astype(np.float32)
    segment_len_sq = np.maximum(segment_len * segment_len, 1e-6)
    cumulative = np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(segment_len, dtype=np.float32)])

    rel = points_xy[:, None, :] - segment_start[None, :, :]
    ratio = np.clip(np.sum(rel * segment_vec[None, :, :], axis=-1) / segment_len_sq[None, :], 0.0, 1.0)
    projected = segment_start[None, :, :] + ratio[..., None] * segment_vec[None, :, :]
    distances = np.linalg.norm(points_xy[:, None, :] - projected, axis=-1)
    best_segment = np.argmin(distances, axis=1)
    row = np.arange(points_xy.shape[0])
    progress = cumulative[best_segment] + ratio[row, best_segment] * segment_len[best_segment]
    lateral_distance = distances[row, best_segment]
    return progress.astype(np.float32, copy=False), lateral_distance.astype(np.float32, copy=False)


def _reference_consistency_scores(trajectories: np.ndarray, ref_path: np.ndarray) -> np.ndarray:
    trajectories = np.asarray(trajectories, dtype=np.float32)
    if trajectories.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if ref_path.ndim != 2 or ref_path.shape[0] < 2:
        return np.zeros((trajectories.shape[0],), dtype=np.float32)

    mid_idx = trajectories.shape[1] // 2
    key_xy = np.stack([trajectories[:, mid_idx, :2], trajectories[:, -1, :2]], axis=1)
    key_heading = np.stack([trajectories[:, mid_idx, 2], trajectories[:, -1, 2]], axis=1)
    key_dist = []
    key_heading_error = []
    for key in range(2):
        dist, heading_error = _nearest_ref_heading_and_distance(key_xy[:, key], key_heading[:, key], ref_path)
        key_dist.append(dist)
        key_heading_error.append(heading_error)

    dist_cost = 0.45 * key_dist[1] + 0.25 * key_dist[0]
    heading_cost = 0.20 * key_heading_error[1] + 0.10 * key_heading_error[0]
    cost = dist_cost / 4.0 + heading_cost / np.deg2rad(35.0)
    return np.exp(-cost).astype(np.float32)


def _state_xy(ego_state: EgoState) -> np.ndarray:
    center = getattr(ego_state, "center", None)
    if center is not None:
        return np.asarray([float(center.x), float(center.y)], dtype=np.float32)
    return np.asarray([float(ego_state.rear_axle.x), float(ego_state.rear_axle.y)], dtype=np.float32)


def _tracked_object_pose_speed_box(tracked_object: Any) -> Optional[Tuple[np.ndarray, float, OrientedBox]]:
    center = getattr(tracked_object, "center", None)
    box = getattr(tracked_object, "box", None)
    if center is None or box is None:
        return None
    speed = float(tracked_object.velocity.magnitude()) if isinstance(tracked_object, Agent) and tracked_object.velocity is not None else 0.0
    return np.asarray([float(center.x), float(center.y), float(center.heading)], dtype=np.float64), speed, box


def _estimate_min_ttc(
    ego_state: EgoState,
    observation: Any,
    time_step_size: float,
    time_horizon: float,
    max_distance_m: float = 60.0,
    stopped_speed_threshold: float = 5e-3,
) -> float:
    tracked_objects = getattr(observation, "tracked_objects", observation)
    objects = getattr(tracked_objects, "tracked_objects", tracked_objects)
    ego_pose = np.asarray([float(ego_state.center.x), float(ego_state.center.y), float(ego_state.center.heading)], dtype=np.float64)
    ego_speed = float(ego_state.dynamic_car_state.speed)
    if ego_speed <= stopped_speed_threshold:
        return float(np.inf)

    track_poses: List[np.ndarray] = []
    track_speeds: List[float] = []
    track_boxes: List[OrientedBox] = []

    for tracked_object in objects:
        center = getattr(tracked_object, "center", None)
        if center is None:
            continue
        state = _tracked_object_pose_speed_box(tracked_object)
        if state is None:
            continue
        pose, speed, box = state
        distance = float(np.linalg.norm(pose[:2] - ego_pose[:2]))
        if distance > 1e-3 and is_agent_behind(ego_state.rear_axle, center):
            continue
        if distance > max_distance_m:
            continue
        if in_collision(ego_state.car_footprint.oriented_box, box):
            return 0.0
        track_poses.append(pose)
        track_speeds.append(speed)
        track_boxes.append(box)

    if not track_poses:
        return float(np.inf)

    ego_box = ego_state.car_footprint.oriented_box
    ego_dx = float(np.cos(ego_pose[2]) * ego_speed * time_step_size)
    ego_dy = float(np.sin(ego_pose[2]) * ego_speed * time_step_size)
    poses = np.asarray(track_poses, dtype=np.float64)
    speeds = np.asarray(track_speeds, dtype=np.float64)
    tracks_dxy = np.stack(
        [
            np.cos(poses[:, 2]) * speeds * time_step_size,
            np.sin(poses[:, 2]) * speeds * time_step_size,
        ],
        axis=1,
    )

    for time_to_collision in np.arange(time_step_size, time_horizon + 1e-6, time_step_size):
        ego_pose[:2] += (ego_dx, ego_dy)
        projected_ego_box = OrientedBox.from_new_pose(ego_box, StateSE2(*ego_pose))
        poses[:, :2] += tracks_dxy
        for track_box, track_pose in zip(track_boxes, poses):
            projected_track_box = OrientedBox.from_new_pose(track_box, StateSE2(*track_pose))
            if in_collision(projected_ego_box, projected_track_box):
                return float(time_to_collision)

    return float(np.inf)


def _model_expert_disagreement(
    model_trajectory: np.ndarray,
    expert_trajectory: np.ndarray,
    wait_speed_mps: float,
    wait_progress_m: float,
    forward_progress_gap_m: float,
    lag_progress_gap_m: float,
    moving_speed_mps: float,
) -> Tuple[bool, str]:
    model_trajectory = np.asarray(model_trajectory, dtype=np.float32)
    expert_trajectory = np.asarray(expert_trajectory, dtype=np.float32)
    if model_trajectory.ndim != 2 or expert_trajectory.ndim != 2 or model_trajectory.shape[0] < 2 or expert_trajectory.shape[0] < 2:
        return False, ""

    horizon = min(model_trajectory.shape[0], expert_trajectory.shape[0] - 1)
    if horizon <= 1:
        return False, ""
    model = model_trajectory[:horizon]
    expert = expert_trajectory[1 : horizon + 1]
    mid_idx = horizon // 2

    model_mid_progress = float(model[mid_idx, 0])
    model_end_progress = float(model[-1, 0])
    expert_mid_progress = float(expert[mid_idx, 0])
    expert_end_progress = float(expert[-1, 0])
    model_end_speed = float(model[-1, 3]) if model.shape[1] > 3 else 0.0
    expert_end_speed = float(expert[-1, 3]) if expert.shape[1] > 3 else 0.0
    expert_mean_speed = float(np.mean(np.abs(expert[:, 3]))) if expert.shape[1] > 3 else 0.0

    expert_waiting = expert_end_progress <= wait_progress_m and expert_mean_speed <= wait_speed_mps
    model_forward = model_end_progress >= wait_progress_m + forward_progress_gap_m or model_end_speed >= moving_speed_mps
    if expert_waiting and model_forward:
        return True, "expert_wait_model_forward"

    progress_gap = expert_end_progress - model_end_progress
    mid_progress_gap = expert_mid_progress - model_mid_progress
    if expert_end_speed >= moving_speed_mps and max(progress_gap, mid_progress_gap) >= lag_progress_gap_m:
        return True, "model_lagging_expert"

    if model_end_progress - expert_end_progress >= forward_progress_gap_m:
        return True, "model_ahead_expert"
    return False, ""


def _expert_consistency_scores(trajectories: np.ndarray, expert_trajectory: np.ndarray, ref_path: Optional[np.ndarray] = None) -> np.ndarray:
    trajectories = np.asarray(trajectories, dtype=np.float32)
    expert_trajectory = np.asarray(expert_trajectory, dtype=np.float32)
    if trajectories.size == 0 or expert_trajectory.ndim != 2 or expert_trajectory.shape[0] < 2:
        return np.zeros((trajectories.shape[0],), dtype=np.float32)

    horizon = min(trajectories.shape[1], expert_trajectory.shape[0] - 1)
    if horizon <= 1:
        return np.zeros((trajectories.shape[0],), dtype=np.float32)
    expert = expert_trajectory[1 : horizon + 1]
    traj = trajectories[:, :horizon]
    hausdorff_cost = np.asarray(
        [float(hausdorff_xy(traj[idx : idx + 1, :, :2], expert[:, :2])) for idx in range(traj.shape[0])],
        dtype=np.float32,
    )
    _, endpoint_heading_error = _nearest_ref_heading_and_distance(
        traj[:, -1, :2],
        traj[:, -1, 2],
        expert[:, :3],
    )

    expert_xy = expert[:, :2]
    projection_path = np.asarray(ref_path, dtype=np.float32)[:, :2] if ref_path is not None else expert_xy
    if projection_path.ndim != 2 or projection_path.shape[0] < 2:
        projection_path = expert_xy
    mid_idx = horizon // 2
    key_indices = np.asarray([mid_idx, horizon - 1], dtype=np.int64)
    key_points = traj[:, key_indices, :2].reshape(-1, 2)
    key_progress, _ = _project_points_to_polyline_progress(key_points, projection_path)
    key_progress = key_progress.reshape(traj.shape[0], 2)
    expert_progress, _ = _project_points_to_polyline_progress(expert[key_indices, :2], projection_path)
    longitudinal_cost = (
        0.40 * np.abs(key_progress[:, 0] - expert_progress[0])
        + 0.60 * np.abs(key_progress[:, 1] - expert_progress[1])
    ).astype(np.float32)

    if traj.shape[-1] > 3 and expert.shape[-1] > 3:
        key_speed = traj[:, key_indices, 3]
        expert_speed = expert[key_indices, 3]
        speed_cost = (
            0.40 * np.abs(key_speed[:, 0] - expert_speed[0])
            + 0.60 * np.abs(key_speed[:, 1] - expert_speed[1])
        ).astype(np.float32)
    else:
        speed_cost = np.zeros((traj.shape[0],), dtype=np.float32)

    cost = (
        0.35 * (hausdorff_cost / 4.0)
        + 0.15 * (endpoint_heading_error / np.deg2rad(35.0))
        + 0.35 * (longitudinal_cost / 4.0)
        + 0.15 * (speed_cost / 3.0)
    )
    return np.exp(-cost).astype(np.float32)


def _build_anchor_prefilter(
    planner_anchor: np.ndarray,
    ref_path: np.ndarray,
    drivable_area_map: Optional[OccupancyMap],
    topk: int,
    max_ref_distance_m: float,
    max_heading_error_rad: float,
) -> np.ndarray:
    anchor_num = int(planner_anchor.shape[0])
    topk = max(1, min(int(topk), anchor_num))
    key_indices = np.asarray([planner_anchor.shape[1] // 2, planner_anchor.shape[1] - 1], dtype=np.int64)
    key_xy = planner_anchor[:, key_indices, :2]
    key_heading = planner_anchor[:, key_indices, 2]

    mid_dist, mid_heading = _nearest_ref_heading_and_distance(key_xy[:, 0], key_heading[:, 0], ref_path)
    end_dist, end_heading = _nearest_ref_heading_and_distance(key_xy[:, 1], key_heading[:, 1], ref_path)
    coarse_mask = (
        np.isfinite(mid_dist)
        & np.isfinite(end_dist)
        & (mid_dist <= float(max_ref_distance_m))
        & (end_dist <= float(max_ref_distance_m))
        & (mid_heading <= float(max_heading_error_rad))
        & (end_heading <= float(max_heading_error_rad))
    )

    candidate_indices = np.flatnonzero(coarse_mask)
    if candidate_indices.size == 0:
        candidate_indices = np.arange(anchor_num, dtype=np.int64)

    if drivable_area_map is not None and candidate_indices.size > 0:
        points = planner_anchor[candidate_indices][:, key_indices, :2].reshape(-1, 2)
        in_drivable = drivable_area_map.points_in_polygons(points).any(axis=0).reshape(candidate_indices.shape[0], 2)
        drivable_mask = in_drivable.all(axis=1)
        if drivable_mask.any():
            candidate_indices = candidate_indices[drivable_mask]

    geo_cost = (
        0.35 * end_dist[candidate_indices]
        + 0.25 * mid_dist[candidate_indices]
        + 0.25 * end_heading[candidate_indices]
        + 0.15 * mid_heading[candidate_indices]
    )
    order = np.argsort(geo_cost)
    return candidate_indices[order[:topk]].astype(np.int64, copy=False)


def _find_stop_anchor_indices(planner_anchor: np.ndarray, max_count: int, max_endpoint_distance_m: float, max_mean_speed_mps: float) -> np.ndarray:
    planner_anchor = np.asarray(planner_anchor, dtype=np.float32)
    if planner_anchor.ndim != 3 or planner_anchor.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    endpoint_distance = np.linalg.norm(planner_anchor[:, -1, :2], axis=1)
    if planner_anchor.shape[-1] > 3:
        mean_speed = np.mean(np.abs(planner_anchor[..., 3]), axis=1)
    else:
        step_distance = np.linalg.norm(np.diff(planner_anchor[..., :2], axis=1), axis=-1)
        mean_speed = np.mean(step_distance, axis=1)

    stop_mask = (endpoint_distance <= float(max_endpoint_distance_m)) & (mean_speed <= float(max_mean_speed_mps))
    stop_indices = np.flatnonzero(stop_mask)
    if stop_indices.size == 0:
        stop_indices = np.asarray([int(np.argmin(endpoint_distance + mean_speed))], dtype=np.int64)

    order = np.lexsort((mean_speed[stop_indices], endpoint_distance[stop_indices]))
    return stop_indices[order[: max(1, int(max_count))]].astype(np.int64, copy=False)


def _append_unique_indices(indices: np.ndarray, extra_indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    extra_indices = np.asarray(extra_indices, dtype=np.int64).reshape(-1)
    if extra_indices.size == 0:
        return indices
    return np.unique(np.concatenate([indices, extra_indices], axis=0)).astype(np.int64, copy=False)


class RolloutCLScenarioWorker:
    def __init__(
        self,
        cfg: DictConfig,
        state_dict: Optional[Dict[str, Any]] = None,
        planner_anchor: Optional[np.ndarray] = None,
        enable_actor: bool = True,
    ) -> None:
        self.cfg = cfg
        self.device = self._resolve_device()
        self.enable_actor = bool(enable_actor)
        self.actor = instantiate(cfg.model) if self.enable_actor else None
        if self.actor is not None and state_dict is not None:
            self.set_weights(state_dict)
        if self.actor is not None:
            self.actor.eval()
        self.actor_device = torch.device("cpu")
        self.env = LightweightLocalDrivingEnv(cfg=cfg) if self.enable_actor else None
        self.rollout_package_root = Path(str(getattr(cfg, "rollout_package_dir", getattr(cfg, "rollout_cache_dir", "rollout_packages"))))
        self.output_cache_root = Path(str(getattr(cfg, "rollout_cl_cache_dir", getattr(cfg.cache, "cache_path", "rollout_cl_cache"))))
        self.debug_dir = Path(str(getattr(cfg, "rollout_debug_dir", self.output_cache_root / "_debug")))
        self.debug = bool(getattr(cfg, "debug_rollout_cache", False))
        self.debug_rollout_topk = max(int(getattr(cfg, "debug_rollout_topk", 64)), 1)
        self.retrieval_style = str(getattr(cfg, "rollout_retrieval_style", "r2lpl")).strip().lower()
        self.keep_success_tail = bool(getattr(cfg, "keep_success_tail", False))
        self.num_samples = max(int(getattr(cfg, "num_samples", 64)), 1)
        self.temperature = float(getattr(cfg, "temperature", 1.0))
        self.top_k = int(getattr(cfg, "top_k", 0))
        self.top_p = float(getattr(cfg, "top_p", 1.0))
        self.deterministic_first = bool(getattr(cfg, "deterministic_first", True))
        self.rollout_dt = float(getattr(cfg, "rollout_dt", 0.2))
        self.failure_window_steps = max(int(getattr(cfg, "failure_window_steps", 15)), 1)
        self.risk_ttc_threshold_s = float(getattr(cfg, "risk_ttc_threshold_s", 1.0))
        self.risk_pre_window_steps = max(int(getattr(cfg, "risk_pre_window_steps", 10)), 0)
        self.ttc_max_distance_m = float(getattr(cfg, "ttc_max_distance_m", 60.0))
        self.ttc_time_step_size = float(getattr(cfg, "ttc_time_step_size", 0.1))
        self.ttc_time_horizon_s = float(getattr(cfg, "ttc_time_horizon_s", 3.0))
        self.ttc_stopped_speed_threshold = float(getattr(cfg, "ttc_stopped_speed_threshold", 5e-3))
        self.disagreement_wait_speed_mps = float(getattr(cfg, "disagreement_wait_speed_mps", 0.5))
        self.disagreement_wait_progress_m = float(getattr(cfg, "disagreement_wait_progress_m", 1.0))
        self.disagreement_forward_progress_gap_m = float(getattr(cfg, "disagreement_forward_progress_gap_m", 2.0))
        self.disagreement_lag_progress_gap_m = float(getattr(cfg, "disagreement_lag_progress_gap_m", 3.0))
        self.disagreement_moving_speed_mps = float(getattr(cfg, "disagreement_moving_speed_mps", 1.0))
        self.oracle_prefilter_topk = max(int(getattr(cfg, "oracle_prefilter_topk", 1024)), 1)
        self.oracle_replay_topk = max(int(getattr(cfg, "oracle_replay_topk", 16)), 1)
        self.road_model_score_weight = float(getattr(cfg, "road_model_score_weight", 0.35))
        self.road_expert_score_weight = float(getattr(cfg, "road_expert_score_weight", 0.65))
        self.oracle_score_min = float(getattr(cfg, "oracle_score_min", 0.15))
        self.oracle_fill_margin = float(getattr(cfg, "oracle_fill_margin", 1.0))
        self.force_stop_anchor = bool(getattr(cfg, "force_stop_anchor", True))
        self.stop_anchor_endpoint_distance_m = float(getattr(cfg, "stop_anchor_endpoint_distance_m", 0.75))
        self.stop_anchor_mean_speed_mps = float(getattr(cfg, "stop_anchor_mean_speed_mps", 0.3))
        self.near_log_xy_m = float(getattr(cfg, "near_log_xy_m", 1.5))
        self.near_log_yaw_rad = float(getattr(cfg, "near_log_yaw_rad", np.deg2rad(15.0)))
        self.near_log_speed_mps = float(getattr(cfg, "near_log_speed_mps", 2.0))
        self.near_log_expert_margin = float(getattr(cfg, "near_log_expert_margin", 0.05))
        self.near_log_expert_min = float(getattr(cfg, "near_log_expert_min", 0.75))
        self.recoverable_xy_m = float(getattr(cfg, "recoverable_xy_m", 5.0))
        self.recoverable_yaw_rad = float(getattr(cfg, "recoverable_yaw_rad", np.deg2rad(45.0)))
        self.max_ref_distance_m = float(getattr(cfg, "oracle_max_ref_distance_m", 6.0))
        self.max_heading_error_rad = float(getattr(cfg, "oracle_max_heading_error_rad", np.deg2rad(60.0)))
        self.anchor_indice_name = str(getattr(cfg, "anchor_indice_name", "anchor_indice.gz"))
        self.anchor_score_name = str(getattr(cfg, "anchor_score_name", "anchor_scores.gz"))
        self.replay_target_name = str(getattr(cfg, "replay_target_name", "replay_planner_targets.gz"))
        self.planner_anchor = self._resolve_planner_anchor(planner_anchor)
        self.stop_anchor_indices = (
            _find_stop_anchor_indices(
                self.planner_anchor,
                max_count=1,
                max_endpoint_distance_m=self.stop_anchor_endpoint_distance_m,
                max_mean_speed_mps=self.stop_anchor_mean_speed_mps,
            )
            if self.force_stop_anchor
            else np.zeros((0,), dtype=np.int64)
        )
        self.proposal_sampling = self._build_proposal_sampling()
        self.simulator = BatchSimulator(self.proposal_sampling, default_dt=DEFAULT_SIMULATION_DT)
        self.evaluator = BatchEvaluator(self.proposal_sampling, default_dt=DEFAULT_SIMULATION_DT)
        self._package_cache: Dict[str, Dict[str, Any]] = {}

    def _resolve_device(self) -> torch.device:
        if float(getattr(self.cfg, "gpus_per_worker", 0.0)) > 0 and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _resolve_planner_anchor(self, planner_anchor: Optional[np.ndarray]) -> np.ndarray:
        if planner_anchor is not None:
            return np.asarray(planner_anchor, dtype=np.float32)
        actor_anchor = getattr(self.actor, "planner_anchor", None) if self.actor is not None else None
        if actor_anchor is None:
            anchor_path = getattr(self.cfg, "oracle_planner_anchor_path", None) or getattr(self.cfg, "planner_anchor_path", None)
            if anchor_path in {None, "", "None"}:
                raise AttributeError("planner_anchor is required from model or oracle_planner_anchor_path")
            return np.load(str(anchor_path), mmap_mode="r")
        return actor_anchor.detach().cpu().numpy().astype(np.float32, copy=False) if torch.is_tensor(actor_anchor) else np.asarray(actor_anchor, dtype=np.float32)

    def _build_proposal_sampling(self) -> TrajectorySampling:
        future_sampling = getattr(self.actor, "future_sampling", None) if self.actor is not None else None
        if future_sampling is not None:
            return TrajectorySampling(num_poses=int(future_sampling.num_poses), interval_length=float(future_sampling.interval_length))
        return TrajectorySampling(num_poses=int(self.planner_anchor.shape[1]), interval_length=float(getattr(self.cfg, "dt", 0.2)))

    def _resolve_rollout_max_steps(self) -> Optional[int]:
        max_steps = getattr(self.cfg, "rollout_max_steps", None)
        if max_steps in {None, "", "None"}:
            max_steps = getattr(self.cfg, "episode_len", None)
        if max_steps in {None, "", "None"}:
            return None
        return int(max_steps)

    def set_weights(self, state_dict: Dict[str, Any]) -> None:
        if self.actor is None:
            return
        if "state_dict" in state_dict:
            state_dict = extract_planner_state_dict(state_dict)
        result = self.actor.load_state_dict(state_dict, strict=False)
        if result.missing_keys or result.unexpected_keys:
            logger.warning("Loaded rollout actor with missing=%s unexpected=%s", result.missing_keys, result.unexpected_keys)

    def _prepare_actor(self) -> None:
        if self.actor is None:
            raise RuntimeError("Actor is disabled for this worker.")
        if self.actor_device != self.device:
            self.actor.to(self.device)
            self.actor_device = self.device
        self.actor.eval()

    def _sample_model_action(self, scene_feature: SceneFeature, expert_trajectory: Optional[np.ndarray] = None) -> Dict[str, Any]:
        self._prepare_actor()
        with torch.no_grad():
            feature_tensor = scene_feature.to_feature_tensor()
            batched_scene_feature = feature_tensor.collate([feature_tensor]).to_device(self.actor_device)
            output = self.actor.sample_trajectories(
                features={"scene_feature": batched_scene_feature},
                num_samples=self.num_samples,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                deterministic_first=self.deterministic_first,
            )
        trajectories = output["trajectories"][0].detach().cpu().numpy().astype(np.float32, copy=False)
        indices = output["indices"][0].detach().cpu().numpy().astype(np.int32, copy=False).reshape(-1)
        scores = output["scores"][0].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        selection_scores = scores
        expert_scores = np.zeros_like(scores, dtype=np.float32)
        if self.retrieval_style == "road" and expert_trajectory is not None:
            expert_scores = _expert_consistency_scores(trajectories, expert_trajectory)
            model_scores = _normalize_scores(scores)
            selection_scores = (
                self.road_model_score_weight * model_scores
                + self.road_expert_score_weight * expert_scores
            ).astype(np.float32, copy=False)
        best = int(np.argmax(selection_scores))
        return {
            "trajectory": trajectories[best],
            "anchor_index": int(indices[best]),
            "model_score": float(scores[best]),
            "expert_score": float(expert_scores[best]) if expert_scores.size else 0.0,
            "selection_score": float(selection_scores[best]),
            "sampled_anchor_indices": indices,
            "sampled_model_scores": scores,
            "sampled_expert_scores": expert_scores,
            "sampled_selection_scores": selection_scores,
        }

    def _check_done(self, scenario: AbstractScenario, ego_state: EgoState, iteration: int, collided_track_ids: set) -> Tuple[bool, str]:
        if iteration >= scenario.get_number_of_iterations() - 1:
            return True, "success"
        if not np.isfinite([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading]).all():
            return True, "invalid_ego_state"

        observation = scenario.get_tracked_objects_at_iteration(iteration)
        new_collisions, collisions_id_data = find_new_collisions(ego_state, observation, collided_track_ids)
        collided_track_ids.update(new_collisions)
        for _, collision_data in collisions_id_data.items():
            if collision_data.collision_type in {
                CollisionType.ACTIVE_FRONT_COLLISION,
                CollisionType.STOPPED_TRACK_COLLISION,
                CollisionType.ACTIVE_LATERAL_COLLISION,
            }:
                return True, "at_fault_collision"

        for corner in ego_state.car_footprint.all_corners():
            _, distance = scenario.map_api.get_distance_to_nearest_map_object(corner, layer=SemanticMapLayer.DRIVABLE_AREA)
            if distance is None or float(distance) > 0.5:
                return True, "offroad"
        return False, ""

    def _package_path(self, scenario: AbstractScenario) -> Path:
        return self.rollout_package_root / scenario.log_name / _scenario_scene_type(scenario) / scenario.token / "rollout_package.gz"

    def _save_rollout_package(self, scenario: AbstractScenario, package: Dict[str, Any]) -> Path:
        package_path = self._package_path(scenario)
        package_path.parent.mkdir(parents=True, exist_ok=True)
        dump_feature_target_to_pickle(package_path, package)
        return package_path

    def collect_rollout_package(self, scenario: AbstractScenario, max_steps: Optional[int] = None) -> Dict[str, Any]:
        if self.env is None:
            raise RuntimeError("Rollout collection requires enable_actor=True.")
        self.env.set_scenario(scenario)
        scene_feature = self.env.reset()
        ego_history: Deque[EgoState] = deque(self.env.get_ego_history())
        ego_vectors = [ego_state_to_vector(state) for state in ego_history]
        frame_records: List[Dict[str, Any]] = []
        action_records: List[Dict[str, Any]] = []
        collided_track_ids = set()
        max_steps = self.env.max_steps if max_steps is None else min(int(max_steps), self.env.max_steps)
        done = False
        failure_type = "success"
        first_failure_type = ""
        first_failure_iteration: Optional[int] = None

        while not done and self.env.current_iteration < max_steps:
            current_state = self.env.get_state()
            current_index = len(ego_vectors) - 1
            observation = scenario.get_tracked_objects_at_iteration(int(self.env.current_iteration))
            min_ttc = _estimate_min_ttc(
                current_state,
                observation,
                time_step_size=self.ttc_time_step_size,
                time_horizon=self.ttc_time_horizon_s,
                max_distance_m=self.ttc_max_distance_m,
                stopped_speed_threshold=self.ttc_stopped_speed_threshold,
            )
            expert_trajectory = get_expert_trajectory_from_scenario_with_rollout_ego(
                scenario,
                rollout_ego_state=current_state,
                run_step=int(self.env.current_iteration),
                time_horizon=float(self.proposal_sampling.time_horizon),
                time_interval=float(self.proposal_sampling.interval_length),
            )
            action = self._sample_model_action(scene_feature, expert_trajectory=expert_trajectory)
            is_disagreement, disagreement_reason = _model_expert_disagreement(
                action["trajectory"],
                expert_trajectory,
                wait_speed_mps=self.disagreement_wait_speed_mps,
                wait_progress_m=self.disagreement_wait_progress_m,
                forward_progress_gap_m=self.disagreement_forward_progress_gap_m,
                lag_progress_gap_m=self.disagreement_lag_progress_gap_m,
                moving_speed_mps=self.disagreement_moving_speed_mps,
            )
            frame_records.append(
                {
                    "iteration": int(self.env.current_iteration),
                    "ego_state_index": int(current_index),
                    "chosen_anchor_index": int(action["anchor_index"]),
                    "chosen_model_score": float(action["model_score"]),
                    "chosen_expert_score": float(action.get("expert_score", 0.0)),
                    "chosen_selection_score": float(action.get("selection_score", action["model_score"])),
                    "sampled_anchor_indices": np.asarray(action.get("sampled_anchor_indices", []), dtype=np.int32).tolist(),
                    "sampled_model_scores": np.asarray(action.get("sampled_model_scores", []), dtype=np.float32).tolist(),
                    "sampled_expert_scores": np.asarray(action.get("sampled_expert_scores", []), dtype=np.float32).tolist(),
                    "sampled_selection_scores": np.asarray(action.get("sampled_selection_scores", []), dtype=np.float32).tolist(),
                    "min_ttc": float(min_ttc),
                    "is_high_risk": bool(np.isfinite(min_ttc) and min_ttc < self.risk_ttc_threshold_s),
                    "is_model_expert_disagreement": bool(is_disagreement),
                    "disagreement_reason": disagreement_reason,
                }
            )
            action_records.append(
                {
                    "iteration": int(self.env.current_iteration),
                    "anchor_index": int(action["anchor_index"]),
                    "model_score": float(action["model_score"]),
                }
            )

            future_horizon = float(action["trajectory"].shape[0]) * float(self.proposal_sampling.interval_length)
            abstract_trajectory = local_trajectory_to_abstract_trajectory(
                trajectory=action["trajectory"],
                ego_state=current_state,
                future_horizon=future_horizon,
                step_interval=float(self.proposal_sampling.interval_length),
            )
            scene_feature, reached_max = self.env.step(abstract_trajectory)
            next_state = self.env.get_state()
            ego_vectors.append(ego_state_to_vector(next_state))
            done, failure_type = self._check_done(scenario, next_state, self.env.current_iteration, collided_track_ids)
            if failure_type and failure_type != "success" and first_failure_iteration is None:
                first_failure_type = failure_type
                first_failure_iteration = int(self.env.current_iteration)
            if self.retrieval_style == "road" and failure_type != "invalid_ego_state":
                done = False
            if reached_max and not done:
                failure_type = "success"
            done = done or reached_max

        failure_iteration = int(self.env.current_iteration)
        if self.retrieval_style == "road" and first_failure_iteration is not None:
            failure_type = first_failure_type
            failure_iteration = int(first_failure_iteration)
        high_risk_iterations = {
            int(record["iteration"])
            for record in frame_records
            if bool(record.get("is_high_risk", False))
        }
        high_risk_context_iterations = set(high_risk_iterations)
        for high_risk_iteration in high_risk_iterations:
            start_iteration = max(0, high_risk_iteration - self.risk_pre_window_steps)
            high_risk_context_iterations.update(range(start_iteration, high_risk_iteration + 1))

        for record in frame_records:
            iteration = int(record["iteration"])
            candidate_reasons: List[str] = []
            record["distance_to_failure_steps"] = max(failure_iteration - int(record["iteration"]), 0)
            record["distance_to_failure_s"] = float(record["distance_to_failure_steps"]) * self.rollout_dt
            if self.retrieval_style != "road":
                if (failure_type != "success" or self.keep_success_tail) and record["distance_to_failure_steps"] <= self.failure_window_steps:
                    candidate_reasons.append("failure_window")
                if iteration in high_risk_context_iterations:
                    candidate_reasons.append("high_risk_context")
                if bool(record.get("is_model_expert_disagreement", False)):
                    candidate_reasons.append("model_expert_disagreement")
            record["oracle_candidate_reasons"] = candidate_reasons

        package = {
            "version": "rollout_cl_package_v1",
            "log_name": scenario.log_name,
            "scene_type": _scenario_scene_type(scenario),
            "scenario_token": scenario.token,
            "scenario_name": getattr(scenario, "scenario_name", scenario.token),
            "failure_type": failure_type,
            "termination_type": failure_type,
            "is_failure": bool(failure_type != "success"),
            "failure_iteration": failure_iteration,
            "ego_state_vectors": np.asarray(ego_vectors, dtype=np.float64),
            "frame_records": frame_records,
            "action_records": action_records,
        }
        package["package_path"] = str(self._save_rollout_package(scenario, package))
        return package

    def collect_rollout_package_task(self, scenario: AbstractScenario) -> Dict[str, Any]:
        package = self.collect_rollout_package(scenario, max_steps=self._resolve_rollout_max_steps())
        return {
            "scenario_token": scenario.token,
            "log_name": scenario.log_name,
            "scene_type": _scenario_scene_type(scenario),
            "package_path": package.get("package_path"),
            "failure_type": package.get("failure_type"),
            "termination_type": package.get("termination_type"),
            "is_failure": bool(package.get("is_failure", package.get("failure_type") != "success")),
            "failure_iteration": package.get("failure_iteration"),
            "frames": len(package.get("frame_records", [])),
            "frame_records": package.get("frame_records", []),
        }

    def _frame_state_class(self, rollout_ego: EgoState, log_ego: EgoState) -> str:
        dx = float(np.linalg.norm(rollout_ego.rear_axle.array - log_ego.rear_axle.array))
        dyaw = float(abs(_wrap_angle(np.asarray([rollout_ego.rear_axle.heading - log_ego.rear_axle.heading], dtype=np.float32))[0]))
        dv = float(abs(rollout_ego.dynamic_car_state.speed - log_ego.dynamic_car_state.speed))
        if dx <= self.near_log_xy_m and dyaw <= self.near_log_yaw_rad and dv <= self.near_log_speed_mps:
            return "near_log"
        if dx <= self.recoverable_xy_m and dyaw <= self.recoverable_yaw_rad:
            return "recoverable"
        return "far_offpolicy"

    def _recover_frame_target(
        self,
        scenario: AbstractScenario,
        package: Dict[str, Any],
        frame_record: Dict[str, Any],
        scene_manager: SceneManager,
    ) -> Optional[Dict[str, Any]]:
        iteration = int(frame_record["iteration"])
        ego_states = [ego_state_from_vector(vector) for vector in package["ego_state_vectors"]]
        ego_index = int(frame_record["ego_state_index"])
        ego_state = ego_states[ego_index]
        ego_history = ego_states[: ego_index + 1]
        log_ego = scenario.get_ego_state_at_iteration(iteration)
        state_class = self._frame_state_class(ego_state, log_ego)

        feature_dict, target_dict = scene_manager.extract_feature_target_from_scenario(
            scenario=scenario,
            iteration=iteration,
            ego_state=ego_state,
            ego_history=ego_history,
            use_route_correction=True,
        )
        scene_feature = SceneFeature.deserialize(feature_dict)
        agent_prediction = AgentPrediction.deserialize(target_dict)
        ref_path = _resolve_ref_path(scene_feature)
        drivable_area_map = _build_drivable_area_map(scene_feature)
        selected_indices = _build_anchor_prefilter(
            planner_anchor=self.planner_anchor,
            ref_path=ref_path,
            drivable_area_map=drivable_area_map,
            topk=self.oracle_prefilter_topk,
            max_ref_distance_m=self.max_ref_distance_m,
            max_heading_error_rad=self.max_heading_error_rad,
        )
        selected_indices = _append_unique_indices(selected_indices, self.stop_anchor_indices)
        if selected_indices.size == 0:
            return None

        selected_anchors = np.asarray(self.planner_anchor[selected_indices], dtype=np.float32)
        simulated = self.simulator.simulate(_prepend_current_state(scene_feature, selected_anchors), ego_state=ego_state)
        expert_trajectory = get_expert_trajectory_from_scenario_with_rollout_ego(
            scenario,
            rollout_ego_state=ego_state,
            run_step=iteration,
            time_horizon=float(self.proposal_sampling.time_horizon),
            time_interval=float(self.proposal_sampling.interval_length),
        )
        scores = self.evaluator.batch_evaluate(
            simulated,
            scene_feature=scene_feature,
            agent_prediction_gt=agent_prediction,
            ref_path=ref_path,
            scene_manager=scene_manager,
            expert_trajectory=expert_trajectory,
            aggregate_only=False,
        )
        eval_scores = np.asarray(scores["aggregate_scores"], dtype=np.float32)
        if eval_scores.size == 0 or float(np.max(eval_scores)) <= self.oracle_score_min:
            return None
        valid_eval_mask = np.isfinite(eval_scores) & (eval_scores > 0.0)
        if not valid_eval_mask.any():
            return None
        selected_indices = selected_indices[valid_eval_mask]
        selected_anchors = selected_anchors[valid_eval_mask]
        eval_scores = eval_scores[valid_eval_mask]

        ref_consistency = _reference_consistency_scores(selected_anchors, ref_path)
        expert_consistency = _expert_consistency_scores(selected_anchors, expert_trajectory, ref_path=ref_path)
        eval_norm = _normalize_scores(eval_scores)
        if state_class == "near_log":
            final_scores = 0.10 * eval_norm + 0.80 * expert_consistency + 0.10 * ref_consistency
            expert_gate = expert_consistency >= max(
                float(np.max(expert_consistency)) - self.near_log_expert_margin,
                self.near_log_expert_min,
            )
            if expert_gate.any():
                final_scores = np.where(expert_gate, final_scores, -1.0).astype(np.float32)
        elif state_class == "recoverable":
            final_scores = 0.65 * eval_norm + 0.30 * ref_consistency + 0.05 * expert_consistency
        else:
            final_scores = 0.80 * eval_norm + 0.20 * ref_consistency

        best_local = int(np.argmax(final_scores))
        best_anchor = int(selected_indices[best_local])
        dense_scores = _build_dense_anchor_scores(
            anchor_num=int(self.planner_anchor.shape[0]),
            sampled_indices=selected_indices,
            sampled_scores=final_scores,
            fill_margin=self.oracle_fill_margin,
        )
        replay_candidates = np.flatnonzero(np.isfinite(final_scores) & (final_scores > 0.0))
        if replay_candidates.size == 0:
            replay_candidates = np.asarray([best_local], dtype=np.int64)
        replay_order = replay_candidates[np.argsort(final_scores[replay_candidates])[::-1][: self.oracle_replay_topk]]
        return {
            "iteration": iteration,
            "state_class": state_class,
            "ego_state": ego_state,
            "ego_history": ego_history,
            "scene_feature": scene_feature,
            "agent_prediction": agent_prediction,
            "expert_trajectory": expert_trajectory,
            "selected_indices": selected_indices,
            "selected_anchors": selected_anchors,
            "eval_scores": eval_scores,
            "final_scores": final_scores.astype(np.float32, copy=False),
            "best_anchor": best_anchor,
            "best_score": float(final_scores[best_local]),
            "dense_scores": dense_scores,
            "replay_anchor_indices": selected_indices[replay_order].astype(np.int32, copy=False),
            "replay_teacher_logits": final_scores[replay_order].astype(np.float16, copy=False),
        }

    def is_candidate_frame(self, package: Dict[str, Any], frame_record: Dict[str, Any]) -> bool:
        return len(frame_record.get("oracle_candidate_reasons", []) or []) > 0

    def process_frame(
        self,
        scenario: AbstractScenario,
        package: Dict[str, Any],
        frame_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.is_candidate_frame(package, frame_record):
            return {
                "scenario_token": scenario.token,
                "iteration": int(frame_record.get("iteration", -1)),
                "kept": 0,
                "drop_reason": "outside_oracle_candidate",
            }
        scene_manager = SceneManager(time_step=float(self.proposal_sampling.interval_length), simluate_expert_trajectory=False, use_ref_path=True)
        result = self._recover_frame_target(scenario, package, frame_record, scene_manager)
        if result is None:
            return {
                "scenario_token": scenario.token,
                "iteration": int(frame_record.get("iteration", -1)),
                "kept": 0,
                "drop_reason": "unrecoverable",
            }
        sample_dir = self._write_cache_sample(scenario, result)
        self._write_debug_plot(scenario, result, sample_dir)
        return {
            "scenario_token": scenario.token,
            "log_name": scenario.log_name,
            "scene_type": _scenario_scene_type(scenario),
            "iteration": int(frame_record["iteration"]),
            "kept": 1,
            "sample_dir": str(sample_dir),
            "state_class": result["state_class"],
            "best_anchor": int(result["best_anchor"]),
            "best_score": float(result["best_score"]),
            "oracle_candidate_reasons": list(frame_record.get("oracle_candidate_reasons", []) or []),
            "min_ttc": float(frame_record.get("min_ttc", np.inf)),
            "disagreement_reason": str(frame_record.get("disagreement_reason", "")),
        }

    def _build_road_frame_target(
        self,
        scenario: AbstractScenario,
        package: Dict[str, Any],
        frame_record: Dict[str, Any],
        scene_manager: SceneManager,
    ) -> Optional[Dict[str, Any]]:
        iteration = int(frame_record["iteration"])
        ego_states = [ego_state_from_vector(vector) for vector in package["ego_state_vectors"]]
        ego_index = int(frame_record["ego_state_index"])
        ego_state = ego_states[ego_index]
        ego_history = ego_states[: ego_index + 1]

        feature_dict, target_dict = scene_manager.extract_feature_target_from_scenario(
            scenario=scenario,
            iteration=iteration,
            ego_state=ego_state,
            ego_history=ego_history,
            use_route_correction=True,
        )
        scene_feature = SceneFeature.deserialize(feature_dict)
        agent_prediction = AgentPrediction.deserialize(target_dict)
        expert_trajectory = get_expert_trajectory_from_scenario_with_rollout_ego(
            scenario,
            rollout_ego_state=ego_state,
            run_step=iteration,
            time_horizon=float(self.proposal_sampling.time_horizon),
            time_interval=float(self.proposal_sampling.interval_length),
        )

        best_anchor = int(frame_record.get("chosen_anchor_index", -1))
        if best_anchor < 0:
            return None
        sampled_indices = np.asarray(frame_record.get("sampled_anchor_indices", []), dtype=np.int64).reshape(-1)
        sampled_scores = np.asarray(frame_record.get("sampled_selection_scores", []), dtype=np.float32).reshape(-1)
        valid_mask = (
            (sampled_indices >= 0)
            & (sampled_indices < int(self.planner_anchor.shape[0]))
            & np.isfinite(sampled_scores)
        )
        sampled_indices = sampled_indices[valid_mask]
        sampled_scores = sampled_scores[valid_mask]
        if sampled_indices.size == 0:
            sampled_indices = np.asarray([best_anchor], dtype=np.int64)
            sampled_scores = np.asarray([float(frame_record.get("chosen_selection_score", 1.0))], dtype=np.float32)

        if best_anchor not in set(int(index) for index in sampled_indices.tolist()):
            sampled_indices = np.concatenate([sampled_indices, np.asarray([best_anchor], dtype=np.int64)])
            sampled_scores = np.concatenate([sampled_scores, np.asarray([float(frame_record.get("chosen_selection_score", 1.0))], dtype=np.float32)])

        dense_scores = _build_dense_anchor_scores(
            anchor_num=int(self.planner_anchor.shape[0]),
            sampled_indices=sampled_indices,
            sampled_scores=sampled_scores,
            fill_margin=self.oracle_fill_margin,
        )
        replay_order = np.argsort(sampled_scores)[::-1][: self.oracle_replay_topk]
        return {
            "iteration": iteration,
            "state_class": self._frame_state_class(ego_state, scenario.get_ego_state_at_iteration(iteration)),
            "ego_state": ego_state,
            "ego_history": ego_history,
            "scene_feature": scene_feature,
            "agent_prediction": agent_prediction,
            "expert_trajectory": expert_trajectory,
            "best_anchor": best_anchor,
            "best_score": float(frame_record.get("chosen_selection_score", sampled_scores.max())),
            "dense_scores": dense_scores,
            "replay_anchor_indices": sampled_indices[replay_order].astype(np.int32, copy=False),
            "replay_teacher_logits": sampled_scores[replay_order].astype(np.float16, copy=False),
        }

    def process_road_frame(
        self,
        scenario: AbstractScenario,
        package: Dict[str, Any],
        frame_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        scene_manager = SceneManager(time_step=float(self.proposal_sampling.interval_length), simluate_expert_trajectory=False, use_ref_path=True)
        result = self._build_road_frame_target(scenario, package, frame_record, scene_manager)
        if result is None:
            return {
                "scenario_token": scenario.token,
                "iteration": int(frame_record.get("iteration", -1)),
                "kept": 0,
                "drop_reason": "missing_road_action",
            }
        sample_dir = self._write_cache_sample(scenario, result)
        self._write_debug_plot(scenario, result, sample_dir)
        return {
            "scenario_token": scenario.token,
            "log_name": scenario.log_name,
            "scene_type": _scenario_scene_type(scenario),
            "iteration": int(frame_record["iteration"]),
            "kept": 1,
            "sample_dir": str(sample_dir),
            "state_class": result["state_class"],
            "best_anchor": int(result["best_anchor"]),
            "best_score": float(result["best_score"]),
            "oracle_candidate_reasons": [],
            "retrieval_style": "road",
            "min_ttc": float(frame_record.get("min_ttc", np.inf)),
            "disagreement_reason": str(frame_record.get("disagreement_reason", "")),
        }

    def _load_rollout_package(self, package_path: str) -> Dict[str, Any]:
        cached = self._package_cache.get(package_path)
        if cached is not None:
            return cached
        with gzip.open(Path(package_path), "rb") as file:
            package = pickle.load(file)
        if not isinstance(package, dict):
            raise TypeError(f"Rollout package must be a dict, got {type(package)!r}: {package_path}")
        self._package_cache = {package_path: package}
        return package

    def process_frame_from_package_path(
        self,
        scenario: AbstractScenario,
        package_path: str,
        frame_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        package = self._load_rollout_package(package_path)
        return self.process_frame(scenario, package, frame_record)

    def process_road_frame_from_package_path(
        self,
        scenario: AbstractScenario,
        package_path: str,
        frame_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        package = self._load_rollout_package(package_path)
        return self.process_road_frame(scenario, package, frame_record)

    def _write_cache_sample(self, scenario: AbstractScenario, result: Dict[str, Any]) -> Path:
        scene_type = _scenario_scene_type(scenario)
        sample_name = f"{scenario.token}_iter_{int(result['iteration']):04d}"
        sample_dir = self.output_cache_root / scenario.log_name / scene_type / _safe_name(sample_name)
        sample_dir.mkdir(parents=True, exist_ok=True)
        dump_feature_target_to_pickle(sample_dir / "scene_feature.gz", result["scene_feature"].serialize())
        dump_feature_target_to_pickle(sample_dir / "agent_prediction.gz", result["agent_prediction"].serialize())
        dump_feature_target_to_pickle(sample_dir / "expert_trajectory.gz", Trajectory(data=result["expert_trajectory"]).serialize())
        dump_feature_target_to_pickle(sample_dir / self.anchor_indice_name, AnchorIndice(indice=np.asarray([result["best_anchor"]], dtype=np.int32)).serialize())
        dump_feature_target_to_pickle(sample_dir / self.anchor_score_name, AnchorScores(aggregated_scores=result["dense_scores"]).serialize())
        dump_feature_target_to_pickle(
            sample_dir / self.replay_target_name,
            ReplayPlannerTargets(
                anchor_indices=result["replay_anchor_indices"],
                teacher_logits=result["replay_teacher_logits"],
            ).serialize(),
        )
        dump_feature_target_to_pickle(sample_dir / "scenario_token.gz", SceneToken(token=f"{scenario.token}_iter_{int(result['iteration']):04d}").serialize())
        return sample_dir

    def _write_debug_plot(self, scenario: AbstractScenario, result: Dict[str, Any], sample_dir: Path) -> None:
        if not self.debug:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            draw_scene_manager = self.env.sm if self.env is not None else SceneManager(
                time_step=float(self.proposal_sampling.interval_length),
                simluate_expert_trajectory=False,
                use_ref_path=True,
            )
            debug_order = np.argsort(result["final_scores"])[::-1][: min(self.debug_rollout_topk, len(result["final_scores"]))]
            image = draw_scene_manager.draw_model_in_out(
                scene_feature=result["scene_feature"],
                agent_prediction=result["agent_prediction"],
                expert_trajectory=result["expert_trajectory"],
                rollout_subset_trajectories=result["selected_anchors"][debug_order],
                rollout_subset_scores=result["final_scores"][debug_order],
                rollout_anchor_indices=result["selected_indices"][debug_order],
            )
            out_path = self.debug_dir / f"{_safe_name(scenario.log_name)}_{_safe_name(scenario.token)}_iter_{int(result['iteration']):04d}.png"
            if hasattr(image, "save"):
                image.save(out_path)
            else:
                plt.imsave(out_path, np.asarray(image))
        except Exception as exc:
            logger.warning("Failed to render rollout debug plot for %s: %s", sample_dir, exc)

    def process_scenario(self, scenario: AbstractScenario) -> Dict[str, Any]:
        package = self.collect_rollout_package(scenario, max_steps=self._resolve_rollout_max_steps())
        scene_manager = SceneManager(time_step=float(self.proposal_sampling.interval_length), simluate_expert_trajectory=False, use_ref_path=True)
        kept = 0
        dropped = 0
        written_dirs: List[str] = []
        for frame_record in package["frame_records"]:
            if not self.is_candidate_frame(package, frame_record):
                continue
            result = self._recover_frame_target(scenario, package, frame_record, scene_manager)
            if result is None:
                dropped += 1
                continue
            sample_dir = self._write_cache_sample(scenario, result)
            self._write_debug_plot(scenario, result, sample_dir)
            written_dirs.append(str(sample_dir))
            kept += 1

        return {
            "scenario_token": scenario.token,
            "log_name": scenario.log_name,
            "scene_type": _scenario_scene_type(scenario),
            "package_path": package.get("package_path"),
            "failure_type": package.get("failure_type"),
            "termination_type": package.get("termination_type"),
            "is_failure": bool(package.get("is_failure", package.get("failure_type") != "success")),
            "failure_iteration": package.get("failure_iteration"),
            "frames": len(package["frame_records"]),
            "kept": kept,
            "dropped_unrecoverable": dropped,
            "written_dirs": written_dirs,
        }
