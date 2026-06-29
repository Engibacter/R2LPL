from typing import Deque, List, Optional, Tuple
from zipfile import Path
import os
import numpy as np
import numpy.typing as npt
from copy import deepcopy

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _se2_vel_acc_to_ego_state,
    _get_velocity_and_acceleration as _get_ml_planner_velocity_and_acceleration,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from lpl_planner.planning.scene.scene_feature.features import (
    SceneFeature,
)
from lpl_planner.planning.scene.evaluate.scene_scorer import BatchEvaluator
# from hybrid_planner.planning.scene.evaluate.utils.control_utils import StateIndex, _get_velocity_and_acceleration
from lpl_planner.planning.scene.evaluate.utils.control_utils import (
    StateIndex as SimulatorStateIndex,
    _get_velocity_and_acceleration as _estimate_simulator_velocity_and_acceleration,
    get_velocity_curvature_profiles_with_derivatives_from_poses,
)

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

def trajectory_to_interpolated_trajectory(
    trajectory: npt.NDArray[np.float32],
    ego_history: Deque[EgoState],
    future_horizon: float,
    step_interval: float,
    use_anchor_velocity: bool = False,
    debug: bool = False,
):
    ego_state = ego_history[-1]
    timesteps = _get_fixed_timesteps(ego_state, future_horizon, step_interval)
    trajectory_states = [StateSE2.deserialize(pose) for pose in trajectory[..., :3]]

    if use_anchor_velocity:
        velocity_x = trajectory[..., 3]
        velocity_y = np.zeros_like(velocity_x)  # 假设y方向速度为0
        velocities = np.concatenate([velocity_x[:, np.newaxis], velocity_y[:, np.newaxis]], axis=-1)
        acceleration_x = trajectory[..., 5]
        acceleration_y = np.zeros_like(acceleration_x)  # 假设y方向加速度为0
        accelerations = np.concatenate([acceleration_x[:, np.newaxis], acceleration_y[:, np.newaxis]], axis=-1)
    else:
        velocities, accelerations = _get_ml_planner_velocity_and_acceleration(
            trajectory_states, ego_history, timesteps
        )

    if debug:
        vel_prof, ref_vel = debug_velocity_profile_from_trajectory(trajectory, dt=0.2)
        print("Ref vel:", ref_vel)
        print("Vel profile:", vel_prof)

    ego_states = [
        _se2_vel_acc_to_ego_state(
            state,
            velocity,
            acceleration,
            timestep,
            ego_state.car_footprint.vehicle_parameters,
        )
        for state, velocity, acceleration, timestep in zip(
            trajectory_states, velocities, accelerations, timesteps
        )
    ]


    ego_states.insert(0, ego_state)

    return InterpolatedTrajectory(ego_states)

def local_trajectory_to_abstract_trajectory(
    trajectory: npt.NDArray[np.float32],
    ego_state: EgoState,
    future_horizon: float,
    step_interval: float,
) -> InterpolatedTrajectory:
    
     # convert relative trajectories to absolute trajectories
    ego_xy = ego_state.rear_axle.array
    ego_yaw = ego_state.rear_axle.heading
    rot_mat = np.array([[np.cos(ego_yaw), -np.sin(ego_yaw)],
                        [np.sin(ego_yaw),  np.cos(ego_yaw)]], dtype=float)
    trajectory_global = np.zeros_like(trajectory,dtype=np.float64)
    trajectory_global[..., :2] = trajectory[..., :2] @ rot_mat.T + ego_xy
    trajectory_global[..., 2] = trajectory[..., 2] + ego_yaw
    trajectory_global[..., 2] = np.arctan2(np.sin(trajectory_global[..., 2]), np.cos(trajectory_global[..., 2]))

    timesteps = _get_fixed_timesteps(ego_state, future_horizon, step_interval)

    trajectory_states = [StateSE2.deserialize(pose) for pose in trajectory_global[..., :3]]
    velocities = np.zeros((len(trajectory_states), 2), dtype=np.float64)
    accelerations = np.zeros((len(trajectory_states), 2), dtype=np.float64)

    ego_states = [
        _se2_vel_acc_to_ego_state(
            state,
            velocity,
            acceleration,
            timestep,
            ego_state.car_footprint.vehicle_parameters,
        )
        for state, velocity, acceleration, timestep in zip(
            trajectory_states, velocities, accelerations, timesteps
        )
    ]

    ego_states.insert(0, ego_state)

    return InterpolatedTrajectory(ego_states)

def interp_valid_yaw(yaw: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """ Interpolate invalid yaw values (NaN or inf) in the input array.
    Args:
        yaw: Array of yaw values [rad].
    Returns:
        Array of yaw values with invalid entries interpolated.
    """
    if np.any(np.isnan(yaw)) or np.any(np.isinf(yaw)):
        valid = ~np.isnan(yaw) & ~np.isinf(yaw)
        indices = np.arange(len(yaw))
        if np.any(valid):
            yaw = np.interp(indices, indices[valid], yaw[valid])
        else:
            yaw = np.zeros_like(yaw)
    return yaw

def finite_diff_yaw(xy: np.ndarray) -> np.ndarray:
    """基于相邻点差分计算 yaw，并用首尾值填充长度"""
    diff = np.diff(xy, axis=0)
    yaw = np.arctan2(diff[:, 1], diff[:, 0])
    yaw = np.concatenate([yaw[:1], yaw])   # 长度对齐
    # 修正无效
    yaw = np.where(np.isfinite(yaw), yaw, 0.0)
    return np.unwrap(yaw)

def smooth_and_limit_yaw(yaw: np.ndarray, ds: np.ndarray, yaw_rate_max: float, dt: float) -> np.ndarray:
    """限制相邻步的航向增量 |Δyaw| ≤ yaw_rate_max*dt，并做一次轻度平滑"""
    yaw_s = yaw.copy()
    max_dyaw = float(yaw_rate_max * dt)
    for i in range(1, len(yaw_s)):
        d = yaw_s[i] - yaw_s[i-1]
        d = (d + np.pi) % (2*np.pi) - np.pi
        d = np.clip(d, -max_dyaw, max_dyaw)
        yaw_s[i] = yaw_s[i-1] + d
    # 一次轻度平滑（3点均值）
    if len(yaw_s) >= 3:
        yaw_s[1:-1] = 0.25*yaw_s[:-2] + 0.5*yaw_s[1:-1] + 0.25*yaw_s[2:]
    return np.unwrap(yaw_s)

def curvature_limited_s_plan(
    s0: float,
    T: int,
    dt: float,
    v0: float,
    ref_s: np.ndarray,
    ref_theta: np.ndarray,
    a_lat_max: float = 2.5,       # m/s^2 允许的侧向加速度
    a_max: float = 2.0,           # m/s^2 纵向加速度上限
    a_min: float = -3.0           # m/s^2 纵向减速度下限(负)
) -> np.ndarray:
    """
    基于参考线曲率 κ(s) 计算限速 v<=sqrt(a_lat_max/|κ|)，并叠加纵向加/减速度约束，前馈得到 s_t。
    """
    # κ = dθ/ds（对 ref_theta 相对 ref_s 的梯度）
    # 避免数值噪声：使用 np.gradient
    kappa = np.gradient(ref_theta, ref_s, edge_order=2)
    # 将 s 映射到 κ 索引（searchsorted 最近邻）
    def idx_of(sv: float) -> int:
        j = int(np.clip(np.searchsorted(ref_s, sv, side="left"), 1, len(ref_s)-1))
        l = ref_s[j-1]; r = ref_s[j]
        return j if (abs(sv-r) < abs(sv-l)) else (j-1)

    s = np.zeros(T, dtype=np.float64)
    s[0] = s0
    v = float(v0)
    for t in range(1, T):
        j = idx_of(s[t-1])
        kap = float(abs(kappa[j]))
        v_lat = np.sqrt(a_lat_max / (kap + 1e-6))  # 曲率限速
        # 纵向约束（加/减速度）
        v = np.clip(v, 0.0, np.inf)
        v_up = v + a_max * dt
        v_dn = max(0.0, v + a_min * dt)
        v_next = min(v_lat, v_up)
        v_next = max(v_next, v_dn)
        s[t] = s[t-1] + v_next * dt
        v = v_next
    return s

def hausdorff_xy(trajs: np.ndarray, fall_back_traj: np.ndarray) -> np.ndarray:
    """
    Calculate Hausdorff distance in XY plane between multiple trajectories and a fallback trajectory.
    Args:
        trajs: [N, T, 2] array of N trajectories.
        fall_back_traj: [T, 2] array of fallback trajectory.
    Returns:
        hausdorff_distances: [N] array of Hausdorff distances.
    """
    K, T, _ = trajs.shape
    # broadcast：(K,1,T,2) 与 (1,T,2)
    a = trajs[:, None, :, :]          # (K,1,T,2)
    b = fall_back_traj[None, :, :]    # (1,T,2)
    # 构造两两差值：(K,T,T,2)
    diff = a - b[:, None, :, :]       # (K,T,T,2)
    D = np.sqrt((diff ** 2).sum(axis=-1))  # (K,T,T)
    # directed distances
    d_ab = D.min(axis=2).max(axis=1)  # (K,)
    d_ba = D.min(axis=1).max(axis=1)  # (K,)
    return np.maximum(d_ab, d_ba)     # (K,)


def moving_average_smooth(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """简单的中心移动平均滤波，首尾做 edge padding。"""
    if window <= 1:
        return signal.copy()
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    pad_width = [(0, 0)] * signal.ndim
    pad_width[-1] = (pad_left, pad_right)
    padded = np.pad(signal, pad_width, mode="edge")
    cumulative = np.cumsum(padded, axis=-1, dtype=np.float64)
    cumulative = np.concatenate(
        [np.zeros_like(cumulative[..., :1]), cumulative],
        axis=-1,
    )
    smoothed = (cumulative[..., window:] - cumulative[..., :-window]) / float(window)
    return smoothed.astype(np.result_type(signal.dtype, np.float64), copy=False)


def _estimate_simulator_longitudinal_profiles_batch(
    trajectories: np.ndarray,
    dt: float,
    init_speed: Optional[float],
    init_accel: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate velocity/acceleration for a batch with the same routine used by the evaluator."""
    num_traj, horizon = trajectories.shape[:2]
    poses = np.zeros((num_traj, horizon + 1, 3), dtype=np.float32)
    poses[:, 1:, :] = trajectories[:, :, :3].astype(np.float32, copy=False)

    current_state = np.zeros((num_traj, SimulatorStateIndex.size()), dtype=np.float32)
    current_state[:, SimulatorStateIndex.VELOCITY_X] = max(float(init_speed or 0.0), 0.0)
    current_state[:, SimulatorStateIndex.ACCELERATION_X] = float(init_accel or 0.0)

    timesteps = (np.arange(horizon, dtype=np.float32) + 1.0) * float(dt)
    velocities_ds, accelerations_ds = _estimate_simulator_velocity_and_acceleration(
        poses,
        current_state,
        timesteps,
        interp_dt=min(0.1, float(dt)),
    )
    return velocities_ds[:, :, 0], accelerations_ds[:, :, 0]


def _estimate_simulator_longitudinal_profiles(
    trajectory: np.ndarray,
    dt: float,
    init_speed: Optional[float],
    init_accel: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate velocity/acceleration with the same routine used by the batch simulator/evaluator."""
    velocities_ds, accelerations_ds = _estimate_simulator_longitudinal_profiles_batch(
        trajectory[None, ...],
        dt=dt,
        init_speed=init_speed,
        init_accel=init_accel,
    )
    return velocities_ds[0], accelerations_ds[0]


def _follow_velocity_target_batch(
    v_target: np.ndarray,
    dt: float,
    max_acc: float,
    min_acc: float,
    max_jerk: float,
    init_speed: Optional[float],
    init_accel: Optional[float],
    trajectories: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Track target speed profiles for a batch under the same acceleration/jerk constraints."""
    num_traj, horizon, state_dim = trajectories.shape
    v = np.zeros((num_traj, horizon), dtype=np.float64)
    a = np.zeros((num_traj, horizon), dtype=np.float64)

    if init_speed is not None:
        v0 = np.full((num_traj,), max(float(init_speed), 0.0), dtype=np.float64)
    elif state_dim >= 4:
        v0 = np.maximum(trajectories[:, 0, 3].astype(np.float64, copy=False), 0.0)
    else:
        v0 = np.maximum(v_target[:, 0], 0.0)

    if init_accel is not None:
        a0 = np.full(
            (num_traj,),
            float(np.clip(init_accel, min_acc, max_acc)),
            dtype=np.float64,
        )
    else:
        a0 = np.clip((v_target[:, 0] - v0) / dt, min_acc, max_acc)

    v[:, 0] = np.maximum(v0 + a0 * dt, 0.0)
    a[:, 0] = a0

    max_da = max_jerk * dt
    for step_idx in range(1, horizon):
        a_des = np.clip((v_target[:, step_idx] - v[:, step_idx - 1]) / dt, min_acc, max_acc)
        a[:, step_idx] = np.clip(
            a[:, step_idx - 1] + np.clip(a_des - a[:, step_idx - 1], -max_da, max_da),
            min_acc,
            max_acc,
        )
        v[:, step_idx] = np.maximum(v[:, step_idx - 1] + a[:, step_idx] * dt, 0.0)

    s_delta = 0.5 * (v[:, :-1] + v[:, 1:]) * dt
    s_new = np.concatenate(
        [np.zeros((num_traj, 1), dtype=np.float64), np.cumsum(s_delta, axis=1)],
        axis=1,
    )
    return v, a, s_new, v_target


def _follow_velocity_target(
    v_target: np.ndarray,
    dt: float,
    max_acc: float,
    min_acc: float,
    max_jerk: float,
    init_speed: Optional[float],
    init_accel: Optional[float],
    traj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Track a target speed profile under acceleration and jerk limits."""
    v, a, s_new, tracked_target = _follow_velocity_target_batch(
        v_target=v_target[None, :],
        dt=dt,
        max_acc=max_acc,
        min_acc=min_acc,
        max_jerk=max_jerk,
        init_speed=init_speed,
        init_accel=init_accel,
        trajectories=traj[None, ...],
    )
    return v[0], a[0], s_new[0], tracked_target[0]


def _interpolate_polyline_by_arclength(
    xy: np.ndarray,
    s_new: np.ndarray,
) -> np.ndarray:
    """Equivalent to sampling a polyline by arclength, but without per-step shapely calls."""
    segment_vec = np.diff(xy, axis=0)
    segment_len = np.linalg.norm(segment_vec, axis=1)
    cumulative_len = np.concatenate([[0.0], np.cumsum(segment_len)])
    total_len = float(cumulative_len[-1])
    clipped_s = np.clip(s_new, 0.0, total_len)

    if total_len < 1e-12 or segment_len.size == 0:
        return np.repeat(xy[:1], len(clipped_s), axis=0)

    segment_idx = np.searchsorted(cumulative_len, clipped_s, side="right") - 1
    segment_idx = np.clip(segment_idx, 0, len(segment_len) - 1)
    segment_start_s = cumulative_len[segment_idx]
    local_ratio = np.zeros_like(clipped_s, dtype=np.float64)

    nonzero_segment = segment_len[segment_idx] > 1e-12
    local_ratio[nonzero_segment] = (
        (clipped_s[nonzero_segment] - segment_start_s[nonzero_segment])
        / segment_len[segment_idx][nonzero_segment]
    )
    local_ratio = np.clip(local_ratio, 0.0, 1.0)

    return xy[segment_idx] + local_ratio[:, None] * segment_vec[segment_idx]


def _rebuild_trajectory_from_arclength(
    traj: np.ndarray,
    s_new: np.ndarray,
) -> np.ndarray:
    """Project a new arclength profile back to the original geometric path."""
    T, D = traj.shape
    new_traj = traj.copy()
    new_xy = _interpolate_polyline_by_arclength(traj[:, :2], s_new)

    diff_xy = np.diff(new_xy, axis=0, prepend=new_xy[0:1, :])
    yaw = np.arctan2(diff_xy[:, 1], diff_xy[:, 0])
    yaw = interp_valid_yaw(yaw)
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

    new_traj[:, 0:2] = new_xy
    if D >= 3:
        new_traj[:, 2] = yaw

    return new_traj


def _rebuild_trajectories_from_arclength_batch(
    trajectories: np.ndarray,
    s_new: np.ndarray,
) -> np.ndarray:
    """Project a batch of arclength profiles back to their original geometric polylines."""
    rebuilt = trajectories.copy()
    for traj_idx in range(trajectories.shape[0]):
        rebuilt[traj_idx] = _rebuild_trajectory_from_arclength(
            trajectories[traj_idx],
            s_new[traj_idx],
        )
    return rebuilt


def _smooth_trajectories_core(
    trajectories: np.ndarray,
    dt: float,
    max_acc: float = 2.4,
    min_acc: float = -4.0,
    max_jerk: float = 4.0,
    window: int = 5,
    init_speed: Optional[float] = None,
    init_accel: Optional[float] = None,
    simulator_acc_margin: float = 0.10,
    max_refine_iters: int = 4,
) -> np.ndarray:
    """Batch implementation of the smoothing pipeline with the same per-trajectory logic."""
    num_traj, horizon, state_dim = trajectories.shape
    if horizon < 3 or num_traj == 0:
        return trajectories.copy()

    smoothed_trajectories = trajectories.copy()
    xy = trajectories[:, :, :2].astype(np.float64, copy=False)
    segment_len = np.linalg.norm(np.diff(xy, axis=1), axis=2)
    line_length = segment_len.sum(axis=1)
    valid_mask = line_length >= 1e-3
    if not np.any(valid_mask):
        return smoothed_trajectories

    valid_trajectories = trajectories[valid_mask].astype(np.float64, copy=True)
    valid_xy = valid_trajectories[:, :, :2]
    valid_segment_len = segment_len[valid_mask]
    ds = np.concatenate(
        [np.zeros((valid_trajectories.shape[0], 1), dtype=np.float64), valid_segment_len],
        axis=1,
    )
    s_geom = np.cumsum(ds, axis=1)
    v_ref = np.maximum(np.gradient(s_geom, dt, axis=1), 0.0)

    if init_speed is not None:
        v_pre = max(float(init_speed), 0.0)
        if init_accel is not None:
            v_pre = max(float(init_speed) + float(init_accel) * dt, 0.0)
        v_ref_smooth_in = np.concatenate(
            [np.full((valid_trajectories.shape[0], 1), v_pre, dtype=np.float64), v_ref],
            axis=1,
        )
    else:
        v_ref_smooth_in = v_ref

    v_target_full = moving_average_smooth(v_ref_smooth_in, window=window)
    v_target = v_target_full[:, 1:] if v_ref_smooth_in.shape[1] > v_ref.shape[1] else v_target_full
    v_target = np.maximum(v_target, 0.0)
    v_target = moving_average_smooth(v_target, window=max(window, 5))
    v_target = moving_average_smooth(v_target, window=max(window - 2, 3))
    v_target = np.maximum(v_target, 0.0)

    safety_max_acc = max(max_acc - simulator_acc_margin, 0.5)
    safety_min_acc = min_acc + simulator_acc_margin

    v, _, s_new, v_target_cap = _follow_velocity_target_batch(
        v_target=v_target,
        dt=dt,
        max_acc=safety_max_acc,
        min_acc=safety_min_acc,
        max_jerk=max_jerk,
        init_speed=init_speed,
        init_accel=init_accel,
        trajectories=valid_trajectories,
    )
    rebuilt = _rebuild_trajectories_from_arclength_batch(valid_trajectories, s_new)

    for _ in range(max_refine_iters):
        sim_vel, sim_acc = _estimate_simulator_longitudinal_profiles_batch(
            rebuilt,
            dt=dt,
            init_speed=init_speed,
            init_accel=init_accel,
        )
        future_acc = sim_acc[:, 1:]
        if future_acc.size == 0:
            break

        violating = (np.max(future_acc, axis=1) > max_acc) | (np.min(future_acc, axis=1) < min_acc)
        if not np.any(violating):
            break

        corrected_target = v[violating].copy()
        sim_future_vel = np.maximum(sim_vel[violating, 1:], 0.0)
        corrected_target = np.minimum(corrected_target, sim_future_vel)

        for step_idx in range(1, horizon):
            max_next = corrected_target[:, step_idx - 1] + safety_max_acc * dt
            min_next = np.maximum(0.0, corrected_target[:, step_idx - 1] + safety_min_acc * dt)
            corrected_target[:, step_idx] = np.clip(corrected_target[:, step_idx], min_next, max_next)

        corrected_target = moving_average_smooth(corrected_target, window=max(window, 5))
        corrected_target = np.minimum(corrected_target, v_target_cap[violating])
        corrected_target = np.maximum(corrected_target, 0.0)

        refined_v, _, refined_s_new, _ = _follow_velocity_target_batch(
            v_target=corrected_target,
            dt=dt,
            max_acc=safety_max_acc,
            min_acc=safety_min_acc,
            max_jerk=max_jerk,
            init_speed=init_speed,
            init_accel=init_accel,
            trajectories=valid_trajectories[violating],
        )
        rebuilt[violating] = _rebuild_trajectories_from_arclength_batch(
            valid_trajectories[violating],
            refined_s_new,
        )
        v[violating] = refined_v

    final_sim_vel, final_sim_acc = _estimate_simulator_longitudinal_profiles_batch(
        rebuilt,
        dt=dt,
        init_speed=init_speed,
        init_accel=init_accel,
    )
    rebuilt[:, :, :3] = rebuilt[:, :, :3]
    if state_dim >= 4:
        rebuilt[:, :, 3] = np.maximum(final_sim_vel[:, 1:], 0.0)
    if state_dim >= 6:
        rebuilt[:, :, 5] = final_sim_acc[:, 1:]

    smoothed_trajectories[valid_mask] = rebuilt.astype(trajectories.dtype, copy=False)
    return smoothed_trajectories


def _smooth_trajectory(
    traj: np.ndarray,
    dt: float,
    max_acc: float = 2.4,
    min_acc: float = -4.0,
    max_jerk: float = 6.0,
    window: int = 5,
    init_speed: Optional[float] = None,
    init_accel: Optional[float] = None,
    simulator_acc_margin: float = 0.10,
    max_refine_iters: int = 4,
) -> np.ndarray:
    """
    单条轨迹平滑：
    - 用几何弧长 s(t) 得到原始 v_ref(t)
    - 对 v_ref 做移动平均得到 v_target(t)
    - 从 (init_speed, init_accel) 出发，在 |a|、jerk 限制下跟踪 v_target
    - 用新 v(t) 重新积分得到 s_new(t)，在 LineString 上按弧长插值 (x, y)
    - 基于新几何重新计算 yaw, v, a
    """
    return _smooth_trajectories_core(
        traj[None, ...],
        dt=dt,
        max_acc=max_acc,
        min_acc=min_acc,
        max_jerk=max_jerk,
        window=window,
        init_speed=init_speed,
        init_accel=init_accel,
        simulator_acc_margin=simulator_acc_margin,
        max_refine_iters=max_refine_iters,
    )[0]


def smooth_trajectories(
    trajectories: np.ndarray,
    dt: float,
    init_speed: Optional[float] = None,
    init_accel: Optional[float] = None,
) -> np.ndarray:
    """
    对多条轨迹做纵向平滑/限幅：
    - 用几何弧长得到原始 v_ref(t)
    - 对 v_ref 做移动平均去掉 anchor 拼接造成的速度跳变
    - 在 |a|、jerk 约束下从 (init_speed, init_accel) 跟踪平滑后的目标速度
    - 重算 (x, y, yaw, v, a)，保持几何尽量接近原始轨迹
    """
    return _smooth_trajectories_core(
        trajectories,
        dt=dt,
        init_speed=init_speed,
        init_accel=init_accel,
    )

def draw_scorer_frames(
    scene_feature: SceneFeature,
    scorer: BatchEvaluator,
    proposal_idx: int,
    chosen_trajectory: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    frame_idx: int = 0,
) -> List[np.ndarray]:
    """Draw the scorer frames with infraction information at each plot time index."""

    plt.ioff()
    proposal_sampling = scorer._proposal_sampling
    img_list: List[np.ndarray] = []

    plot_interval = 1.0  # seconds
    step = max(1, int(plot_interval / proposal_sampling.interval_length))
    plot_indices = np.arange(0, proposal_sampling.num_poses + 1, step)

    # load features
    road_feature = scene_feature.road_feature
    ref_path_feature = scene_feature.ref_path_feature
    ego_feature = scene_feature.ego_feature

    # ensure save_path exists
    if save_path is not None:
        os.makedirs(str(save_path), exist_ok=True)

    for time_idx in plot_indices:
        fig, axes = plt.subplots(figsize=(10, 10))
        axes.set_aspect('equal')
        axes.set(xlim=(-30, 50))
        axes.set(ylim=(-40, 40))

        # 绘制道路中心线
        for center_line in road_feature.center_line:
            center_line = np.array(center_line)
            if center_line.shape[0] > 1:
                axes.plot(
                    center_line[:, 0],
                    center_line[:, 1],
                    color='gray',
                    linestyle='--',
                    linewidth=1,
                    alpha=0.7,
                )

        # 绘制道路多边形
        for polygon in road_feature.road_geometry:
            polygon = np.array(polygon)
            if polygon.shape[0] > 2:
                axes.fill(
                    polygon[:, 0],
                    polygon[:, 1],
                    color='lightgray',
                    alpha=0.3,
                    edgecolor='gray',
                )

        # 绘制选中轨迹
        if chosen_trajectory is not None:
            axes.plot(
                chosen_trajectory[:, 0],
                chosen_trajectory[:, 1],
                color='red',
                linewidth=2,
                alpha=0.8,
                label='Chosen Trajectory',
            )

        # 绘制当前本车多边形（t=0，在局部坐标原点）
        ego_pose = np.array([0.0, 0.0, 0.0])
        ego_half_width = float(ego_feature.ego_geometry[0])
        ego_half_length = float(ego_feature.ego_geometry[1])
        headings = ego_pose[2]
        cos, sin = np.cos(headings), np.sin(headings)

        rear_axle_to_center = 1.461
        ego_pose[:2] = np.array(
            [rear_axle_to_center * cos, rear_axle_to_center * sin]
        )

        ego_poly = np.array(
            [
                [ego_pose[0] - ego_half_length, ego_pose[1] - ego_half_width],
                [ego_pose[0] - ego_half_length, ego_pose[1] + ego_half_width],
                [ego_pose[0] + ego_half_length, ego_pose[1] + ego_half_width],
                [ego_pose[0] + ego_half_length, ego_pose[1] - ego_half_width],
            ]
        )
        axes.fill(
            ego_poly[:, 0],
            ego_poly[:, 1],
            color='blue',
            alpha=0.5,
            label='Ego Vehicle',
            edgecolor='blue',
        )

        # 绘制该时刻的本车预测多边形
        if (
            scorer._ego_coords is not None
            and time_idx < scorer._ego_coords.shape[1]
        ):
            ego_future_polyon = scorer._ego_coords[proposal_idx, time_idx, ...].copy()
            # 只用四个角点闭合多边形（0: FL,1:FR,2:RR,3:RL; 4: center）
            corners = ego_future_polyon[:4, :]
            corners = np.vstack([corners, corners[0:1, :]])
            axes.fill(
                corners[:, 0],
                corners[:, 1],
                color='blue',
                alpha=0.3,
                edgecolor='blue',
            )

        # 绘制周围障碍物多边形（来自 future_collision_map）
        if (
            scorer._future_collision_map is not None
            and time_idx < len(scorer._future_collision_map)
        ):
            surrounding_obj_polygons = scorer._future_collision_map[
                time_idx
            ].polygons_for_plot()
            for poly in surrounding_obj_polygons:
                poly = np.array(poly)
                axes.fill(
                    poly[0],
                    poly[1],
                    color='orange',
                    alpha=0.5,
                    edgecolor='orange',
                )

        # 绘制参考路径及其左右边界
        ref_path = ref_path_feature.numpy()
        lane_bound = ref_path[:, 3:5]
        left_bound = np.expand_dims(lane_bound[:, 0], axis=1) * np.vstack(
            (
                np.cos(ref_path[:, 2] + np.pi / 2),
                np.sin(ref_path[:, 2] + np.pi / 2),
            )
        ).T + ref_path[:, :2]
        right_bound = np.expand_dims(lane_bound[:, 1], axis=1) * np.vstack(
            (
                np.cos(ref_path[:, 2] + np.pi / 2),
                np.sin(ref_path[:, 2] + np.pi / 2),
            )
        ).T + ref_path[:, :2]

        axes.plot(
            ref_path[:, 0],
            ref_path[:, 1],
            color='green',
            linestyle='-',
            linewidth=2,
            alpha=0.5,
        )

        ref_path_poly = np.vstack([left_bound, right_bound[::-1]])
        axes.fill(
            ref_path_poly[:, 0],
            ref_path_poly[:, 1],
            color='green',
            alpha=0.2,
        )

        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
            int(height), int(width), 3
        )
        img_list.append(img)

        # 保存到文件，以 time_idx 命名
        if save_path is not None:
            out_file = os.path.join(str(save_path), f"frame{frame_idx}_{int(time_idx)}.png")
            fig.savefig(out_file, dpi=150, bbox_inches='tight')

        plt.close(fig)

    return img_list


def debug_velocity_profile_from_trajectory(
    trajectory: np.ndarray,
    dt: float,
    jerk_penalty: float = 1e-4,
    curvature_rate_penalty: float = 1e-2,
    tracking_horizon: int = 5,
    current_index: int = 0,
) -> Tuple[np.ndarray, float]:
    """
    基于 control_utils 里的同一套拟合逻辑，从单条 trajectory 计算 velocity_profile，
    并给出和 BatchLQRTracker 中 reference_velocity 相同定义的参考速度。
    Args:
        trajectory: [T, D]，至少前 3 维为 [x, y, heading]
        dt: 采样时间，对应 BatchLQRTracker.discretization_time
        jerk_penalty, curvature_rate_penalty: 和控制里保持一致即可
        tracking_horizon: LQR 的 tracking_horizon
        current_index: 当前时刻在 trajectory 里的索引（一般从 0 开始）
    Returns:
        velocity_profile: [T] 拟合得到的速度轨迹
        reference_velocity: 标量，等价于控制里用的 reference_velocity
    """
    assert trajectory.ndim == 2, "trajectory 应为 [T, D]"
    assert trajectory.shape[1] >= 3, "trajectory 至少要有 [x, y, heading] 三列"

    poses = trajectory[:, :3].astype(np.float64)      # [T, 3]
    poses_batched = poses[None, ...]                  # [1, T, 3]

    vel_prof, acc_prof, curv_prof, curv_rate_prof = get_velocity_curvature_profiles_with_derivatives_from_poses(
        discretization_time=dt,
        poses=poses_batched,
        jerk_penalty=jerk_penalty,
        curvature_rate_penalty=curvature_rate_penalty,
    )

    vel_prof = vel_prof[0]        # [T]
    T = vel_prof.shape[0]

    # 和 BatchLQRTracker._compute_reference_velocity_and_curvature_profile 保持一致
    ref_idx = min(current_index + tracking_horizon, T - 1)
    ref_vel = float(vel_prof[ref_idx])

    return vel_prof, ref_vel