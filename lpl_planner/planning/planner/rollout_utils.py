from typing import List, Optional, Tuple

import numpy as np


PLANNER_NUMPY_DTYPE = np.float32
GLOBAL_COORD_DTYPE = np.float64
ORACLE_LONGITUDINAL_INVALID_PENALTY = 1e3
ORACLE_ENDPOINT_INVALID_PENALTY = 1e5
ORACLE_ENDPOINT_PATH_MAX_ERROR = 0.5
ORACLE_PATH_MEAN_LATERAL_MAX_ERROR = 0.6
ORACLE_PATH_MAX_LATERAL_MAX_ERROR = 1.5
ORACLE_PATH_MEAN_YAW_MAX_ERROR = 0.3
ORACLE_PATH_TRACK_INVALID_PENALTY = 2e4
ORACLE_PROGRESS_CAP_INVALID_PENALTY = 5e3
ORACLE_EXPERT_PROGRESS_MAX_MARGIN = 1.5
ORACLE_MAX_LONGITUDINAL_ACCEL = 2.0
ORACLE_MAX_LONGITUDINAL_DECEL = 4.0
ORACLE_MAX_LONGITUDINAL_JERK = 4.0


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def compute_checkpoint_path_distance(
    anchors_xy: np.ndarray,
    target_xy: np.ndarray,
) -> np.ndarray:
    checkpoint_num = min(4, target_xy.shape[0])
    checkpoint_idx = np.linspace(0, target_xy.shape[0] - 1, checkpoint_num, dtype=np.int64)
    checkpoint_idx = np.unique(checkpoint_idx)
    return np.mean(
        np.linalg.norm(anchors_xy[:, checkpoint_idx, :] - target_xy[checkpoint_idx][None, :, :], axis=-1),
        axis=1,
    )


def compute_nearest_path_distance(
    anchors_xy: np.ndarray,
    target_xy: np.ndarray,
    checkpoint_num: int = 8,
) -> np.ndarray:
    if target_xy.shape[0] == 0:
        return np.zeros((anchors_xy.shape[0],), dtype=GLOBAL_COORD_DTYPE)
    checkpoint_num = min(max(checkpoint_num, 2), anchors_xy.shape[1])
    checkpoint_idx = np.linspace(0, anchors_xy.shape[1] - 1, checkpoint_num, dtype=np.int64)
    checkpoint_idx = np.unique(checkpoint_idx)
    anchor_ckpt = anchors_xy[:, checkpoint_idx, :]
    pairwise = np.linalg.norm(anchor_ckpt[:, :, None, :] - target_xy[None, None, :, :], axis=-1)
    return np.min(pairwise, axis=-1).mean(axis=-1)


def sample_path_to_length(path_xyyaw: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        return np.zeros((0, path_xyyaw.shape[1]), dtype=GLOBAL_COORD_DTYPE)
    path_xyyaw = np.asarray(path_xyyaw, dtype=GLOBAL_COORD_DTYPE)
    if path_xyyaw.shape[0] == target_len:
        return path_xyyaw
    if path_xyyaw.shape[0] == 0:
        return np.zeros((target_len, path_xyyaw.shape[1]), dtype=GLOBAL_COORD_DTYPE)
    sample_idx = np.linspace(0, path_xyyaw.shape[0] - 1, target_len, dtype=np.int64)
    return path_xyyaw[sample_idx]


def align_path_to_start_pose(path_xyyaw: np.ndarray) -> np.ndarray:
    path_xyyaw = np.asarray(path_xyyaw, dtype=GLOBAL_COORD_DTYPE)
    if path_xyyaw.shape[0] == 0:
        return path_xyyaw.copy()
    start_xy = path_xyyaw[0, :2]
    start_yaw = float(path_xyyaw[0, 2])
    rotation = np.array(
        [[np.cos(start_yaw), np.sin(start_yaw)], [-np.sin(start_yaw), np.cos(start_yaw)]],
        dtype=GLOBAL_COORD_DTYPE,
    )
    aligned = path_xyyaw.copy()
    aligned[:, :2] = (aligned[:, :2] - start_xy) @ rotation.T
    aligned[:, 2] = wrap_angle(aligned[:, 2] - start_yaw)
    return aligned


def align_paths_to_start_pose(paths_xyyaw: np.ndarray) -> np.ndarray:
    paths_xyyaw = np.asarray(paths_xyyaw, dtype=GLOBAL_COORD_DTYPE)
    if paths_xyyaw.shape[0] == 0:
        return paths_xyyaw.copy()
    start_xy = paths_xyyaw[:, :1, :2]
    start_yaw = paths_xyyaw[:, 0, 2]
    cos_yaw = np.cos(start_yaw)
    sin_yaw = np.sin(start_yaw)
    delta_xy = paths_xyyaw[:, :, :2] - start_xy
    aligned_xy = np.empty_like(delta_xy, dtype=GLOBAL_COORD_DTYPE)
    aligned_xy[:, :, 0] = delta_xy[:, :, 0] * cos_yaw[:, None] + delta_xy[:, :, 1] * sin_yaw[:, None]
    aligned_xy[:, :, 1] = -delta_xy[:, :, 0] * sin_yaw[:, None] + delta_xy[:, :, 1] * cos_yaw[:, None]
    aligned = paths_xyyaw.copy()
    aligned[:, :, :2] = aligned_xy
    aligned[:, :, 2] = wrap_angle(aligned[:, :, 2] - start_yaw[:, None])
    return aligned


def compute_shape_prefilter_score(
    anchors_xyyaw: np.ndarray,
    target_xyyaw: np.ndarray,
) -> np.ndarray:
    anchors = np.asarray(anchors_xyyaw, dtype=GLOBAL_COORD_DTYPE)
    target = np.asarray(target_xyyaw, dtype=GLOBAL_COORD_DTYPE)
    if anchors.shape[0] == 0:
        return np.zeros((0,), dtype=GLOBAL_COORD_DTYPE)
    if target.shape[0] == 0 or anchors.shape[1] == 0:
        return np.zeros((anchors.shape[0],), dtype=GLOBAL_COORD_DTYPE)
    if target.shape[0] != anchors.shape[1]:
        target = sample_path_to_length(target, anchors.shape[1])

    target_end_y = target[-1, 1]
    target_end_yaw = float(wrap_angle(target[-1, 2] - target[0, 2]))
    target_path_len = float(np.sum(np.linalg.norm(np.diff(target[:, :2], axis=0), axis=-1)))
    target_turn_energy = float(np.sum(np.abs(wrap_angle(np.diff(target[:, 2])))))

    anchor_end_y = anchors[:, -1, 1]
    anchor_end_yaw = wrap_angle(anchors[:, -1, 2] - anchors[:, 0, 2])
    anchor_path_len = np.sum(np.linalg.norm(np.diff(anchors[:, :, :2], axis=1), axis=-1), axis=1)
    anchor_turn_energy = np.sum(np.abs(wrap_angle(np.diff(anchors[:, :, 2], axis=1))), axis=1)

    checkpoint_ids = np.array([
        max(1, int(0.25 * (target.shape[0] - 1))),
        max(1, int(0.50 * (target.shape[0] - 1))),
        max(1, int(0.75 * (target.shape[0] - 1))),
    ], dtype=np.int64)
    checkpoint_ids = np.unique(np.clip(checkpoint_ids, 0, target.shape[0] - 1))

    target_y_ckpt = target[checkpoint_ids, 1]
    target_yaw_ckpt = target[checkpoint_ids, 2]
    anchor_y_ckpt = anchors[:, checkpoint_ids, 1]
    anchor_yaw_ckpt = anchors[:, checkpoint_ids, 2]

    y_ckpt_cost = np.mean(np.abs(anchor_y_ckpt - target_y_ckpt[None, :]), axis=1)
    yaw_ckpt_cost = np.mean(np.abs(wrap_angle(anchor_yaw_ckpt - target_yaw_ckpt[None, :])), axis=1)

    turn_ratio = np.clip(abs(target_end_yaw) / 0.35, 0.0, 1.0)
    w_end_y = 3.0 - 1.8 * turn_ratio
    w_ckpt_y = 2.0 - 1.0 * turn_ratio
    w_end_yaw = 2.5 + 1.0 * turn_ratio
    w_turn = 1.5 + 0.8 * turn_ratio
    w_path = 0.5

    return (
        w_end_y * np.abs(anchor_end_y - target_end_y)
        + w_end_yaw * np.abs(wrap_angle(anchor_end_yaw - target_end_yaw))
        + w_turn * np.abs(anchor_turn_energy - target_turn_energy)
        + w_path * np.abs(anchor_path_len - target_path_len)
        + w_ckpt_y * y_ckpt_cost
        + 1.0 * yaw_ckpt_cost
    )


def compute_tail_path_distance(
    anchors_xy: np.ndarray,
    target_xy: np.ndarray,
    tail_ratio: float = 0.4,
) -> np.ndarray:
    if anchors_xy.shape[0] == 0 or target_xy.shape[0] == 0:
        return np.zeros((anchors_xy.shape[0],), dtype=GLOBAL_COORD_DTYPE)
    tail_start = max(int(np.floor((1.0 - tail_ratio) * anchors_xy.shape[1])), 0)
    tail_xy = anchors_xy[:, tail_start:, :]
    pairwise = np.linalg.norm(tail_xy[:, :, None, :] - target_xy[None, None, :, :], axis=-1)
    return np.min(pairwise, axis=-1).mean(axis=-1)


def compute_endpoint_path_distance(anchor_end_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    if anchor_end_xy.shape[0] == 0 or target_xy.shape[0] == 0:
        return np.zeros((anchor_end_xy.shape[0],), dtype=GLOBAL_COORD_DTYPE)
    return np.min(np.linalg.norm(anchor_end_xy[:, None, :] - target_xy[None, :, :], axis=-1), axis=1)


def append_unique_topk_indices(
    selected: List[int],
    used: set[int],
    ordered_indices: np.ndarray,
    topk: int,
    valid_mask: Optional[np.ndarray] = None,
) -> None:
    added = 0
    for idx in np.asarray(ordered_indices, dtype=np.int64).tolist():
        if valid_mask is not None and not bool(valid_mask[idx]):
            continue
        if idx in used:
            continue
        selected.append(int(idx))
        used.add(int(idx))
        added += 1
        if added >= topk:
            break


def huber(x: np.ndarray, delta: float = 1.0) -> np.ndarray:
    abs_x = np.abs(x)
    return np.where(abs_x <= delta, 0.5 * abs_x ** 2, delta * (abs_x - 0.5 * delta))


def resample_trajectories_by_arclength_batch(
    trajectories: np.ndarray,
    target_s: np.ndarray,
) -> np.ndarray:
    batch_size, point_num, dim = trajectories.shape
    xy = trajectories[:, :, :2]

    seg_len = np.linalg.norm(np.diff(xy, axis=1), axis=-1)
    seg_len = np.maximum(seg_len, 1e-3)
    cum_s = np.concatenate(
        [np.zeros((batch_size, 1), dtype=GLOBAL_COORD_DTYPE), np.cumsum(seg_len, axis=1)],
        axis=1,
    )
    max_s = cum_s[:, -1:]
    target_s = np.clip(np.asarray(target_s, dtype=GLOBAL_COORD_DTYPE), 0.0, max_s)

    right = np.sum(cum_s[:, :, None] < target_s[:, None, :], axis=1)
    right = np.clip(right, 1, point_num - 1)
    left = right - 1

    batch_idx = np.arange(batch_size)[:, None]
    s_left = cum_s[batch_idx, left]
    s_right = cum_s[batch_idx, right]
    denom = np.maximum(s_right - s_left, 1e-6)
    weight = ((target_s - s_left) / denom)[..., None]

    traj_left = trajectories[batch_idx, left]
    traj_right = trajectories[batch_idx, right]
    resampled = traj_left + weight * (traj_right - traj_left)

    yaw = np.unwrap(trajectories[:, :, 2], axis=1)
    yaw_left = yaw[batch_idx, left]
    yaw_right = yaw[batch_idx, right]
    yaw_interp = yaw_left + (target_s - s_left) / denom * (yaw_right - yaw_left)
    resampled[:, :, 2] = wrap_angle(yaw_interp)
    if dim > 3:
        resampled[:, :, 3:] = traj_left[:, :, 3:] + weight * (traj_right[:, :, 3:] - traj_left[:, :, 3:])

    return resampled


def compute_expert_frame_errors_batch(
    expert_traj: np.ndarray,
    candidate_trajs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    expert_traj = np.asarray(expert_traj, dtype=GLOBAL_COORD_DTYPE)
    candidate_trajs = np.asarray(candidate_trajs, dtype=GLOBAL_COORD_DTYPE)
    if expert_traj.ndim == 2:
        expert_traj = np.repeat(expert_traj[None, :, :], candidate_trajs.shape[0], axis=0)

    expert_xy = expert_traj[:, :, :2]
    tangent = np.zeros_like(expert_xy)
    tangent[:, 1:-1] = expert_xy[:, 2:] - expert_xy[:, :-2]
    tangent[:, 0] = expert_xy[:, 1] - expert_xy[:, 0]
    tangent[:, -1] = expert_xy[:, -1] - expert_xy[:, -2]

    tangent_norm = np.linalg.norm(tangent, axis=2, keepdims=True)
    tangent_norm = np.maximum(tangent_norm, 1e-6)
    tangent = tangent / tangent_norm
    normal = np.stack([-tangent[:, :, 1], tangent[:, :, 0]], axis=-1)

    delta_xy = candidate_trajs[:, :, :2] - expert_traj[:, :, :2]
    e_lon = np.sum(delta_xy * tangent, axis=-1)
    e_lat = np.sum(delta_xy * normal, axis=-1)
    e_yaw = wrap_angle(candidate_trajs[:, :, 2] - expert_traj[:, :, 2])
    return e_lon, e_lat, e_yaw


def compute_path_frame_teacher_score(
    anchors_xyyaw: np.ndarray,
    target_xyyaw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    anchors_xyyaw = np.asarray(anchors_xyyaw, dtype=GLOBAL_COORD_DTYPE)
    target_xyyaw = np.asarray(target_xyyaw, dtype=GLOBAL_COORD_DTYPE)
    if anchors_xyyaw.shape[0] == 0 or anchors_xyyaw.shape[1] < 2 or target_xyyaw.shape[0] < 2:
        return np.zeros((anchors_xyyaw.shape[0],), dtype=GLOBAL_COORD_DTYPE), np.zeros((anchors_xyyaw.shape[0],), dtype=bool)

    num_resample = max(2 * max(anchors_xyyaw.shape[1], target_xyyaw.shape[0]), 32)
    target_seg_len = np.linalg.norm(np.diff(target_xyyaw[:, :2], axis=0), axis=-1)
    target_total_len = max(float(np.sum(np.maximum(target_seg_len, 1e-3))), 1e-3)

    anchor_seg_len = np.linalg.norm(np.diff(anchors_xyyaw[:, :, :2], axis=1), axis=-1)
    anchor_seg_len = np.maximum(anchor_seg_len, 1e-3)
    anchor_total_len = np.sum(anchor_seg_len, axis=1)
    target_s = np.linspace(0.0, target_total_len, num_resample, dtype=GLOBAL_COORD_DTYPE)[None, :]
    target_s = np.repeat(target_s, anchors_xyyaw.shape[0], axis=0)

    anchor_rs = resample_trajectories_by_arclength_batch(anchors_xyyaw, target_s)
    target_batch = np.repeat(target_xyyaw[None, :, :], anchors_xyyaw.shape[0], axis=0)
    target_rs = resample_trajectories_by_arclength_batch(target_batch, target_s)
    e_lon, e_lat, e_yaw = compute_expert_frame_errors_batch(target_rs, anchor_rs)

    weights = np.linspace(0.8, 1.2, num_resample, dtype=GLOBAL_COORD_DTYPE)
    weights[-max(4, num_resample // 6):] *= 2.0
    weights = weights / np.maximum(weights.sum(), 1e-6)

    lat_cost = huber(e_lat / 0.60, delta=1.0)
    lon_cost = huber(e_lon / 3.50, delta=1.0)
    yaw_cost = huber((2.0 * np.sin(0.5 * e_yaw)) / 0.20, delta=1.0)

    point_cost = (
        3.2 * np.sum(lat_cost * weights[None, :], axis=1)
        + 0.25 * np.sum(lon_cost * weights[None, :], axis=1)
        + 2.4 * np.sum(yaw_cost * weights[None, :], axis=1)
    )
    path_length_gap = np.maximum(target_total_len - anchor_total_len, 0.0)
    terminal_cost = 4.0 * np.abs(e_lat[:, -1]) + 0.3 * np.abs(e_lon[:, -1]) + 3.0 * np.abs(e_yaw[:, -1])
    score = point_cost + terminal_cost + 1.25 * path_length_gap

    abs_lat = np.abs(e_lat)
    abs_yaw = np.abs(e_yaw)
    valid_mask = (
        (np.mean(abs_lat, axis=1) <= ORACLE_PATH_MEAN_LATERAL_MAX_ERROR)
        & (np.max(abs_lat, axis=1) <= ORACLE_PATH_MAX_LATERAL_MAX_ERROR)
        & (np.mean(abs_yaw, axis=1) <= ORACLE_PATH_MEAN_YAW_MAX_ERROR)
        & (anchor_total_len >= 0.80 * target_total_len)
    )
    return score, valid_mask


def project_points_to_path_progress(
    points_xy: np.ndarray,
    path_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    points_xy = np.asarray(points_xy, dtype=GLOBAL_COORD_DTYPE)
    path_xy = np.asarray(path_xy, dtype=GLOBAL_COORD_DTYPE)
    if points_xy.ndim == 1:
        points_xy = points_xy[None, :]
    if path_xy.shape[0] < 2 or points_xy.shape[0] == 0:
        return (
            np.zeros((points_xy.shape[0],), dtype=GLOBAL_COORD_DTYPE),
            np.full((points_xy.shape[0],), np.inf, dtype=GLOBAL_COORD_DTYPE),
        )

    seg_start = path_xy[:-1]
    seg_end = path_xy[1:]
    seg_vec = seg_end - seg_start
    seg_len = np.linalg.norm(seg_vec, axis=1)
    seg_len = np.maximum(seg_len, 1e-6)
    seg_len_sq = seg_len ** 2
    cum_s = np.concatenate(([0.0], np.cumsum(seg_len)))

    rel = points_xy[:, None, :] - seg_start[None, :, :]
    proj_t = np.sum(rel * seg_vec[None, :, :], axis=-1) / seg_len_sq[None, :]
    proj_t = np.clip(proj_t, 0.0, 1.0)
    proj_xy = seg_start[None, :, :] + proj_t[:, :, None] * seg_vec[None, :, :]
    proj_dist = np.linalg.norm(points_xy[:, None, :] - proj_xy, axis=-1)
    best_seg = np.argmin(proj_dist, axis=1)
    best_t = proj_t[np.arange(points_xy.shape[0]), best_seg]
    best_progress = cum_s[best_seg] + best_t * seg_len[best_seg]
    best_dist = proj_dist[np.arange(points_xy.shape[0]), best_seg]
    return best_progress.astype(GLOBAL_COORD_DTYPE, copy=False), best_dist.astype(GLOBAL_COORD_DTYPE, copy=False)


def compute_expert_progress_cap(
    anchors_xyyaw: np.ndarray,
    route_ref_path: np.ndarray,
    expert_future_local: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    route_xy = np.asarray(route_ref_path[:, :2], dtype=GLOBAL_COORD_DTYPE)
    if route_xy.shape[0] < 2 or expert_future_local.shape[0] == 0:
        return (
            np.zeros((anchors_xyyaw.shape[0],), dtype=GLOBAL_COORD_DTYPE),
            np.ones((anchors_xyyaw.shape[0],), dtype=bool),
            float("inf"),
        )

    expert_endpoint = np.asarray(expert_future_local[-1, :2], dtype=GLOBAL_COORD_DTYPE)
    expert_max_progress, _ = project_points_to_path_progress(expert_endpoint, route_xy)
    expert_max_progress = float(expert_max_progress[0])

    anchor_endpoint_progress, _ = project_points_to_path_progress(np.asarray(anchors_xyyaw[:, -1, :2], dtype=GLOBAL_COORD_DTYPE), route_xy)
    progress_overshoot = np.maximum(anchor_endpoint_progress - expert_max_progress, 0.0)
    progress_valid_mask = anchor_endpoint_progress <= (expert_max_progress + ORACLE_EXPERT_PROGRESS_MAX_MARGIN)
    return progress_overshoot, progress_valid_mask, expert_max_progress


def compute_longitudinal_distance_scores(
    anchors: np.ndarray,
    current_speed: float,
    future_interval_length: float,
    brake_like: bool,
) -> np.ndarray:
    if anchors.shape[0] == 0:
        return np.zeros((0,), dtype=GLOBAL_COORD_DTYPE)
    total_time = future_interval_length * anchors.shape[1]
    anchor_progress = np.linalg.norm(np.diff(anchors[:, :, :2], axis=1), axis=-1).sum(axis=1)
    max_progress = max(float(current_speed), 0.0) * total_time + 0.5 * ORACLE_MAX_LONGITUDINAL_ACCEL * (total_time ** 2)
    stop_time = min(total_time, max(float(current_speed), 0.0) / max(ORACLE_MAX_LONGITUDINAL_DECEL, 1e-3))
    min_progress = max(float(current_speed), 0.0) * stop_time - 0.5 * ORACLE_MAX_LONGITUDINAL_DECEL * (stop_time ** 2)
    if stop_time < total_time:
        min_progress = max(min_progress, 0.0)
    if brake_like:
        max_progress = min(max_progress, max(float(current_speed), 0.0) * total_time)
    return np.maximum(min_progress - anchor_progress, 0.0) + np.maximum(anchor_progress - max_progress, 0.0)


def compute_longitudinal_valid_mask(
    anchors: np.ndarray,
    current_speed: float,
    current_acceleration: float,
    future_interval_length: float,
) -> np.ndarray:
    if anchors.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    dt = max(future_interval_length, 1e-3)
    horizon = anchors.shape[1]

    if anchors.shape[-1] >= 4:
        speed_profile = np.maximum(np.asarray(anchors[:, :horizon, 3], dtype=GLOBAL_COORD_DTYPE), 0.0)
    else:
        step_distance = np.linalg.norm(np.diff(anchors[:, :horizon, :2], axis=1), axis=-1)
        step_speed = step_distance / dt
        initial_speed = np.full((anchors.shape[0], 1), max(float(current_speed), 0.0), dtype=GLOBAL_COORD_DTYPE)
        speed_profile = np.concatenate([initial_speed, step_speed], axis=1)[:, :horizon]

    prev_speed = np.concatenate(
        [np.full((anchors.shape[0], 1), max(float(current_speed), 0.0), dtype=GLOBAL_COORD_DTYPE), speed_profile[:, :-1]],
        axis=1,
    )
    derived_accel = (speed_profile - prev_speed) / dt

    if anchors.shape[-1] >= 5:
        accel_profile = np.asarray(anchors[:, :horizon, 4], dtype=GLOBAL_COORD_DTYPE)
        accel_profile = np.where(np.isfinite(accel_profile), accel_profile, derived_accel)
    else:
        accel_profile = derived_accel

    prev_accel = np.concatenate(
        [np.full((anchors.shape[0], 1), float(current_acceleration), dtype=GLOBAL_COORD_DTYPE), accel_profile[:, :-1]],
        axis=1,
    )
    jerk_profile = (accel_profile - prev_accel) / dt

    step_valid = (
        np.isfinite(speed_profile)
        & np.isfinite(accel_profile)
        & np.isfinite(jerk_profile)
        & (speed_profile >= -1e-3)
        & (accel_profile <= ORACLE_MAX_LONGITUDINAL_ACCEL)
        & (accel_profile >= -ORACLE_MAX_LONGITUDINAL_DECEL)
        & (np.abs(jerk_profile) <= ORACLE_MAX_LONGITUDINAL_JERK)
    )
    return np.all(step_valid, axis=1)


def shrink_oracle_shortlist(
    anchor_indices: np.ndarray,
    anchor_trajectories: np.ndarray,
    model_log_scores: np.ndarray,
    eval_scores: np.ndarray,
    selection_scores: np.ndarray,
    best_local_index: int,
    keep_num: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if anchor_indices.size <= keep_num:
        return anchor_indices, anchor_trajectories, model_log_scores, eval_scores, selection_scores
    best_traj = anchor_trajectories[best_local_index]
    endpoint_dist = np.linalg.norm(anchor_trajectories[:, -1, :2] - best_traj[-1, :2][None, :], axis=1)
    progress_dist = np.abs(
        np.linalg.norm(np.diff(anchor_trajectories[:, :, :2], axis=1), axis=-1).sum(axis=1)
        - np.linalg.norm(np.diff(best_traj[:, :2], axis=0), axis=-1).sum()
    )
    yaw_dist = np.abs(wrap_angle(anchor_trajectories[:, -1, 2] - best_traj[-1, 2]))
    neighbor_metric = endpoint_dist + 0.25 * progress_dist + 0.5 * yaw_dist
    shortlist = np.argsort(neighbor_metric)[:keep_num]
    if best_local_index not in shortlist:
        shortlist[-1] = best_local_index
    shortlist = np.unique(shortlist)
    shortlist = shortlist[np.argsort(selection_scores[shortlist])[::-1]]
    return (
        anchor_indices[shortlist],
        anchor_trajectories[shortlist],
        model_log_scores[shortlist],
        eval_scores[shortlist],
        selection_scores[shortlist],
    )