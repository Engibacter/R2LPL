from typing import Optional

import torch


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    abs_x = x.abs()
    return torch.where(abs_x <= delta, 0.5 * abs_x.square(), delta * (abs_x - 0.5 * delta))


def _normalize_expert_anchor_inputs(
    expert_traj_local: torch.Tensor,
    anchors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if expert_traj_local.ndim == 2 and anchors.ndim == 3:
        expert_traj_local = expert_traj_local.unsqueeze(0)
        anchors = anchors.unsqueeze(0)
        squeeze_batch = True
    elif expert_traj_local.ndim == 3 and anchors.ndim == 4:
        squeeze_batch = False
    else:
        raise ValueError(
            "Expected expert/anchor shapes [T,D] & [K,T,D] or [B,T,D] & [B,K,T,D]."
        )

    if expert_traj_local.shape[0] != anchors.shape[0]:
        raise ValueError("Batch size of expert trajectories and anchors must match.")

    if expert_traj_local.shape[1] != anchors.shape[2]:
        raise ValueError("Expert and anchors must have the same trajectory length.")

    expert_traj_local = expert_traj_local[..., :6]
    anchors = anchors[..., :6].to(device=expert_traj_local.device, dtype=expert_traj_local.dtype)
    return expert_traj_local, anchors, squeeze_batch


def _unwrap_angle_sequence(angle: torch.Tensor) -> torch.Tensor:
    if angle.shape[1] <= 1:
        return angle

    delta = angle[:, 1:] - angle[:, :-1]
    wrapped_delta = wrap_angle(delta)
    pi = angle.new_tensor(torch.pi)
    wrapped_delta = torch.where(
        (wrapped_delta == -pi) & (delta > 0),
        wrapped_delta + 2 * pi,
        wrapped_delta,
    )
    correction = wrapped_delta - delta
    correction = torch.where(delta.abs() < pi, torch.zeros_like(correction), correction)
    return torch.cat([angle[:, :1], angle[:, 1:] + correction.cumsum(dim=1)], dim=1)


def resample_trajectories_by_arclength_batch(
    trajectories: torch.Tensor,
    target_s: torch.Tensor,
) -> torch.Tensor:
    """
    trajectories: [B, T, D]
    target_s: [B, R]
    return: [B, R, D]
    """
    if trajectories.ndim != 3 or target_s.ndim != 2:
        raise ValueError("Expected trajectories [B,T,D] and target_s [B,R].")

    batch_size, traj_len, feat_dim = trajectories.shape
    if target_s.shape[0] != batch_size:
        raise ValueError("Batch size of trajectories and target_s must match.")

    xy = trajectories[:, :, :2]
    seg_len = torch.linalg.norm(xy[:, 1:] - xy[:, :-1], dim=-1).clamp_min(1e-3)
    cum_s = torch.cat([torch.zeros_like(seg_len[:, :1]), seg_len.cumsum(dim=1)], dim=1)

    right = (cum_s.unsqueeze(-1) < target_s.unsqueeze(1)).sum(dim=1).clamp_(1, traj_len - 1)
    left = right - 1

    gather_left = left.unsqueeze(-1).expand(-1, -1, feat_dim)
    gather_right = right.unsqueeze(-1).expand(-1, -1, feat_dim)

    s_left = cum_s.gather(1, left)
    s_right = cum_s.gather(1, right)
    denom = (s_right - s_left).clamp_min(1e-6)
    interp_weight = ((target_s - s_left) / denom).unsqueeze(-1)

    traj_left = trajectories.gather(1, gather_left)
    traj_right = trajectories.gather(1, gather_right)
    resampled = traj_left + interp_weight * (traj_right - traj_left)

    yaw = _unwrap_angle_sequence(trajectories[:, :, 2])
    yaw_left = yaw.gather(1, left)
    yaw_right = yaw.gather(1, right)
    yaw_interp = yaw_left + ((target_s - s_left) / denom) * (yaw_right - yaw_left)
    resampled[:, :, 2] = wrap_angle(yaw_interp)

    return resampled


def resample_trajectory_by_arclength(
    trajectory: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """
    trajectory: [B, T, D] or [T, D]
    return: [B, R, D] or [R, D]
    """
    squeeze_batch = False
    if trajectory.ndim == 2:
        trajectory = trajectory.unsqueeze(0)
        squeeze_batch = True
    elif trajectory.ndim != 3:
        raise ValueError("Expected trajectory [T,D] or [B,T,D].")

    xy = trajectory[:, :, :2]
    seg_len = torch.linalg.norm(xy[:, 1:] - xy[:, :-1], dim=-1).clamp_min(1e-3)
    arclength = torch.cat([torch.zeros_like(seg_len[:, :1]), seg_len.cumsum(dim=1)], dim=1)
    total_length = arclength[:, -1]
    target_ratio = torch.linspace(0.0, 1.0, num_samples, device=trajectory.device, dtype=trajectory.dtype)
    target_s = target_ratio.unsqueeze(0) * total_length.unsqueeze(1)

    resampled = resample_trajectories_by_arclength_batch(trajectory, target_s)
    degenerate_mask = total_length < 1e-3
    if degenerate_mask.any():
        repeated = trajectory[degenerate_mask, :1].expand(-1, num_samples, -1)
        resampled[degenerate_mask] = repeated

    return resampled.squeeze(0) if squeeze_batch else resampled


def compute_expert_frame_errors_batch(
    expert_traj: torch.Tensor,
    candidate_trajs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    expert_traj: [B, R, 6]
    candidate_trajs: [B, K, R, 6]
    return:
        e_lon: [B, K, R]
        e_lat: [B, K, R]
        e_yaw: [B, K, R]
    """
    if expert_traj.ndim != 3 or candidate_trajs.ndim != 4:
        raise ValueError("Expected expert_traj [B,R,D] and candidate_trajs [B,K,R,D].")

    expert_xy = expert_traj[:, :, :2]
    tangent = torch.zeros_like(expert_xy)
    tangent[:, 1:-1] = expert_xy[:, 2:] - expert_xy[:, :-2]
    tangent[:, 0] = expert_xy[:, 1] - expert_xy[:, 0]
    tangent[:, -1] = expert_xy[:, -1] - expert_xy[:, -2]

    tangent = tangent / torch.linalg.norm(tangent, dim=-1, keepdim=True).clamp_min(1e-6)
    normal = torch.stack([-tangent[:, :, 1], tangent[:, :, 0]], dim=-1)

    delta_xy = candidate_trajs[:, :, :, :2] - expert_xy.unsqueeze(1)
    e_lon = (delta_xy * tangent.unsqueeze(1)).sum(dim=-1)
    e_lat = (delta_xy * normal.unsqueeze(1)).sum(dim=-1)
    e_yaw = wrap_angle(candidate_trajs[:, :, :, 2] - expert_traj[:, None, :, 2])
    return e_lon, e_lat, e_yaw


def compute_anchor_prefilter_score(
    expert_traj_local: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    """
    expert_traj_local: [T, 6] or [B, T, 6]
    anchors: [K, T, 6] or [B, K, T, 6]
    return: [K] or [B, K]
    """
    expert_traj_local, anchors, squeeze_batch = _normalize_expert_anchor_inputs(
        expert_traj_local, anchors
    )
    batch_size, num_anchors, traj_len, _ = anchors.shape

    expert_end_y = expert_traj_local[:, -1, 1]
    expert_end_yaw = wrap_angle(expert_traj_local[:, -1, 2] - expert_traj_local[:, 0, 2])
    expert_path_len = torch.linalg.norm(
        expert_traj_local[:, 1:, :2] - expert_traj_local[:, :-1, :2], dim=-1
    ).sum(dim=1)
    expert_turn_energy = wrap_angle(
        expert_traj_local[:, 1:, 2] - expert_traj_local[:, :-1, 2]
    ).abs().sum(dim=1)

    anchor_end_y = anchors[:, :, -1, 1]
    anchor_end_yaw = wrap_angle(anchors[:, :, -1, 2] - anchors[:, :, 0, 2])
    anchor_path_len = torch.linalg.norm(anchors[:, :, 1:, :2] - anchors[:, :, :-1, :2], dim=-1).sum(dim=-1)
    anchor_turn_energy = wrap_angle(anchors[:, :, 1:, 2] - anchors[:, :, :-1, 2]).abs().sum(dim=-1)

    checkpoint_ids = torch.tensor(
        [
            max(1, int(0.25 * (traj_len - 1))),
            max(1, int(0.50 * (traj_len - 1))),
            max(1, int(0.75 * (traj_len - 1))),
        ],
        device=anchors.device,
        dtype=torch.long,
    )

    expert_y_ckpt = expert_traj_local[:, checkpoint_ids, 1]
    expert_yaw_ckpt = expert_traj_local[:, checkpoint_ids, 2]
    anchor_y_ckpt = anchors[:, :, checkpoint_ids, 1]
    anchor_yaw_ckpt = anchors[:, :, checkpoint_ids, 2]

    y_ckpt_cost = (anchor_y_ckpt - expert_y_ckpt.unsqueeze(1)).abs().mean(dim=-1)
    yaw_ckpt_cost = wrap_angle(anchor_yaw_ckpt - expert_yaw_ckpt.unsqueeze(1)).abs().mean(dim=-1)

    turn_ratio = (expert_end_yaw.abs() / 0.35).clamp(0.0, 1.0)

    w_end_y = 3.0 - 1.8 * turn_ratio
    w_ckpt_y = 2.0 - 1.0 * turn_ratio
    w_end_yaw = 2.5 + 1.0 * turn_ratio
    w_turn = 1.5 + 0.8 * turn_ratio
    w_path = expert_traj_local.new_full((batch_size,), 0.5)

    prefilter_score = (
        w_end_y.unsqueeze(1) * (anchor_end_y - expert_end_y.unsqueeze(1)).abs()
        + w_end_yaw.unsqueeze(1) * wrap_angle(anchor_end_yaw - expert_end_yaw.unsqueeze(1)).abs()
        + w_turn.unsqueeze(1) * (anchor_turn_energy - expert_turn_energy.unsqueeze(1)).abs()
        + w_path.unsqueeze(1) * (anchor_path_len - expert_path_len.unsqueeze(1)).abs()
        + w_ckpt_y.unsqueeze(1) * y_ckpt_cost
        + yaw_ckpt_cost
    )

    return prefilter_score.squeeze(0) if squeeze_batch else prefilter_score


def compute_anchor_prefilter_score_shared_bank(
    expert_traj_local: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    """
    expert_traj_local: [T, 6] or [B, T, 6]
    anchors: [K, T, 6]
    return: [K] or [B, K]
    """
    squeeze_batch = False
    if expert_traj_local.ndim == 2:
        expert_traj_local = expert_traj_local.unsqueeze(0)
        squeeze_batch = True
    elif expert_traj_local.ndim != 3:
        raise ValueError("Expected expert trajectories [T,D] or [B,T,D].")

    if anchors.ndim != 3:
        raise ValueError("Expected anchors [K,T,D].")

    expert_traj_local = expert_traj_local[..., :6]
    anchors = anchors[..., :6].to(device=expert_traj_local.device, dtype=expert_traj_local.dtype)

    batch_size, traj_len, _ = expert_traj_local.shape

    expert_end_y = expert_traj_local[:, -1, 1]
    expert_end_yaw = wrap_angle(expert_traj_local[:, -1, 2] - expert_traj_local[:, 0, 2])
    expert_path_len = torch.linalg.norm(
        expert_traj_local[:, 1:, :2] - expert_traj_local[:, :-1, :2], dim=-1
    ).sum(dim=1)
    expert_turn_energy = wrap_angle(
        expert_traj_local[:, 1:, 2] - expert_traj_local[:, :-1, 2]
    ).abs().sum(dim=1)

    anchor_end_y = anchors[:, -1, 1]
    anchor_end_yaw = wrap_angle(anchors[:, -1, 2] - anchors[:, 0, 2])
    anchor_path_len = torch.linalg.norm(anchors[:, 1:, :2] - anchors[:, :-1, :2], dim=-1).sum(dim=-1)
    anchor_turn_energy = wrap_angle(anchors[:, 1:, 2] - anchors[:, :-1, 2]).abs().sum(dim=-1)

    checkpoint_ids = torch.tensor(
        [
            max(1, int(0.25 * (traj_len - 1))),
            max(1, int(0.50 * (traj_len - 1))),
            max(1, int(0.75 * (traj_len - 1))),
        ],
        device=anchors.device,
        dtype=torch.long,
    )

    expert_y_ckpt = expert_traj_local[:, checkpoint_ids, 1]
    expert_yaw_ckpt = expert_traj_local[:, checkpoint_ids, 2]
    anchor_y_ckpt = anchors[:, checkpoint_ids, 1]
    anchor_yaw_ckpt = anchors[:, checkpoint_ids, 2]

    y_ckpt_cost = (anchor_y_ckpt.unsqueeze(0) - expert_y_ckpt.unsqueeze(1)).abs().mean(dim=-1)
    yaw_ckpt_cost = wrap_angle(anchor_yaw_ckpt.unsqueeze(0) - expert_yaw_ckpt.unsqueeze(1)).abs().mean(dim=-1)

    turn_ratio = (expert_end_yaw.abs() / 0.35).clamp(0.0, 1.0)

    w_end_y = 3.0 - 1.8 * turn_ratio
    w_ckpt_y = 2.0 - 1.0 * turn_ratio
    w_end_yaw = 2.5 + 1.0 * turn_ratio
    w_turn = 1.5 + 0.8 * turn_ratio
    w_path = expert_traj_local.new_full((batch_size,), 0.5)

    prefilter_score = (
        w_end_y.unsqueeze(1) * (anchor_end_y.unsqueeze(0) - expert_end_y.unsqueeze(1)).abs()
        + w_end_yaw.unsqueeze(1) * wrap_angle(anchor_end_yaw.unsqueeze(0) - expert_end_yaw.unsqueeze(1)).abs()
        + w_turn.unsqueeze(1) * (anchor_turn_energy.unsqueeze(0) - expert_turn_energy.unsqueeze(1)).abs()
        + w_path.unsqueeze(1) * (anchor_path_len.unsqueeze(0) - expert_path_len.unsqueeze(1)).abs()
        + w_ckpt_y.unsqueeze(1) * y_ckpt_cost
        + yaw_ckpt_cost
    )

    return prefilter_score.squeeze(0) if squeeze_batch else prefilter_score


def preselect_anchors_by_expert_shape(
    expert_traj_local: torch.Tensor,
    anchors: torch.Tensor,
    preselect_k: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        selected_indices: [preselect_k] or [B, preselect_k]
        prefilter_scores: [K] or [B, K]
    """
    prefilter_scores = compute_anchor_prefilter_score(expert_traj_local, anchors)
    num_anchors = prefilter_scores.shape[-1]
    preselect_k = min(max(preselect_k, 1), num_anchors)
    selected_indices = torch.topk(prefilter_scores, k=preselect_k, dim=-1, largest=False).indices
    return selected_indices, prefilter_scores


def select_best_anchors_by_expert_shape(
    expert_traj_local: torch.Tensor,
    anchors: torch.Tensor,
    preselect_k: int = 128,
    topk: int = 1,
    num_resample: Optional[int] = None,
    regularization_weight: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-safe teacher-anchor selection with a shared anchor bank.

    expert_traj_local: [T, 6] or [B, T, 6]
    anchors: [K, T, 6]
    return:
        selected_indices: [topk] or [B, topk]
        selected_scores: [topk] or [B, topk]
    """
    if anchors.ndim != 3:
        raise ValueError("Expected anchors shape [K, T, D].")

    squeeze_batch = False
    if expert_traj_local.ndim == 2:
        expert_traj_local = expert_traj_local.unsqueeze(0)
        squeeze_batch = True
    elif expert_traj_local.ndim != 3:
        raise ValueError("Expected expert trajectories [T,D] or [B,T,D].")

    anchors = anchors[..., :6].to(device=expert_traj_local.device, dtype=expert_traj_local.dtype)
    batch_size = expert_traj_local.shape[0]
    num_anchors = anchors.shape[0]
    topk = min(max(int(topk), 1), num_anchors)
    preselect_k = min(max(int(preselect_k), topk), num_anchors)

    coarse_prefilter_scores = compute_anchor_prefilter_score_shared_bank(
        expert_traj_local,
        anchors,
    )
    coarse_indices = torch.topk(
        coarse_prefilter_scores,
        k=preselect_k,
        dim=-1,
        largest=False,
    ).indices

    traj_len, feat_dim = anchors.shape[1], anchors.shape[2]
    coarse_anchors = anchors.index_select(0, coarse_indices.reshape(-1)).view(
        batch_size,
        preselect_k,
        traj_len,
        feat_dim,
    )
    coarse_scores = score_anchors_by_expert_traj(
        expert_traj_local,
        coarse_anchors,
        num_resample=num_resample,
        regularization_weight=regularization_weight,
    )
    top_local = torch.topk(
        coarse_scores,
        k=topk,
        dim=-1,
        largest=False,
    ).indices
    selected_indices = coarse_indices.gather(1, top_local)
    selected_scores = coarse_scores.gather(1, top_local)

    if squeeze_batch:
        return selected_indices.squeeze(0), selected_scores.squeeze(0)
    return selected_indices, selected_scores


def score_anchors_by_expert_traj(
    expert_traj_local: torch.Tensor,
    anchors: torch.Tensor,
    num_resample: Optional[int] = None,
    regularization_weight: float = 0.15,
) -> torch.Tensor:
    """
    Full anchor ranking score without preselection.

    expert_traj_local: [T, 6] or [B, T, 6]
    anchors: [K, T, 6] or [B, K, T, 6]
    return: geometry_scores [K] or [B, K]
    """
    expert_traj_local, anchors, squeeze_batch = _normalize_expert_anchor_inputs(
        expert_traj_local, anchors
    )
    batch_size, num_anchors, traj_len, feat_dim = anchors.shape

    prefilter_scores = compute_anchor_prefilter_score(expert_traj_local, anchors)

    if num_resample is None:
        num_resample = max(2 * traj_len, 32)

    expert_rs = resample_trajectory_by_arclength(expert_traj_local, num_resample)

    anchors_flat = anchors.reshape(batch_size * num_anchors, traj_len, feat_dim)
    anchor_seg_len = torch.linalg.norm(anchors_flat[:, 1:, :2] - anchors_flat[:, :-1, :2], dim=-1).clamp_min(1e-3)
    anchor_cum_s = torch.cat([torch.zeros_like(anchor_seg_len[:, :1]), anchor_seg_len.cumsum(dim=1)], dim=1)
    total_len = anchor_cum_s[:, -1].clamp_min(1e-3)
    target_ratio = torch.linspace(0.0, 1.0, num_resample, device=anchors.device, dtype=anchors.dtype)
    target_s = target_ratio.unsqueeze(0) * total_len.unsqueeze(1)
    anchor_rs = resample_trajectories_by_arclength_batch(anchors_flat, target_s)
    anchor_rs = anchor_rs.view(batch_size, num_anchors, num_resample, feat_dim)

    e_lon, e_lat, e_yaw = compute_expert_frame_errors_batch(expert_rs, anchor_rs)

    weights = torch.linspace(0.8, 1.2, num_resample, device=anchors.device, dtype=anchors.dtype)
    weights[-max(4, num_resample // 6):] *= 2.0
    weights = weights / weights.sum()

    lat_cost = huber(e_lat / 0.60, delta=1.0)
    lon_cost = huber(e_lon / 2.50, delta=1.0)
    yaw_cost = huber((2.0 * torch.sin(0.5 * e_yaw)) / 0.20, delta=1.0)

    turn_ratio = (wrap_angle(expert_traj_local[:, -1, 2] - expert_traj_local[:, 0, 2]).abs() / 0.35).clamp(0.0, 1.0)

    w_lat = 2.5 - 0.7 * turn_ratio
    w_lon = expert_traj_local.new_full((batch_size,), 0.8)
    w_yaw = 1.5 + 0.8 * turn_ratio
    w_terminal_lat = 4.0 - 1.8 * turn_ratio
    w_terminal_lon = expert_traj_local.new_full((batch_size,), 1.0)
    w_terminal_yaw = 3.0 + 1.5 * turn_ratio

    point_cost = (
        w_lat.unsqueeze(1) * (lat_cost * weights.view(1, 1, -1)).sum(dim=-1)
        + w_lon.unsqueeze(1) * (lon_cost * weights.view(1, 1, -1)).sum(dim=-1)
        + w_yaw.unsqueeze(1) * (yaw_cost * weights.view(1, 1, -1)).sum(dim=-1)
    )

    terminal_cost = (
        w_terminal_lat.unsqueeze(1) * e_lat[:, :, -1].abs()
        + w_terminal_lon.unsqueeze(1) * e_lon[:, :, -1].abs()
        + w_terminal_yaw.unsqueeze(1) * e_yaw[:, :, -1].abs()
    )

    coarse_score_norm = prefilter_scores / torch.quantile(
        prefilter_scores,
        0.5,
        dim=1,
        keepdim=True,
    ).clamp_min(1e-6)
    geometry_distance = point_cost + terminal_cost + regularization_weight * coarse_score_norm
    return geometry_distance.squeeze(0) if squeeze_batch else geometry_distance

def compute_vehicle_corners(center_xy: torch.Tensor,
                             yaw: torch.Tensor,
                             half_length: float = 2.5880,
                             half_width: float = 1.1485,
                             rear_axle_to_center: float = 1.461,
                            ) -> torch.Tensor:
    """
    center_xy: [...,2]
    yaw:       [...,1]
    Returns: corners [...,4,2] (FL, FR, RR, RL)
    """

    cos, sin = torch.cos(yaw), torch.sin(yaw) # [B, K, 1]

    # calculate ego center from rear axle
    rear_axle_to_center_translate = torch.cat(
        (rear_axle_to_center * cos, rear_axle_to_center * sin), dim=-1
    ) # [B, K, 2]
    ego_centers = center_xy + rear_axle_to_center_translate # [B, K, 2]

    hl = half_length
    hw = half_width
    # local corners
    local = torch.tensor([[ hl,  hw],
                          [ hl, -hw],
                          [-hl, -hw],
                          [-hl,  hw]], dtype=center_xy.dtype, device=center_xy.device)  # [4,2]

    # rotate and translate to global
    rot = torch.stack([
        cos * local[None, :, 0] - sin * local[None, :, 1],
        sin * local[None, :, 0] + cos * local[None, :, 1]
    ], dim=-1)  # [B, K, 4, 2]
    return rot + ego_centers[..., None, :]  

def batched_point_in_polygon(points: torch.Tensor,
                             polygons: torch.Tensor) -> torch.Tensor:
    """
    Vectorized ray casting.
    points: [B,N,2]
    polygons: [B,P,V,2]  (padded with 0; V=20)
    Returns: inside_mask [B,N,P] (point inside each polygon)
    """
    B, N, _ = points.shape
    _, P, V, _ = polygons.shape

    # Valid vertex mask (True if not all zeros)
    v_valid = ~(polygons == 0).all(-1)                 # [B,P,V]
    # Roll vertices to form edges
    v0 = polygons                                      # [B,P,V,2]
    v1 = torch.roll(polygons, shifts=-1, dims=2)       # [B,P,V,2]
    v1_valid = torch.roll(v_valid, shifts=-1, dims=2)  # [B,P,V]
    edge_valid = v_valid & v1_valid                   # both ends valid

    # Extract coordinates
    x0 = v0[..., 0]            # [B,P,V]
    y0 = v0[..., 1]
    x1 = v1[..., 0]
    y1 = v1[..., 1]

    # Points coords
    x = points[..., 0].unsqueeze(2).unsqueeze(3)       # [B,N,1,1]
    y = points[..., 1].unsqueeze(2).unsqueeze(3)       # [B,N,1,1]

    # Broadcast to [B,N,P,V]
    y0b = y0.unsqueeze(1)        # [B,1,P,V]
    y1b = y1.unsqueeze(1)
    x0b = x0.unsqueeze(1)
    x1b = x1.unsqueeze(1)
    edge_valid_b = edge_valid.unsqueeze(1)  # [B,1,P,V]

    # Ray casting conditions
    cond_cross = ((y0b > y) != (y1b > y)) & edge_valid_b          # [B,N,P,V]
    denom = (y1b - y0b).clamp_min(1e-12)
    x_int = (x1b - x0b) * (y - y0b) / denom + x0b
    cond_right = x < x_int
    crossings = (cond_cross & cond_right).sum(dim=-1)             # [B,N,P]

    inside = (crossings % 2 == 1) & (v_valid.any(-1).unsqueeze(1))  # polygon with >=1 valid vert
    return inside  # [B,N,P]

def corners_in_any_road(corners: torch.Tensor,
                        road_polygons_batch: torch.Tensor) -> torch.Tensor:
    """
    corners: [B,K,4,2]
    road_polygons_batch: [B,P,V,2]
    Returns [B,K] True if each of the 4 corner points lies inside at least one polygon
    (不要求四个角同时位于同一个 polygon 中).
    """
    B, K, C, _ = corners.shape
    assert C == 4, "Expected 4 corners"
    pts_flat = corners.view(B, K * C, 2)                 # [B,K*4,2]
    inside_each = batched_point_in_polygon(pts_flat, road_polygons_batch)  # [B,K*4,P]
    inside_each = inside_each.view(B, K, C, -1)          # [B,K,4,P]
    # 每个角是否在任一 polygon 内
    corner_in_some_poly = inside_each.any(dim=-1)        # [B,K,4]
    # 所有四个角都至少落在某个 polygon
    all_corners_ok = corner_in_some_poly.all(dim=2)      # [B,K]
    return all_corners_ok

def centers_in_route_score(centers: torch.Tensor,
                         route_polygons_batch: torch.Tensor) -> torch.Tensor:
    """
    centers: [B,K,N,2]
    route_polygons_batch: [B,P,V,2]
    Returns [B,K] True if each center point lies inside at least one polygon
    """
    B, K, N, _ = centers.shape
    pts_flat = centers.reshape(B, K * N, 2)                 # [B,K*N,2]
    inside_each = batched_point_in_polygon(pts_flat, route_polygons_batch)  # [B,K*N,P]
    inside_each = inside_each.reshape(B, K, N, -1)          # [B,K,N,P]
    inside_any = inside_each.any(dim=-1)  # [B,K,N]
    inside_all = inside_any.all(dim=-1)  # [B,K]
    inside_any = inside_any.any(dim=-1)  # [B,K]
    score = inside_all.float()
    score[~inside_any] = 0.0  # 如果一个点都不在路线上，得分为0
    score[inside_all] = 1.0  # 如果所有点都在路线上，得分为1
    score[(~inside_all) & inside_any] = 0.5  # 部分点在路线上，得分为0.5
    return score

def compute_progress(last_pos: torch.Tensor,
                      expert_traj_xy: torch.Tensor) -> torch.Tensor:
    """
    last_pos: [B,K,2]
    expert_traj_xy: [B,T,2]
    Returns progress scalar [B,K] = projected cumulative distance index / (T-1).
    """
    B,K,_ = last_pos.shape
    T = expert_traj_xy.shape[1]
    # cumulative distances along expert
    diffs = expert_traj_xy[:, 1:] - expert_traj_xy[:, :-1]
    seg_len = torch.linalg.norm(diffs, dim=-1)  # [B,T-1]
    cum = torch.cat([torch.zeros(B,1, device=last_pos.device), seg_len.cumsum(dim=1)], dim=1)  # [B,T]
    # find closest point index
    # Compute squared dist: [B,K,T]
    d2 = torch.sum((last_pos.unsqueeze(-2) - expert_traj_xy.unsqueeze(1))**2, dim=-1)
    idx = torch.argmin(d2, dim=-1)  # [B,K]
    prog = cum.gather(1, idx) / (cum[:, -1:].clamp_min(1e-6))  # [B,K]
    prog = prog.clamp(0.0, 1.0)
    return prog

