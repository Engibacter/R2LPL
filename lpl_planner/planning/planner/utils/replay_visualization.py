from typing import Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from lpl_planner.planning.scene.scene_feature.features import (
    AgentPrediction,
    SceneFeature,
)


DEFAULT_REPLAY_STATE_RANGES: Tuple[Tuple[float, float], Tuple[float, float]] = (
    (-30.0, 50.0),
    (-40.0, 40.0),
)


def normalize_visual_scores(
    scores: np.ndarray,
    lower_percentile: float = 5.0,
    upper_percentile: float = 95.0,
    gamma: float = 1.5,
) -> np.ndarray:
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


def _rotate_poly(poly: np.ndarray, yaw: float, center: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    rot_mat = np.array([[c, -s], [s, c]])
    return (poly - center) @ rot_mat.T + center


def _box_corners(center: np.ndarray, half_length: float, half_width: float, yaw: float) -> np.ndarray:
    poly = np.array(
        [
            [center[0] - half_length, center[1] - half_width],
            [center[0] - half_length, center[1] + half_width],
            [center[0] + half_length, center[1] + half_width],
            [center[0] + half_length, center[1] - half_width],
        ],
        dtype=np.float64,
    )
    return _rotate_poly(poly, yaw, center)


def _plot_polyline(
    axes,
    trajectory: Optional[np.ndarray],
    color: str,
    linewidth: float,
    alpha: float,
    linestyle: str = "-",
    zorder: float = 2.0,
    label: Optional[str] = None,
) -> None:
    if trajectory is None:
        return
    trajectory = np.asarray(trajectory)
    if trajectory.shape[0] <= 1:
        return
    axes.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        zorder=zorder,
        label=label,
    )


def draw_muvo_replay_frame(
    scene_feature: SceneFeature,
    chosen_trajectory: Optional[np.ndarray] = None,
    all_trajectories: Optional[np.ndarray] = None,
    all_trajectory_scores: Optional[np.ndarray] = None,
    agent_prediction: Optional[AgentPrediction] = None,
    prediction_mode: str = "prediction",
    expert_future_trajectory: Optional[np.ndarray] = None,
    ego_history_trajectory: Optional[np.ndarray] = None,
    expert_history_trajectory: Optional[np.ndarray] = None,
    expert_current_pose: Optional[np.ndarray] = None,
    rollout_subset_trajectories: Optional[np.ndarray] = None,
    rollout_subset_scores: Optional[np.ndarray] = None,
    rollout_anchor_indices: Optional[np.ndarray] = None,
    rollout_teacher_sources: Optional[np.ndarray] = None,
    expert_path_local: Optional[np.ndarray] = None,
    expert_route_ref_path: Optional[np.ndarray] = None,
    state_ranges: Tuple[Tuple[float, float], Tuple[float, float]] = DEFAULT_REPLAY_STATE_RANGES,
    image_size_px: int = 2048,
) -> np.ndarray:
    """Draw a MUVO replay frame without depending on SceneManager runtime state."""
    plt.ioff()
    dpi = image_size_px / 10.0
    fig, axes = plt.subplots(figsize=(10, 10), dpi=dpi)
    axes.set_aspect("equal")
    axes.set(xlim=state_ranges[0])
    axes.set(ylim=state_ranges[1])

    road_feature = scene_feature.road_feature
    route_feature = scene_feature.route_feature
    ego_feature = scene_feature.ego_feature
    static_obstacle_feature = scene_feature.static_obstacle_feature
    agent_feature = scene_feature.agent_feature

    # Road and route layers.
    for idx, center_line in enumerate(road_feature.center_line):
        center_line = np.array(center_line)
        tl_data = road_feature.road_traffic_light[idx]
        if center_line.shape[0] > 1:
            axes.plot(
                center_line[:, 0],
                center_line[:, 1],
                color="red" if tl_data == 3 else "gray",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                zorder=0.8,
            )

    for polygon in road_feature.road_geometry:
        polygon = np.array(polygon)
        if polygon.shape[0] > 2:
            axes.fill(
                polygon[:, 0],
                polygon[:, 1],
                color="lightgray",
                alpha=0.3,
                edgecolor="gray",
                linewidth=1.0,
                zorder=0.4,
            )

    for route_polygon in route_feature.route_geometry:
        route_polygon = np.array(route_polygon)
        if route_polygon.shape[0] > 2:
            axes.fill(
                route_polygon[:, 0],
                route_polygon[:, 1],
                color="green",
                alpha=0.2,
                edgecolor="green",
                linewidth=1.0,
                zorder=0.6,
            )

    # Historical traces cover road layers but stay below boxes.
    _plot_polyline(
        axes,
        ego_history_trajectory,
        color="#1d4ed8",
        linewidth=3.0,
        alpha=0.88,
        zorder=1.6,
        label="Ego History",
    )
    _plot_polyline(
        axes,
        expert_history_trajectory,
        color="#f59e0b",
        linewidth=2.4,
        alpha=0.52,
        linestyle="--",
        zorder=1.55,
        label="Expert History",
    )

    # Static obstacles.
    if len(static_obstacle_feature.static_obstacle_position) > 0:
        static_obj_pos = np.asarray(static_obstacle_feature.static_obstacle_position)
        static_obj_dim = np.asarray(static_obstacle_feature.static_object_dimension)
        for static_obj_idx, pose in enumerate(static_obj_pos):
            poly = _box_corners(
                center=pose[:2],
                half_length=static_obj_dim[static_obj_idx][0],
                half_width=static_obj_dim[static_obj_idx][1],
                yaw=pose[2],
            )
            axes.fill(
                poly[:, 0],
                poly[:, 1],
                color="#ef4444",
                alpha=0.30,
                label="Static Obstacle" if static_obj_idx == 0 else None,
                edgecolor="#991b1b",
                linewidth=1.45,
                zorder=4.7,
            )

    # Agents and their history/future.
    if len(agent_feature.agent_current_state) > 0:
        agent_current = np.asarray(agent_feature.agent_current_state)
        agent_type = np.asarray(agent_feature.agent_type)
        agent_geo = agent_feature.agent_geometry
        agent_hist = agent_feature.agent_history_state
        agent_hist_mask = agent_feature.agent_history_mask

        for i, current in enumerate(agent_current):
            poly = _box_corners(
                center=current[:2],
                half_length=agent_geo[i][0],
                half_width=agent_geo[i][1],
                yaw=current[2],
            )
            axes.fill(
                poly[:, 0],
                poly[:, 1],
                color="#f97316",
                alpha=0.30,
                label="Agent Vehicle" if i == 0 else None,
                edgecolor="#9a3412",
                linewidth=1.55,
                zorder=5.2,
            )

            valid_idx = np.where(agent_hist_mask[i])[0]
            if len(valid_idx) > 1:
                valid_history = np.asarray(agent_hist[i])[valid_idx]
                axes.plot(
                    valid_history[:, 0],
                    valid_history[:, 1],
                    color="#7e22ce",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.5,
                    zorder=1.75,
                )

        if agent_prediction is not None and prediction_mode == "prediction":
            agent_future = np.array(agent_prediction.agent_future_state)
            agent_future_mask = np.array(agent_prediction.agent_future_mask)
            for i, future in enumerate(agent_future):
                valid_idx = np.where(agent_future_mask[i])[0]
                if len(valid_idx) > 1:
                    valid_future = future[valid_idx]
                    axes.plot(
                        valid_future[:, 0],
                        valid_future[:, 1],
                        color="#f97316",
                        linestyle="-",
                        linewidth=2,
                        alpha=0.6,
                        zorder=2.1,
                    )
        elif prediction_mode in {"CV", "CA", "CYAW"}:
            for i, current in enumerate(agent_current):
                x, y = current[0], current[1]
                yaw = current[2]
                speed = current[3]
                acceleration = current[4]
                yaw_rate = current[5]
                if agent_type[i] == 2:
                    yaw_rate = 0.0
                future_positions = []
                for _ in range(20):
                    x = x + speed * np.cos(yaw) * 0.2
                    y = y + speed * np.sin(yaw) * 0.2
                    if prediction_mode in {"CA", "CYAW"}:
                        speed = np.clip(speed + acceleration * 0.2, 0, None)
                    if prediction_mode == "CYAW":
                        yaw = yaw + yaw_rate * 0.2
                    future_positions.append([x, y])
                future_positions = np.array(future_positions)
                axes.plot(
                    future_positions[:, 0],
                    future_positions[:, 1],
                    color="#f97316",
                    linestyle=":",
                    linewidth=2,
                    alpha=0.6,
                    zorder=2.1,
                )

    # Candidate and rollout trajectories.
    if all_trajectories is not None and all_trajectory_scores is not None:
        score_norm = normalize_visual_scores(all_trajectory_scores)
        cmap = plt.get_cmap("Blues")
        for idx, traj_score in enumerate(score_norm):
            color_rgba = cmap(0.10 + 0.75 * traj_score)
            alpha = 0.06 + 0.88 * traj_score
            axes.plot(
                all_trajectories[idx][:, 0],
                all_trajectories[idx][:, 1],
                color=color_rgba,
                linewidth=1.5,
                alpha=alpha,
                zorder=2.0,
                label="Candidate Trajectory" if idx == 0 else None,
            )

    if rollout_subset_trajectories is not None and rollout_subset_scores is not None:
        rollout_subset_trajectories = np.asarray(rollout_subset_trajectories, dtype=np.float32)
        rollout_subset_scores = np.asarray(rollout_subset_scores, dtype=np.float32).reshape(-1)
        rollout_teacher_sources = None if rollout_teacher_sources is None else np.asarray(rollout_teacher_sources, dtype=np.int32).reshape(-1)
        score_norm = normalize_visual_scores(rollout_subset_scores)
        best_rollout_idx = int(np.argmax(rollout_subset_scores)) if rollout_subset_scores.size > 0 else -1
        teacher_styles = {
            0: {"color": "#7f1d1d", "linestyle": "-", "label": "Expert Teacher"},
            1: {"color": "#1f4e5f", "linestyle": "-", "label": "Expert Route Teacher"},
            2: {"color": "#5b4b1a", "linestyle": "-", "label": "Policy Teacher"},
            3: {"color": "#5c2a72", "linestyle": "-", "label": "Chosen Policy Anchor"},
            10: {"color": "#b45309", "linestyle": "-", "label": "Expert Teacher Cruise"},
            11: {"color": "#0f766e", "linestyle": "-", "label": "Expert Route Cruise"},
            12: {"color": "#6b7280", "linestyle": "-", "label": "Policy Teacher Cruise"},
            13: {"color": "#7c3aed", "linestyle": "-", "label": "Chosen Policy Cruise"},
            20: {"color": "#ea580c", "linestyle": "-", "label": "Expert Teacher Mild Accel"},
            21: {"color": "#0891b2", "linestyle": "-", "label": "Expert Route Mild Accel"},
            22: {"color": "#4b5563", "linestyle": "-", "label": "Policy Teacher Mild Accel"},
            23: {"color": "#9333ea", "linestyle": "-", "label": "Chosen Policy Mild Accel"},
        }
        used_labels = set()
        for idx, traj_score in enumerate(score_norm):
            teacher_source = -1 if rollout_teacher_sources is None or idx >= rollout_teacher_sources.shape[0] else int(rollout_teacher_sources[idx])
            style = teacher_styles.get(teacher_source, {"color": "#3f3f46", "linestyle": "-", "label": "Rollout Anchor"})
            alpha = 0.28 + 0.44 * traj_score
            trajectory = rollout_subset_trajectories[idx]
            is_best_rollout = idx == best_rollout_idx
            label = None if style["label"] in used_labels else style["label"]
            used_labels.add(style["label"])
            axes.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color="#111827" if is_best_rollout else style["color"],
                linewidth=2.2 if is_best_rollout else 1.35,
                linestyle=(0, (8, 4)) if is_best_rollout else style["linestyle"],
                alpha=1.0 if is_best_rollout else alpha,
                zorder=3.7 if is_best_rollout else 2.4,
                label="Best Rollout Anchor" if is_best_rollout else label,
            )
            if trajectory.shape[0] > 0:
                end_x, end_y = trajectory[-1, 0], trajectory[-1, 1]
                source_text = style["label"].replace(" Teacher", "").replace(" Anchor", "").replace(" Rollout", "")
                axes.text(
                    end_x,
                    end_y,
                    source_text,
                    color="#111827" if is_best_rollout else style["color"],
                    fontsize=6.5,
                    alpha=0.9 if is_best_rollout else min(alpha + 0.15, 0.9),
                    zorder=4.0 if is_best_rollout else 3.0,
                )
                if is_best_rollout:
                    axes.scatter(
                        [end_x],
                        [end_y],
                        s=52,
                        color="#111827",
                        marker="D",
                        edgecolors="#f9fafb",
                        linewidths=0.9,
                        zorder=3.9,
                    )

    _plot_polyline(
        axes,
        expert_path_local,
        color="#6b5b2a",
        linewidth=0.95,
        alpha=0.95,
        zorder=3.2,
        label="Expert Path" if expert_future_trajectory is None else None,
    )
    _plot_polyline(
        axes,
        expert_route_ref_path,
        color="#355c7d",
        linewidth=0.95,
        alpha=0.95,
        zorder=3.1,
        label="Expert Route Ref Path",
    )

    # Expert future and chosen trajectory are intentionally visually distinct.
    _plot_polyline(
        axes,
        expert_future_trajectory,
        color="#fb923c",
        linewidth=2.0,
        alpha=0.72,
        linestyle="-.",
        zorder=3.85,
        label="Expert Future",
    )
    _plot_polyline(
        axes,
        chosen_trajectory,
        color="#dc2626",
        linewidth=2.8,
        alpha=0.95,
        zorder=4.4,
        label="Chosen Trajectory",
    )

    # Expert virtual box, then simulated ego box on top.
    ego_half_width = float(ego_feature.ego_geometry[0])
    ego_half_length = float(ego_feature.ego_geometry[1])
    rear_axle_to_center = 1.461
    if expert_current_pose is not None:
        expert_current_pose = np.asarray(expert_current_pose, dtype=np.float64)
        expert_center = expert_current_pose[:2] + rear_axle_to_center * np.array(
            [np.cos(expert_current_pose[2]), np.sin(expert_current_pose[2])]
        )
        expert_poly = _box_corners(expert_center, ego_half_length, ego_half_width, expert_current_pose[2])
        axes.fill(
            expert_poly[:, 0],
            expert_poly[:, 1],
            color="#60a5fa",
            alpha=0.18,
            edgecolor="#1d4ed8",
            linewidth=1.45,
            linestyle="--",
            label="Expert Ego",
            zorder=5.8,
        )

    ego_center = np.array([rear_axle_to_center, 0.0], dtype=np.float64)
    ego_poly = _box_corners(ego_center, ego_half_length, ego_half_width, 0.0)
    axes.fill(
        ego_poly[:, 0],
        ego_poly[:, 1],
        color="#2563eb",
        alpha=0.50,
        label="Ego Vehicle",
        edgecolor="#1e3a8a",
        linewidth=1.65,
        zorder=6.0,
    )

    handles, labels = axes.get_legend_handles_labels()
    if labels:
        dedup = dict(zip(labels, handles))
        axes.legend(dedup.values(), dedup.keys(), loc="upper right")

    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
        int(height),
        int(width),
        3,
    )
    plt.close(fig)
    return img
