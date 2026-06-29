import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from matplotlib import patheffects, pyplot as plt

from lpl_planner.planning.scene.evaluate.utils.evaluate_utils import MultiMetricIndex, WeightedMetricIndex
from lpl_planner.planning.scene.scene_feature.features import AgentPrediction, SceneFeature
from lpl_planner.planning.scene.scene_manager import SceneManager


WEIGHTED_METRIC_DESCRIPTIONS = {
    WeightedMetricIndex.PROGRESS.name: "Route/reference progress reward after TTC gating.",
    WeightedMetricIndex.TTC.name: "Time-to-collision safety score.",
    WeightedMetricIndex.COMFORTABLE.name: "Comfort score from jerk, acceleration, and yaw-rate limits.",
    WeightedMetricIndex.SPEED_LIMIT.name: "Speed limit compliance score.",
    WeightedMetricIndex.LANE_CENTER_DISTANCE.name: "Lane-center distance alignment score.",
    WeightedMetricIndex.HEADING_COMPLIANCE.name: "Heading alignment with lane/reference direction.",
}

MULTI_METRIC_DESCRIPTIONS = {
    MultiMetricIndex.NO_COLLISION.name: "Binary multiplicative no-at-fault collision score.",
    MultiMetricIndex.DRIVABLE_AREA.name: "Binary multiplicative drivable-area compliance score.",
    MultiMetricIndex.DRIVING_DIRECTION.name: "Binary multiplicative wrong-way driving score.",
    MultiMetricIndex.WITHIN_LANE.name: "Binary multiplicative within-lane compliance score.",
    MultiMetricIndex.RED_LIGHT_COMPLIANCE.name: "Binary multiplicative red-light compliance score.",
    MultiMetricIndex.FOLLOWING_COMPLIANCE.name: "Continuous multiplicative following compliance score from forward violating progress.",
}


def _as_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach") and callable(getattr(value, "detach")):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _pad_candidate_array(values: Optional[np.ndarray], target_size: int, fill_value: float = np.nan) -> np.ndarray:
    if target_size <= 0:
        return np.zeros((0,), dtype=np.float32)
    if values is None:
        return np.full((target_size,), fill_value, dtype=np.float32)
    values = np.asarray(values).reshape(-1)
    if values.shape[0] >= target_size:
        return values[:target_size]
    return np.pad(values, (0, target_size - values.shape[0]), constant_values=fill_value)


def _rotate_poly(poly: np.ndarray, yaw: float, center: np.ndarray) -> np.ndarray:
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rot_mat = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
    return (poly - center) @ rot_mat.T + center


def _draw_scene_context(axes, scene_feature: SceneFeature, agent_prediction: Optional[AgentPrediction], prediction_mode: str) -> None:
    road_feature = scene_feature.road_feature
    route_feature = scene_feature.route_feature
    ego_feature = scene_feature.ego_feature
    static_obstacle_feature = scene_feature.static_obstacle_feature
    agent_feature = scene_feature.agent_feature

    for idx, center_line in enumerate(road_feature.center_line):
        center_line = np.asarray(center_line)
        tl_state = road_feature.road_traffic_light[idx]
        if center_line.shape[0] > 1:
            color = "red" if tl_state == 3 else "gray"
            axes.plot(center_line[:, 0], center_line[:, 1], color=color, linestyle="--", linewidth=1, alpha=0.7)

    for polygon in road_feature.road_geometry:
        polygon = np.asarray(polygon)
        if polygon.shape[0] > 2:
            axes.fill(polygon[:, 0], polygon[:, 1], color="lightgray", alpha=0.3, edgecolor="gray")

    for route_polygon in route_feature.route_geometry:
        route_polygon = np.asarray(route_polygon)
        if route_polygon.shape[0] > 2:
            axes.fill(route_polygon[:, 0], route_polygon[:, 1], color="green", alpha=0.2, edgecolor="green")

    ego_pose = np.array([1.461, 0.0], dtype=np.float32)
    ego_half_width = float(ego_feature.ego_geometry[0])
    ego_half_length = float(ego_feature.ego_geometry[1])
    ego_poly = np.array(
        [
            [ego_pose[0] - ego_half_length, -ego_half_width],
            [ego_pose[0] - ego_half_length, ego_half_width],
            [ego_pose[0] + ego_half_length, ego_half_width],
            [ego_pose[0] + ego_half_length, -ego_half_width],
        ],
        dtype=np.float32,
    )
    axes.fill(ego_poly[:, 0], ego_poly[:, 1], color="blue", alpha=0.5, edgecolor="blue")

    static_pos = _as_numpy(static_obstacle_feature.static_obstacle_position)
    static_dim = _as_numpy(static_obstacle_feature.static_object_dimension)
    if static_pos is not None and static_dim is not None and len(static_pos) > 0:
        for idx, pos in enumerate(static_pos):
            dim = static_dim[idx]
            poly = np.array(
                [
                    [pos[0] - dim[0], pos[1] - dim[1]],
                    [pos[0] - dim[0], pos[1] + dim[1]],
                    [pos[0] + dim[0], pos[1] + dim[1]],
                    [pos[0] + dim[0], pos[1] - dim[1]],
                ],
                dtype=np.float32,
            )
            poly = _rotate_poly(poly, float(pos[2]), np.asarray(pos[:2], dtype=np.float32))
            axes.fill(poly[:, 0], poly[:, 1], color="red", alpha=0.3, edgecolor="red")

    agent_current = _as_numpy(agent_feature.agent_current_state)
    agent_geo = _as_numpy(agent_feature.agent_geometry)
    agent_hist = _as_numpy(agent_feature.agent_history_state)
    agent_hist_mask = _as_numpy(agent_feature.agent_history_mask)
    if agent_current is not None and agent_geo is not None and len(agent_current) > 0:
        for idx, current in enumerate(agent_current):
            geo = agent_geo[idx]
            poly = np.array(
                [
                    [current[0] - geo[0], current[1] - geo[1]],
                    [current[0] - geo[0], current[1] + geo[1]],
                    [current[0] + geo[0], current[1] + geo[1]],
                    [current[0] + geo[0], current[1] - geo[1]],
                ],
                dtype=np.float32,
            )
            poly = _rotate_poly(poly, float(current[2]), np.asarray(current[:2], dtype=np.float32))
            axes.fill(poly[:, 0], poly[:, 1], color="orange", alpha=0.3, edgecolor="orange")

            if agent_hist is not None and agent_hist_mask is not None:
                valid_idx = np.where(agent_hist_mask[idx])[0]
                if len(valid_idx) > 1:
                    valid_history = agent_hist[idx][valid_idx]
                    axes.plot(valid_history[:, 0], valid_history[:, 1], color="purple", linestyle="--", linewidth=1.5, alpha=0.45)

    if prediction_mode == "prediction" and agent_prediction is not None:
        agent_future = _as_numpy(agent_prediction.agent_future_state)
        agent_future_mask = _as_numpy(agent_prediction.agent_future_mask)
        if agent_future is not None and agent_future_mask is not None:
            for idx, future in enumerate(agent_future):
                valid_idx = np.where(agent_future_mask[idx])[0]
                if len(valid_idx) > 1:
                    valid_future = future[valid_idx]
                    axes.plot(valid_future[:, 0], valid_future[:, 1], color="orange", linestyle="-", linewidth=1.5, alpha=0.55)


def _compute_anchor_plot_extent(trajectories: np.ndarray, chosen_trajectory: Optional[np.ndarray], expert_trajectory: Optional[np.ndarray], extent_scale: float = 1.35) -> float:
    extent = 0.0
    for candidate in (trajectories, chosen_trajectory, expert_trajectory):
        if candidate is None:
            continue
        candidate_np = np.asarray(candidate, dtype=np.float32)
        if candidate_np.size == 0:
            continue
        extent = max(extent, float(np.nanmax(np.abs(candidate_np[..., :2]))))
    return max(extent * extent_scale, 25.0)


def _save_scored_visualization(
    scene_manager: SceneManager,
    scene_feature: SceneFeature,
    agent_prediction: Optional[AgentPrediction],
    expert_trajectory: Optional[np.ndarray],
    trajectories: np.ndarray,
    chosen_trajectory: Optional[np.ndarray],
    raw_scores: Optional[np.ndarray],
    image_name: str,
    output_dir: Path,
    prediction_mode: str,
) -> int:
    if raw_scores is None:
        return -1
    raw_scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
    if raw_scores.size == 0:
        return -1
    best_idx = int(np.argmax(raw_scores))
    image = scene_manager.draw_model_in_out(
        scene_feature=scene_feature,
        chosen_trajectory=chosen_trajectory,
        all_trajectories=trajectories,
        all_trajectory_scores=raw_scores,
        agent_prediction=agent_prediction,
        prediction_mode=prediction_mode,
        expert_trajectory=expert_trajectory,
    )
    image_path = output_dir / image_name
    plt.imsave(image_path, image)
    return best_idx


def _save_anchor_overview(
    scene_feature: SceneFeature,
    agent_prediction: Optional[AgentPrediction],
    trajectories: np.ndarray,
    anchor_indices: np.ndarray,
    chosen_trajectory: Optional[np.ndarray],
    expert_trajectory: Optional[np.ndarray],
    executed_candidate_idx: int,
    output_dir: Path,
    prediction_mode: str,
) -> None:
    fig, axes = plt.subplots(figsize=(12, 12))
    axes.set_aspect("equal")
    _draw_scene_context(axes, scene_feature, agent_prediction=agent_prediction, prediction_mode=prediction_mode)

    extent = _compute_anchor_plot_extent(trajectories, chosen_trajectory, expert_trajectory, extent_scale=1.45)
    axes.set_xlim(-extent, extent)
    axes.set_ylim(-extent, extent)
    axes.set_title("Anchor Index Overview")

    trajectories = np.asarray(trajectories, dtype=np.float32)
    anchor_indices = np.asarray(anchor_indices, dtype=np.int32).reshape(-1)
    cmap = plt.get_cmap("gist_ncar")
    text_effects = [patheffects.withStroke(linewidth=3.2, foreground=(1.0, 1.0, 1.0, 0.9))]

    for idx, trajectory in enumerate(trajectories):
        color = cmap(idx / max(len(trajectories), 1))
        is_executed = idx == executed_candidate_idx
        axes.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="#dc2626" if is_executed else color,
            linewidth=3.2 if is_executed else 1.9,
            alpha=0.98 if is_executed else 0.82,
            zorder=4 if is_executed else 3,
        )
        end_point = trajectory[-1, :2]
        axes.scatter([end_point[0]], [end_point[1]], color="#dc2626" if is_executed else color, s=42, zorder=5)

        if trajectory.shape[0] > 1:
            direction = trajectory[-1, :2] - trajectory[-2, :2]
        else:
            direction = np.asarray(end_point, dtype=np.float32)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-3:
            direction = np.asarray(end_point, dtype=np.float32)
            direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-3:
            direction = np.array([1.0, 1.0], dtype=np.float32)
            direction_norm = float(np.linalg.norm(direction))
        direction = direction / max(direction_norm, 1e-6)
        offset = direction * max(0.08 * extent, 3.0)
        text_xy = end_point + offset

        annotation = axes.annotate(
            f"{int(anchor_indices[idx])}",
            xy=(float(end_point[0]), float(end_point[1])),
            xytext=(float(text_xy[0]), float(text_xy[1])),
            textcoords="data",
            fontsize=11.5,
            color="#111827",
            ha="center",
            va="center",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.8, "alpha": 0.75},
            zorder=6,
        )
        annotation.set_path_effects(text_effects)

    if expert_trajectory is not None and np.asarray(expert_trajectory).size > 0:
        expert_traj = np.asarray(expert_trajectory, dtype=np.float32)
        axes.plot(expert_traj[:, 0], expert_traj[:, 1], color="black", linewidth=2.0, alpha=0.85, linestyle="--", zorder=4)

    if chosen_trajectory is not None and np.asarray(chosen_trajectory).size > 0:
        chosen_traj = np.asarray(chosen_trajectory, dtype=np.float32)
        axes.plot(chosen_traj[:, 0], chosen_traj[:, 1], color="#dc2626", linewidth=2.4, alpha=0.95, zorder=5)

    image_path = output_dir / "anchor_indices.png"
    fig.tight_layout()
    fig.savefig(image_path, dpi=180)
    plt.close(fig)


def _save_score_report(
    output_dir: Path,
    iteration: int,
    anchor_indices: np.ndarray,
    model_scores: Optional[np.ndarray],
    eval_scores: Optional[np.ndarray],
    selection_scores: np.ndarray,
    executed_candidate_idx: int,
    best_candidate_idx: int,
    eval_details: Optional[Dict[str, Any]],
    extra_metadata: Optional[Dict[str, Any]],
) -> None:
    num_candidates = int(anchor_indices.shape[0])
    model_scores = _pad_candidate_array(model_scores, num_candidates)
    eval_scores = _pad_candidate_array(eval_scores, num_candidates)
    selection_scores = _pad_candidate_array(selection_scores, num_candidates)

    report: Dict[str, Any] = {
        "iteration": int(iteration),
        "executed_candidate_idx": int(executed_candidate_idx),
        "best_candidate_idx": int(best_candidate_idx),
        "weighted_metric_semantics": WEIGHTED_METRIC_DESCRIPTIONS,
        "multi_metric_semantics": MULTI_METRIC_DESCRIPTIONS,
        "candidates": [],
    }
    if extra_metadata:
        report.update(extra_metadata)

    multi_metrics = None
    weighted_metrics = None
    following_penalty = None
    collision_times = None
    ttc_times = None
    if eval_details is not None:
        multi_metrics = _as_numpy(eval_details.get("multi_metrics"))
        weighted_metrics = _as_numpy(eval_details.get("weighted_metrics"))
        following_penalty = _pad_candidate_array(_as_numpy(eval_details.get("following_penalty")), num_candidates)
        collision_times = _pad_candidate_array(_as_numpy(eval_details.get("collision_times")), num_candidates)
        ttc_times = _pad_candidate_array(_as_numpy(eval_details.get("ttc_times")), num_candidates)

    for idx, anchor_index in enumerate(anchor_indices.tolist()):
        candidate_entry: Dict[str, Any] = {
            "candidate_idx": int(idx),
            "anchor_index": int(anchor_index),
            "model_log_score": float(model_scores[idx]),
            "eval_score": float(eval_scores[idx]),
            "selection_score": float(selection_scores[idx]),
        }
        if following_penalty is not None:
            candidate_entry["following_penalty"] = float(following_penalty[idx])
        if collision_times is not None:
            candidate_entry["collision_time"] = float(collision_times[idx])
        if ttc_times is not None:
            candidate_entry["ttc_time"] = float(ttc_times[idx])
        if multi_metrics is not None:
            candidate_entry["multi_metrics"] = {
                metric.name: float(multi_metrics[int(metric), idx]) for metric in MultiMetricIndex
            }
        if weighted_metrics is not None:
            candidate_entry["weighted_metrics"] = {
                metric.name: float(weighted_metrics[int(metric), idx]) for metric in WeightedMetricIndex
            }
        report["candidates"].append(candidate_entry)

    report_path = output_dir / "score_report.json"
    with open(report_path, "w", encoding="utf-8") as file_obj:
        json.dump(_to_serializable(report), file_obj, ensure_ascii=False, indent=2)


def dump_planner_debug_artifacts(
    output_root: Path,
    scenario_token: str,
    iteration: int,
    scene_manager: SceneManager,
    scene_feature: SceneFeature,
    trajectories: np.ndarray,
    anchor_indices: np.ndarray,
    selection_scores: np.ndarray,
    prediction_mode: str,
    agent_prediction: Optional[AgentPrediction] = None,
    chosen_trajectory: Optional[np.ndarray] = None,
    expert_trajectory: Optional[np.ndarray] = None,
    model_scores: Optional[np.ndarray] = None,
    eval_scores: Optional[np.ndarray] = None,
    executed_candidate_idx: int = -1,
    best_candidate_idx: int = -1,
    eval_details: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    output_dir = Path(output_root) / str(scenario_token) / str(iteration)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_best_idx = _save_scored_visualization(
        scene_manager=scene_manager,
        scene_feature=scene_feature,
        agent_prediction=agent_prediction,
        expert_trajectory=expert_trajectory,
        trajectories=trajectories,
        chosen_trajectory=chosen_trajectory,
        raw_scores=model_scores,
        image_name="model_pred_score.png",
        output_dir=output_dir,
        prediction_mode=prediction_mode,
    )
    eval_best_idx = _save_scored_visualization(
        scene_manager=scene_manager,
        scene_feature=scene_feature,
        agent_prediction=agent_prediction,
        expert_trajectory=expert_trajectory,
        trajectories=trajectories,
        chosen_trajectory=chosen_trajectory,
        raw_scores=eval_scores,
        image_name="eval_score.png",
        output_dir=output_dir,
        prediction_mode=prediction_mode,
    )
    selection_best_idx = _save_scored_visualization(
        scene_manager=scene_manager,
        scene_feature=scene_feature,
        agent_prediction=agent_prediction,
        expert_trajectory=expert_trajectory,
        trajectories=trajectories,
        chosen_trajectory=chosen_trajectory,
        raw_scores=selection_scores,
        image_name="fused_score.png",
        output_dir=output_dir,
        prediction_mode=prediction_mode,
    )

    _save_anchor_overview(
        scene_feature=scene_feature,
        agent_prediction=agent_prediction,
        trajectories=trajectories,
        anchor_indices=anchor_indices,
        chosen_trajectory=chosen_trajectory,
        expert_trajectory=expert_trajectory,
        executed_candidate_idx=executed_candidate_idx,
        output_dir=output_dir,
        prediction_mode=prediction_mode,
    )
    summary_metadata = {
        "model_best_idx": int(model_best_idx),
        "eval_best_idx": int(eval_best_idx),
        "selection_best_idx": int(selection_best_idx),
    }
    if extra_metadata:
        summary_metadata.update(extra_metadata)
    _save_score_report(
        output_dir=output_dir,
        iteration=iteration,
        anchor_indices=np.asarray(anchor_indices, dtype=np.int32).reshape(-1),
        model_scores=model_scores,
        eval_scores=eval_scores,
        selection_scores=selection_scores,
        executed_candidate_idx=executed_candidate_idx,
        best_candidate_idx=best_candidate_idx,
        eval_details=eval_details,
        extra_metadata=summary_metadata,
    )
    return output_dir