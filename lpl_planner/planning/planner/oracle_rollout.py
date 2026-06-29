from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from lpl_planner.planning.planner import rollout_utils as ru
from lpl_planner.planning.planner.planner_utils import (
    _follow_velocity_target_batch,
    _rebuild_trajectories_from_arclength_batch,
    _estimate_simulator_longitudinal_profiles_batch,
)
from lpl_planner.planning.scene.evaluate.scene_scorer import BatchEvaluator
from lpl_planner.planning.scene.evaluate.simulator import BatchSimulator
from lpl_planner.planning.scene.scene_feature.features import (
    AgentPrediction,
    RolloutTeacherMetadata,
    SceneFeature,
    Trajectory,
)
from lpl_planner.planning.scene.scene_manager import SceneManager
from lpl_planner.training.dataset.dataset_utils import dump_feature_target_to_pickle


class OracleRolloutBuilder:
    def __init__(
        self,
        scenario: AbstractScenario,
        future_sampling: TrajectorySampling,
        scene_manager: SceneManager,
        simulator: BatchSimulator,
        evaluator: BatchEvaluator,
        rollout_cache_dir: Path,
        ref_path_num_points: int,
        topk: int,
        max_eval_candidates: int,
        num_samples: int,
        planner_model: Any,
        planner_device: torch.device,
        pred_logprob_weight: float,
        eval_logprob_weight: float,
        eval_score_temperature: float,
    ) -> None:
        self._scenario = scenario
        self._future_sampling = future_sampling
        self._scene_manager = scene_manager
        self._simulator = simulator
        self._evaluator = evaluator
        self._rollout_cache_dir = rollout_cache_dir
        self._ref_path_num_points = max(int(ref_path_num_points), 16)
        self._topk = max(int(topk), 0)
        self._max_eval_candidates = max(int(max_eval_candidates), 0)
        self._num_samples = max(int(num_samples), 1)
        self._planner_model = planner_model
        self._planner_device = planner_device
        self._pred_logprob_weight = float(pred_logprob_weight)
        self._eval_logprob_weight = float(eval_logprob_weight)
        self._eval_score_temperature = float(eval_score_temperature)
        self._planner_anchors = planner_model.planner_anchor.detach().cpu().numpy().astype(np.float32)

    def _augment_longitudinal_candidates(
        self,
        trajectories: np.ndarray,
        teacher_sources: np.ndarray,
        current_speed: float,
        current_acceleration: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if trajectories.shape[0] == 0:
            return trajectories, teacher_sources

        dt = max(float(self._future_sampling.interval_length), 1e-3)
        horizon = trajectories.shape[1]
        time_axis = np.arange(horizon, dtype=np.float64) * dt

        base_speed = np.maximum(np.asarray(trajectories[:, :, 3], dtype=np.float64), 0.0)
        cruise_target = np.maximum(base_speed, max(float(current_speed), 0.0))
        mild_accel_target = np.maximum(
            cruise_target,
            max(float(current_speed), 0.0) + 0.75 * time_axis[None, :],
        )

        augmented_trajectories = []
        augmented_sources = []
        for target_profile, source_offset in ((cruise_target, 10), (mild_accel_target, 20)):
            _, _, s_new, _ = _follow_velocity_target_batch(
                v_target=target_profile,
                dt=dt,
                max_acc=ru.ORACLE_MAX_LONGITUDINAL_ACCEL,
                min_acc=-1.5,
                max_jerk=ru.ORACLE_MAX_LONGITUDINAL_JERK,
                init_speed=current_speed,
                init_accel=current_acceleration,
                trajectories=np.asarray(trajectories, dtype=np.float64),
            )
            rebuilt = _rebuild_trajectories_from_arclength_batch(
                np.asarray(trajectories, dtype=np.float64),
                s_new,
            ).astype(np.float32, copy=False)
            sim_vel, sim_acc = _estimate_simulator_longitudinal_profiles_batch(
                rebuilt,
                dt=dt,
                init_speed=current_speed,
                init_accel=current_acceleration,
            )
            rebuilt[:, :, 3] = np.maximum(sim_vel[:, 1:], 0.0)
            rebuilt[:, :, 5] = sim_acc[:, 1:]

            speed_gain = rebuilt[:, -1, 3] - trajectories[:, -1, 3]
            keep_mask = speed_gain > 0.5
            if np.any(keep_mask):
                augmented_trajectories.append(rebuilt[keep_mask])
                augmented_sources.append(teacher_sources[keep_mask] + source_offset)

        if not augmented_trajectories:
            return trajectories, teacher_sources

        merged_trajectories = np.concatenate([trajectories] + augmented_trajectories, axis=0)
        merged_sources = np.concatenate([teacher_sources] + augmented_sources, axis=0)
        return merged_trajectories.astype(np.float32, copy=False), merged_sources.astype(np.int32, copy=False)

    @staticmethod
    def _softmax_normalize(values: np.ndarray, temperature: float) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return values
        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
        safe_values = values.copy()
        safe_values[~finite_mask] = np.min(safe_values[finite_mask])
        safe_values /= max(float(temperature), 1e-6)
        safe_values -= np.max(safe_values)
        probs = np.exp(safe_values)
        probs = np.where(np.isfinite(probs), probs, 0.0)
        total = probs.sum()
        if total <= 0.0:
            return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
        return probs / total

    @staticmethod
    def _expand_model_log_scores(model_log_scores: Optional[np.ndarray], target_num_candidates: int) -> np.ndarray:
        if target_num_candidates <= 0:
            return np.zeros((0,), dtype=np.float32)
        if model_log_scores is None or len(model_log_scores) == 0:
            return np.zeros((target_num_candidates,), dtype=np.float32)
        model_log_scores = np.asarray(model_log_scores, dtype=np.float32).reshape(-1)
        if model_log_scores.shape[0] >= target_num_candidates:
            return model_log_scores[:target_num_candidates]
        pad_value = float(np.min(model_log_scores) - 2.0)
        return np.pad(model_log_scores, (0, target_num_candidates - model_log_scores.shape[0]), constant_values=pad_value)

    @staticmethod
    def _normalize_teacher_scores(scores: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        normalized = np.full(scores.shape, np.inf, dtype=np.float64)
        finite_mask = np.isfinite(scores)
        if valid_mask is not None:
            finite_mask &= np.asarray(valid_mask, dtype=bool)
        if not np.any(finite_mask):
            return normalized
        valid_scores = scores[finite_mask]
        scale = float(np.median(np.abs(valid_scores)))
        scale = max(scale, 1e-6)
        normalized[finite_mask] = scores[finite_mask] / scale
        return normalized

    def _compute_expert_teacher_scores(
        self,
        anchors: np.ndarray,
        expert_path: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        anchor_pose = np.asarray(anchors[:, :, :3], dtype=ru.GLOBAL_COORD_DTYPE)
        anchor_shape = ru.align_paths_to_start_pose(anchor_pose)
        expert_target = ru.align_path_to_start_pose(np.asarray(expert_path[:, :3], dtype=ru.GLOBAL_COORD_DTYPE))
        tracking_score, tracking_valid_mask = ru.compute_path_frame_teacher_score(anchor_shape, expert_target)
        endpoint_dist = ru.compute_endpoint_path_distance(anchor_shape[:, -1, :2], expert_target[:, :2])
        endpoint_valid_mask = endpoint_dist <= ru.ORACLE_ENDPOINT_PATH_MAX_ERROR
        shape_score = ru.compute_shape_prefilter_score(anchor_shape, expert_target)
        score = tracking_score + 0.75 * endpoint_dist + 0.35 * shape_score
        score = score + (~endpoint_valid_mask).astype(ru.GLOBAL_COORD_DTYPE) * ru.ORACLE_ENDPOINT_INVALID_PENALTY
        score = score + (~tracking_valid_mask).astype(ru.GLOBAL_COORD_DTYPE) * ru.ORACLE_PATH_TRACK_INVALID_PENALTY
        teacher_valid_mask = endpoint_valid_mask & tracking_valid_mask
        return score, teacher_valid_mask

    def _compute_route_teacher_scores(
        self,
        anchors: np.ndarray,
        route_ref_path: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        anchor_pose = np.asarray(anchors[:, :, :3], dtype=ru.GLOBAL_COORD_DTYPE)
        anchor_shape = ru.align_paths_to_start_pose(anchor_pose)
        route_target = ru.align_path_to_start_pose(np.asarray(route_ref_path[:, :3], dtype=ru.GLOBAL_COORD_DTYPE))
        tracking_score, tracking_valid_mask = ru.compute_path_frame_teacher_score(anchor_shape, route_target)
        endpoint_dist = ru.compute_endpoint_path_distance(anchor_shape[:, -1, :2], route_target[:, :2])
        endpoint_valid_mask = endpoint_dist <= ru.ORACLE_ENDPOINT_PATH_MAX_ERROR
        shape_score = ru.compute_shape_prefilter_score(anchor_shape, route_target)
        score = tracking_score + 0.5 * endpoint_dist + 0.35 * shape_score
        score = score + (~endpoint_valid_mask).astype(ru.GLOBAL_COORD_DTYPE) * ru.ORACLE_ENDPOINT_INVALID_PENALTY
        score = score + (~tracking_valid_mask).astype(ru.GLOBAL_COORD_DTYPE) * ru.ORACLE_PATH_TRACK_INVALID_PENALTY
        teacher_valid_mask = endpoint_valid_mask & tracking_valid_mask
        return score, teacher_valid_mask

    def _compute_candidate_path_align_scores(
        self,
        candidate_trajectories: np.ndarray,
        candidate_teacher_sources: np.ndarray,
        expert_path: np.ndarray,
        route_ref_path: np.ndarray,
    ) -> np.ndarray:
        if candidate_trajectories.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)

        candidate_pose = np.asarray(candidate_trajectories[:, :, :3], dtype=ru.GLOBAL_COORD_DTYPE)
        candidate_shape = ru.align_paths_to_start_pose(candidate_pose)
        expert_target = ru.align_path_to_start_pose(np.asarray(expert_path[:, :3], dtype=ru.GLOBAL_COORD_DTYPE))
        route_target = ru.align_path_to_start_pose(np.asarray(route_ref_path[:, :3], dtype=ru.GLOBAL_COORD_DTYPE))
        expert_score, _ = ru.compute_path_frame_teacher_score(candidate_shape, expert_target)
        route_score, _ = ru.compute_path_frame_teacher_score(candidate_shape, route_target)

        align_scores = np.zeros((candidate_trajectories.shape[0],), dtype=np.float32)
        for idx, source in enumerate(np.asarray(candidate_teacher_sources, dtype=np.int32).tolist()):
            if source in (0, 10, 20):
                align_scores[idx] = -float(expert_score[idx])
            elif source in (1, 11, 21):
                align_scores[idx] = -float(route_score[idx])
            else:
                align_scores[idx] = -float(min(expert_score[idx], route_score[idx]))
        return align_scores

    @staticmethod
    def _compute_policy_teacher_scores(
        policy_anchor_indices: np.ndarray,
        policy_fused_scores: np.ndarray,
        anchor_num: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        teacher_scores = np.full((anchor_num,), np.inf, dtype=np.float32)
        valid_mask = np.zeros((anchor_num,), dtype=bool)
        anchor_indices = np.asarray(policy_anchor_indices, dtype=np.int64).reshape(-1)
        fused_scores = np.asarray(policy_fused_scores, dtype=np.float32).reshape(-1)
        count = min(anchor_indices.shape[0], fused_scores.shape[0])
        if count == 0:
            return teacher_scores, valid_mask
        anchor_indices = anchor_indices[:count]
        fused_scores = fused_scores[:count]
        in_bounds = (anchor_indices >= 0) & (anchor_indices < anchor_num) & np.isfinite(fused_scores)
        if not np.any(in_bounds):
            return teacher_scores, valid_mask
        anchor_indices = anchor_indices[in_bounds]
        fused_scores = fused_scores[in_bounds]
        teacher_scores[anchor_indices] = -fused_scores
        valid_mask[anchor_indices] = True
        return teacher_scores, valid_mask

    def _select_oracle_anchor_indices(
        self,
        expert_path: np.ndarray,
        route_ref_path: np.ndarray,
        policy_anchor_indices: np.ndarray,
        policy_fused_scores: np.ndarray,
        rollout_anchor_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        anchor_bank = self._planner_anchors
        anchor_num = anchor_bank.shape[0]
        teacher_topk = self._topk if self._topk > 0 else int(np.ceil(self._num_samples / 3.0))
        teacher_topk = min(max(teacher_topk, 1), anchor_num)
        max_eval_candidates = self._max_eval_candidates if self._max_eval_candidates > 0 else max(self._num_samples, teacher_topk)
        max_eval_candidates = min(max(max_eval_candidates, 1), anchor_num)

        expert_scores, expert_valid_mask = self._compute_expert_teacher_scores(
            anchors=anchor_bank,
            expert_path=expert_path,
        )
        route_scores, route_valid_mask = self._compute_route_teacher_scores(
            anchors=anchor_bank,
            route_ref_path=route_ref_path,
        )
        policy_scores, policy_valid_mask = self._compute_policy_teacher_scores(
            policy_anchor_indices=policy_anchor_indices,
            policy_fused_scores=policy_fused_scores,
            anchor_num=anchor_num,
        )

        selected: list[int] = []
        used: set[int] = set()
        source_by_index: Dict[int, int] = {}

        for idx in np.asarray(np.argsort(expert_scores), dtype=np.int64).tolist():
            if not bool(expert_valid_mask[idx]) or idx in used:
                continue
            selected.append(int(idx))
            used.add(int(idx))
            source_by_index[int(idx)] = 0
            if sum(1 for item in selected if source_by_index.get(item, -1) == 0) >= teacher_topk:
                break

        for idx in np.asarray(np.argsort(route_scores), dtype=np.int64).tolist():
            if not bool(route_valid_mask[idx]) or idx in used:
                continue
            selected.append(int(idx))
            used.add(int(idx))
            source_by_index[int(idx)] = 1
            if sum(1 for item in selected if source_by_index.get(item, -1) == 1) >= teacher_topk:
                break

        for idx in np.asarray(np.argsort(policy_scores), dtype=np.int64).tolist():
            if not bool(policy_valid_mask[idx]) or idx in used:
                continue
            selected.append(int(idx))
            used.add(int(idx))
            source_by_index[int(idx)] = 2
            if sum(1 for item in selected if source_by_index.get(item, -1) == 2) >= teacher_topk:
                break

        if rollout_anchor_index >= 0 and rollout_anchor_index not in used:
            selected.append(int(rollout_anchor_index))
            used.add(int(rollout_anchor_index))
            source_by_index[int(rollout_anchor_index)] = 3

        if not selected:
            for idx in np.asarray(np.argsort(expert_scores), dtype=np.int64).tolist()[:teacher_topk]:
                if idx in used:
                    continue
                selected.append(int(idx))
                used.add(int(idx))
                source_by_index[int(idx)] = 0
            for idx in np.asarray(np.argsort(route_scores), dtype=np.int64).tolist()[:teacher_topk]:
                if idx in used:
                    continue
                selected.append(int(idx))
                used.add(int(idx))
                source_by_index[int(idx)] = 1

        selected_indices = np.asarray(selected, dtype=np.int64)
        expert_norm = self._normalize_teacher_scores(expert_scores, expert_valid_mask)
        route_norm = self._normalize_teacher_scores(route_scores, route_valid_mask)
        policy_norm = self._normalize_teacher_scores(policy_scores, policy_valid_mask)
        priority = np.minimum(expert_norm[selected_indices], route_norm[selected_indices])
        priority = np.minimum(priority, policy_norm[selected_indices])
        order = np.argsort(priority, kind="stable")[:max_eval_candidates]
        selected_indices = selected_indices[order]
        selected_sources = np.asarray([source_by_index[int(idx)] for idx in selected_indices], dtype=np.int32)
        return selected_indices, selected_sources

    def _compute_oracle_selection_scores(
        self,
        candidate_eval_scores: np.ndarray,
        candidate_model_log_scores: np.ndarray,
        candidate_path_align_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        num_candidates = candidate_model_log_scores.shape[0]
        model_log_scores = self._expand_model_log_scores(candidate_model_log_scores, num_candidates)
        if candidate_eval_scores is None or len(candidate_eval_scores) != num_candidates:
            return np.asarray(model_log_scores, dtype=np.float32)

        eval_scores = np.asarray(candidate_eval_scores, dtype=np.float32)
        eval_centered = eval_scores - np.max(eval_scores)
        model_centered = model_log_scores - np.max(model_log_scores)
        align_centered = np.zeros_like(eval_centered)
        if candidate_path_align_scores is not None and len(candidate_path_align_scores) == num_candidates:
            align_scores = np.asarray(candidate_path_align_scores, dtype=np.float32)
            align_centered = align_scores - np.max(align_scores)
        selection_scores = eval_centered + 0.20 * align_centered + 0.05 * model_centered
        return np.asarray(selection_scores, dtype=np.float32)

    def _score_oracle_anchor_subset(
        self,
        scene_feature_tensor: SceneFeature,
        anchor_indices: np.ndarray,
        fallback_model_logp: Optional[np.ndarray],
    ) -> np.ndarray:
        if anchor_indices.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        valid_anchor_mask = np.asarray(anchor_indices, dtype=np.int64) >= 0
        scores = np.full((anchor_indices.shape[0],), -8.0, dtype=np.float32)
        if hasattr(self._planner_model, "score_candidate_subset") and np.any(valid_anchor_mask):
            valid_anchor_indices = np.asarray(anchor_indices[valid_anchor_mask], dtype=np.int64)
            anchor_index_tensor = torch.from_numpy(valid_anchor_indices.astype(np.int64, copy=False)).unsqueeze(0).to(self._planner_device)
            score_out = self._planner_model.score_candidate_subset(
                scene_features=scene_feature_tensor,
                anchor_indices=anchor_index_tensor,
                no_grad=True,
                return_prediction=False,
            )
            scores[valid_anchor_mask] = score_out["candidate_log_probs"].detach().cpu().numpy()[0].astype(np.float32, copy=False)
            return scores
        if fallback_model_logp is None or len(fallback_model_logp) == 0:
            return scores
        fallback_scores = np.asarray(fallback_model_logp, dtype=np.float32).reshape(-1)
        if np.any(valid_anchor_mask):
            scores[valid_anchor_mask] = fallback_scores[np.asarray(anchor_indices[valid_anchor_mask], dtype=np.int64)]
        return scores

    def _evaluate_oracle_anchor_subset(
        self,
        anchor_trajectories: np.ndarray,
        ego_state: EgoState,
        scene_feature: SceneFeature,
        agent_prediction: Optional[AgentPrediction],
        route_ref_path: np.ndarray,
    ) -> np.ndarray:
        if anchor_trajectories.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        current_state = np.zeros((anchor_trajectories.shape[0], 1, anchor_trajectories.shape[-1]), dtype=np.float32)
        current_state[:, 0, 3] = ego_state.dynamic_car_state.speed
        current_state[:, 0, 4] = ego_state.dynamic_car_state.acceleration
        current_state[:, 0, 5] = ego_state.dynamic_car_state.angular_velocity
        extended_trajectories = np.concatenate((current_state, anchor_trajectories.astype(np.float32, copy=False)), axis=1)
        simulated_trajectories = self._simulator.simulate(extended_trajectories, ego_state=ego_state)
        scores_eval = self._evaluator.batch_evaluate(
            simulated_trajectories,
            scene_feature=scene_feature,
            agent_prediction_gt=agent_prediction,
            discount_factor=1.0,
            ref_path=route_ref_path,
            prediction_mode="CYAW",
        )
        return np.asarray(scores_eval["aggregate_scores"], dtype=np.float32)

    def build_rollout_oracle(
        self,
        scene_feature_tensor: SceneFeature,
        scene_feature: SceneFeature,
        ego_state: EgoState,
        agent_prediction: Optional[AgentPrediction],
        rollout_anchor_index: int,
        rollout_score: float,
        rollout_model_logp: Optional[np.ndarray],
        rollout_policy_indices: np.ndarray,
        rollout_policy_fused_scores: np.ndarray,
        expert_path: np.ndarray,
        expert_path_local: np.ndarray,
        route_ref_path: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        candidate_anchor_indices, candidate_teacher_sources = self._select_oracle_anchor_indices(
            expert_path=expert_path,
            route_ref_path=route_ref_path,
            policy_anchor_indices=rollout_policy_indices,
            policy_fused_scores=rollout_policy_fused_scores,
            rollout_anchor_index=rollout_anchor_index,
        )
        candidate_trajectories = self._planner_anchors[candidate_anchor_indices]
        candidate_trajectories, candidate_teacher_sources = self._augment_longitudinal_candidates(
            trajectories=candidate_trajectories,
            teacher_sources=candidate_teacher_sources,
            current_speed=float(ego_state.dynamic_car_state.speed),
            current_acceleration=float(ego_state.dynamic_car_state.acceleration),
        )
        if candidate_trajectories.shape[0] > candidate_anchor_indices.shape[0]:
            extra_num = candidate_trajectories.shape[0] - candidate_anchor_indices.shape[0]
            candidate_anchor_indices = np.concatenate(
                [candidate_anchor_indices, np.full((extra_num,), -1, dtype=np.int64)],
                axis=0,
            )
        candidate_model_log_scores = self._score_oracle_anchor_subset(scene_feature_tensor, candidate_anchor_indices, rollout_model_logp)
        if candidate_model_log_scores.shape[0] < candidate_trajectories.shape[0]:
            extra_num = candidate_trajectories.shape[0] - candidate_model_log_scores.shape[0]
            candidate_model_log_scores = np.concatenate(
                [candidate_model_log_scores, np.full((extra_num,), np.min(candidate_model_log_scores) - 4.0 if candidate_model_log_scores.size > 0 else -8.0, dtype=np.float32)],
                axis=0,
            )
        candidate_eval_scores = self._evaluate_oracle_anchor_subset(
            anchor_trajectories=candidate_trajectories,
            ego_state=ego_state,
            scene_feature=scene_feature,
            agent_prediction=agent_prediction,
            route_ref_path=route_ref_path,
        )
        candidate_path_align_scores = self._compute_candidate_path_align_scores(
            candidate_trajectories=candidate_trajectories,
            candidate_teacher_sources=candidate_teacher_sources,
            expert_path=expert_path,
            route_ref_path=route_ref_path,
        )
        candidate_selection_scores = self._compute_oracle_selection_scores(
            candidate_eval_scores=candidate_eval_scores,
            candidate_model_log_scores=candidate_model_log_scores,
            candidate_path_align_scores=candidate_path_align_scores,
        )
        keep_num = min(16, candidate_anchor_indices.shape[0])
        if keep_num > 0 and candidate_anchor_indices.shape[0] > keep_num:
            keep_indices = np.argsort(candidate_selection_scores)[-keep_num:][::-1]
            candidate_anchor_indices = candidate_anchor_indices[keep_indices]
            candidate_teacher_sources = candidate_teacher_sources[keep_indices]
            candidate_trajectories = candidate_trajectories[keep_indices]
            candidate_model_log_scores = candidate_model_log_scores[keep_indices]
            candidate_eval_scores = candidate_eval_scores[keep_indices]
            candidate_selection_scores = candidate_selection_scores[keep_indices]
        return {
            "candidate_anchor_indices": candidate_anchor_indices.astype(np.int32, copy=False),
            "candidate_teacher_sources": candidate_teacher_sources.astype(np.int32, copy=False),
            "candidate_trajectories": candidate_trajectories.astype(np.float32, copy=False),
            "candidate_model_log_scores": candidate_model_log_scores.astype(np.float32, copy=False),
            "candidate_eval_scores": candidate_eval_scores.astype(np.float32, copy=False),
            "candidate_selection_scores": candidate_selection_scores.astype(np.float32, copy=False),
            "expert_path_local": np.asarray(expert_path_local, dtype=np.float32),
            "expert_route_ref_path": np.asarray(route_ref_path, dtype=np.float32),
            "rollout_score": np.asarray([rollout_score], dtype=np.float32),
        }

    def save_rollout_sample(
        self,
        scene_feature: SceneFeature,
        agent_prediction: Optional[AgentPrediction],
        ego_state: EgoState,
        sampled_trajectories: np.ndarray,
        sampled_anchor_indices: np.ndarray,
        sampled_teacher_sources: np.ndarray,
        sampled_model_log_scores: np.ndarray,
        sampled_eval_scores: np.ndarray,
        sampled_selection_scores: np.ndarray,
        chosen_local_index: int,
        chosen_anchor_index: int,
        chosen_score: float,
        emergency_brake: bool,
        expert_path_local: np.ndarray,
        expert_route_ref_path: np.ndarray,
        iteration: int,
    ) -> None:
        log_name = str(getattr(self._scenario, "log_name", "rollout"))
        token = str(getattr(self._scenario, "token", f"iter_{iteration:04d}"))
        dump_dir = self._rollout_cache_dir / log_name / token / f"iter_{iteration:04d}"
        dump_dir.mkdir(parents=True, exist_ok=True)

        dump_feature_target_to_pickle(dump_dir / "scene_feature.gz", scene_feature.serialize())
        if agent_prediction is None:
            agent_prediction = AgentPrediction(
                agent_future_state=np.zeros((0, self._future_sampling.num_poses, 5), dtype=np.float32),
                agent_future_mask=np.zeros((0, self._future_sampling.num_poses), dtype=np.float32),
            )
        dump_feature_target_to_pickle(dump_dir / "agent_prediction.gz", agent_prediction.serialize())
        dump_feature_target_to_pickle(
            dump_dir / "expert_trajectory.gz",
            Trajectory(data=np.asarray(expert_path_local, dtype=np.float32)).serialize(),
        )

        planner_ref_path = self._scene_manager.lane_map.get_ref_path_feature(ref_path_num_points=self._ref_path_num_points)
        rollout_metadata = RolloutTeacherMetadata(
            sampled_anchor_indices=np.asarray(sampled_anchor_indices, dtype=np.int32),
            sampled_teacher_sources=np.asarray(sampled_teacher_sources, dtype=np.int32),
            sampled_trajectories=np.asarray(sampled_trajectories, dtype=np.float32),
            sampled_model_log_scores=np.asarray(sampled_model_log_scores, dtype=np.float32),
            sampled_eval_scores=np.asarray(sampled_eval_scores, dtype=np.float32),
            sampled_selection_scores=np.asarray(sampled_selection_scores, dtype=np.float32),
            chosen_local_index=np.asarray([chosen_local_index], dtype=np.int32),
            chosen_anchor_index=np.asarray([chosen_anchor_index], dtype=np.int32),
            chosen_score=np.asarray([chosen_score], dtype=np.float32),
            planner_ref_path=planner_ref_path.astype(np.float32, copy=False),
            expert_path=np.asarray(expert_path_local, dtype=np.float32),
            expert_ref_path=np.asarray(expert_route_ref_path, dtype=np.float32),
            timestamp_us=np.asarray([ego_state.time_point.time_us], dtype=np.int64),
            iteration=np.asarray([iteration], dtype=np.int32),
            emergency_brake=np.asarray([int(emergency_brake)], dtype=np.int32),
        )
        dump_feature_target_to_pickle(dump_dir / "rollout_teacher_metadata.gz", rollout_metadata.serialize())