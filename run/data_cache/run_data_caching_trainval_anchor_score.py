# RM_data_caching.py
# This module appends anchor scores / indices to already cached train/val samples.

from typing import Any, Dict, List, Optional
from pathlib import Path
import gc
import importlib.util
import logging
import multiprocessing as mp
import os
import resource
import sys
import time
import uuid

import hydra
import numpy as np
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from lpl_planner.planning.scene.scene_feature.features.agent_prediction import AgentPrediction
from nuplan.planning.script.builders.utils.utils_type import is_target_type, validate_type
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from nuplan.planning.utils.multithreading.worker_ray import RayDistributed
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from lpl_planner.planning.scene.evaluate.scene_scorer import BatchEvaluator
from lpl_planner.planning.scene.evaluate.simulator import BatchSimulator
from lpl_planner.planning.scene.scene_feature.features import (
    AnchorIndice,
    AnchorScores,
    SceneFeature,
    Trajectory,
)
from lpl_planner.planning.scene.trajectory_library import TrajectoryState
from lpl_planner.training.dataset import (
    dump_feature_target_to_pickle,
    load_feature_target_from_pickle,
)
from lpl_planner.training.dataset.scene_token_json import (
    DEFAULT_SCENE_TOKENS_JSON_NAME,
    load_scene_token_selection,
    resolve_scene_tokens_path,
)
from lpl_planner.utils.default_paths import configure_default_paths


configure_default_paths()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

hybrid_planner_spec = importlib.util.find_spec("hybrid_planner")
hybrid_planner_dir = os.path.dirname(hybrid_planner_spec.origin)

CONFIG_PATH = os.path.join(hybrid_planner_dir, "config/training")
CONFIG_NAME = "custom_caching"

PLANNER_ANCHOR_PATH = Path(os.environ["R2LPL_RESULTS_ROOT"]) / "planner_anchors"
PLANNER_ANCHOR_FILE = PLANNER_ANCHOR_PATH / "planner_anchors_M4096s_T4.0_step20_full.npy"
ANCHOR_SCORE_FILE = "anchor_scores.gz"
ANCHOR_INDICE_FILE = "anchor_indice.gz"

ANCHOR_TIME_HORIZON = 4.0
ANCHOR_NUM_POSES = 20
MAX_EVAL_ANCHORS = 128
EVAL_BATCH_SIZE = 64
BENCHMARK_SCENARIO_LIMIT = 1000
EXPERT_TRAJECTORY_INTERVAL_S = 0.2
EXPERT_STATIONARY_CHECK_TIME_S = 2.0
EXPERT_STATIONARY_DISPLACEMENT_THRESHOLD_M = 1.0


def _get_process_rss_mb() -> float:
    page_size = os.sysconf("SC_PAGE_SIZE")
    with open("/proc/self/statm", "r", encoding="utf-8") as statm_file:
        resident_pages = int(statm_file.readline().split()[1])
    return resident_pages * page_size / (1024.0 ** 2)


def _get_process_peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _summarize_metric(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    values_np = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values_np.mean()),
        "p50": float(np.percentile(values_np, 50)),
        "p95": float(np.percentile(values_np, 95)),
        "max": float(values_np.max()),
    }


def _get_zero_anchor_index(planner_anchor: np.ndarray) -> int:
    anchor_abs_sum = np.sum(np.abs(planner_anchor[:, :, :6]), axis=(1, 2))
    zero_anchor_index = int(np.argmin(anchor_abs_sum))
    if not np.isclose(anchor_abs_sum[zero_anchor_index], 0.0):
        logger.warning(
            "Planner anchor file does not contain an exact zero anchor; using closest anchor index=%d (sum_abs=%.6f)",
            zero_anchor_index,
            float(anchor_abs_sum[zero_anchor_index]),
        )
    return zero_anchor_index


def _is_expert_stationary_at_time(
    expert_trajectory: np.ndarray,
    check_time_s: float = EXPERT_STATIONARY_CHECK_TIME_S,
    displacement_threshold_m: float = EXPERT_STATIONARY_DISPLACEMENT_THRESHOLD_M,
    trajectory_interval_s: float = EXPERT_TRAJECTORY_INTERVAL_S,
) -> tuple[bool, float, int]:
    if expert_trajectory.shape[0] < 2:
        return True, 0.0, 0

    check_index = min(
        max(int(round(check_time_s / trajectory_interval_s)), 1),
        expert_trajectory.shape[0] - 1,
    )
    origin_xy = expert_trajectory[0, [TrajectoryState.X, TrajectoryState.Y]].astype(np.float32, copy=False)
    check_xy = expert_trajectory[check_index, [TrajectoryState.X, TrajectoryState.Y]].astype(np.float32, copy=False)
    displacement_m = float(np.linalg.norm(check_xy - origin_xy))
    return displacement_m < displacement_threshold_m, displacement_m, check_index


def _run_benchmark(
    scenario_paths: List[Path],
    benchmark_limit: int = BENCHMARK_SCENARIO_LIMIT,
) -> None:
    benchmark_paths = scenario_paths[: min(len(scenario_paths), benchmark_limit)]
    if not benchmark_paths:
        logger.info("No cached scenarios found for benchmark.")
        return

    logger.info("Starting benchmark on %d cached scenarios...", len(benchmark_paths))

    future_sampling = TrajectorySampling(num_poses=ANCHOR_NUM_POSES, time_horizon=ANCHOR_TIME_HORIZON)
    simulator = BatchSimulator(future_sampling)
    evaluator = BatchEvaluator(future_sampling,
                               use_following_penalty=True)
    planner_anchor = np.load(PLANNER_ANCHOR_FILE, mmap_mode="r")

    scenario_times_ms: List[float] = []
    rss_deltas_mb: List[float] = []
    peak_deltas_mb: List[float] = []
    rss_after_mb: List[float] = []
    failures = 0

    benchmark_start = time.perf_counter()
    benchmark_peak_before = _get_process_peak_rss_mb()

    for scenario_idx, scenario_cache_dir in enumerate(tqdm(benchmark_paths, desc="Benchmarking Scenarios"), start=1):
        scene_feature_path = scenario_cache_dir / "scene_feature.gz"
        agent_prediction_path = scenario_cache_dir / "agent_prediction.gz"
        expert_trajectory_path = scenario_cache_dir / "expert_trajectory.gz"

        if not scene_feature_path.exists() or not expert_trajectory_path.exists() or not agent_prediction_path.exists():
            failures += 1
            continue

        rss_before = _get_process_rss_mb()
        peak_before = _get_process_peak_rss_mb()
        start_time = time.perf_counter()
        scene_feature = None
        agent_prediction = None
        expert_trajectory_feature = None
        expert_trajectory = None

        try:
            scene_feature = load_feature_target_from_pickle(
                scene_feature_path,
                feature_type=SceneFeature,
            )
            agent_prediction = load_feature_target_from_pickle(
                agent_prediction_path,
                feature_type=AgentPrediction,
            )
            expert_trajectory_feature = load_feature_target_from_pickle(
                expert_trajectory_path,
                feature_type=Trajectory,
            )
            expert_trajectory = np.asarray(expert_trajectory_feature.data, dtype=np.float32)

            _evaluate_anchor_subset(
                planner_anchor=planner_anchor,
                expert_trajectory=expert_trajectory,
                scene_feature=scene_feature,
                agent_prediction=agent_prediction,
                simulator=simulator,
                evaluator=evaluator,
            )
        except Exception as exc:
            failures += 1
            logger.info("Benchmark failed for %s: %s", scenario_cache_dir.name, str(exc))
            continue
        finally:
            del scene_feature, agent_prediction, expert_trajectory_feature, expert_trajectory

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        rss_after = _get_process_rss_mb()
        peak_after = _get_process_peak_rss_mb()

        scenario_times_ms.append(elapsed_ms)
        rss_deltas_mb.append(rss_after - rss_before)
        peak_deltas_mb.append(peak_after - peak_before)
        rss_after_mb.append(rss_after)

        if scenario_idx % 32 == 0:
            gc.collect()

    total_elapsed_s = time.perf_counter() - benchmark_start
    benchmark_peak_after = _get_process_peak_rss_mb()

    time_summary = _summarize_metric(scenario_times_ms)
    rss_delta_summary = _summarize_metric(rss_deltas_mb)
    peak_delta_summary = _summarize_metric(peak_deltas_mb)
    rss_after_summary = _summarize_metric(rss_after_mb)

    logger.info("=" * 80)
    logger.info("Benchmark Summary (current code only)")
    logger.info("Scenarios attempted: %d", len(benchmark_paths))
    logger.info("Scenarios succeeded: %d", len(scenario_times_ms))
    logger.info("Scenarios failed/skipped: %d", failures)
    logger.info("Total elapsed: %.3f s", total_elapsed_s)
    logger.info(
        "Per-scenario time [ms]: mean=%.3f p50=%.3f p95=%.3f max=%.3f",
        time_summary["mean"],
        time_summary["p50"],
        time_summary["p95"],
        time_summary["max"],
    )
    logger.info(
        "Per-scenario RSS delta [MB]: mean=%.3f p50=%.3f p95=%.3f max=%.3f",
        rss_delta_summary["mean"],
        rss_delta_summary["p50"],
        rss_delta_summary["p95"],
        rss_delta_summary["max"],
    )
    logger.info(
        "Per-scenario peak RSS delta [MB]: mean=%.3f p50=%.3f p95=%.3f max=%.3f",
        peak_delta_summary["mean"],
        peak_delta_summary["p50"],
        peak_delta_summary["p95"],
        peak_delta_summary["max"],
    )
    logger.info(
        "Resident RSS after scenario [MB]: mean=%.3f p50=%.3f p95=%.3f max=%.3f",
        rss_after_summary["mean"],
        rss_after_summary["p50"],
        rss_after_summary["p95"],
        rss_after_summary["max"],
    )
    logger.info(
        "Process peak RSS over benchmark [MB]: before=%.3f after=%.3f delta=%.3f",
        benchmark_peak_before,
        benchmark_peak_after,
        benchmark_peak_after - benchmark_peak_before,
    )
    logger.info("=" * 80)


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _huber(x: np.ndarray, delta: float = 1.0) -> np.ndarray:
    abs_x = np.abs(x)
    return np.where(abs_x <= delta, 0.5 * abs_x ** 2, delta * (abs_x - 0.5 * delta))


def _resample_trajectories_by_arclength_batch(
    trajectories: np.ndarray,
    target_s: np.ndarray,
) -> np.ndarray:
    batch_size, num_steps, _ = trajectories.shape
    xy = trajectories[:, :, :2]

    seg_len = np.linalg.norm(np.diff(xy, axis=1), axis=-1)
    seg_len = np.maximum(seg_len, 1e-3)
    cum_s = np.concatenate(
        [np.zeros((batch_size, 1), dtype=np.float32), np.cumsum(seg_len, axis=1, dtype=np.float32)],
        axis=1,
    )

    right = np.sum(cum_s[:, :, None] < target_s[:, None, :], axis=1)
    right = np.clip(right, 1, num_steps - 1)
    left = right - 1

    batch_idx = np.arange(batch_size)[:, None]
    s_left = cum_s[batch_idx, left]
    s_right = cum_s[batch_idx, right]
    denom = np.maximum(s_right - s_left, 1e-6)
    interp_weight = ((target_s - s_left) / denom)[..., None]

    traj_left = trajectories[batch_idx, left]
    traj_right = trajectories[batch_idx, right]
    resampled = traj_left + interp_weight.astype(trajectories.dtype, copy=False) * (traj_right - traj_left)

    yaw = np.unwrap(trajectories[:, :, 2], axis=1)
    yaw_left = yaw[batch_idx, left]
    yaw_right = yaw[batch_idx, right]
    yaw_interp = yaw_left + (target_s - s_left) / denom * (yaw_right - yaw_left)
    resampled[:, :, 2] = _wrap_angle(yaw_interp)

    return resampled.astype(trajectories.dtype, copy=False)


def _resample_trajectory_by_arclength(
    trajectory: np.ndarray,
    num_samples: int,
) -> np.ndarray:
    xy = trajectory[:, :2]
    segment_length = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    segment_length = np.maximum(segment_length, 1e-3)
    arclength = np.concatenate(([0.0], np.cumsum(segment_length, dtype=np.float32)))

    if arclength[-1] < 1e-3:
        return np.repeat(trajectory[:1], num_samples, axis=0)

    target_s = np.linspace(0.0, arclength[-1], num_samples, dtype=np.float32)
    return _resample_trajectories_by_arclength_batch(
        trajectory[None, :, :], target_s[None, :]
    )[0]


def _compute_expert_frame_errors_batch(
    expert_traj: np.ndarray,
    candidate_trajs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expert_xy = expert_traj[:, :2]
    tangent = np.zeros_like(expert_xy)
    tangent[1:-1] = expert_xy[2:] - expert_xy[:-2]
    tangent[0] = expert_xy[1] - expert_xy[0]
    tangent[-1] = expert_xy[-1] - expert_xy[-2]

    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent_norm = np.maximum(tangent_norm, 1e-6)
    tangent = tangent / tangent_norm
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)

    delta_xy = candidate_trajs[:, :, :2] - expert_traj[None, :, :2]
    e_lon = np.sum(delta_xy * tangent[None, :, :], axis=-1)
    e_lat = np.sum(delta_xy * normal[None, :, :], axis=-1)
    e_yaw = _wrap_angle(candidate_trajs[:, :, 2] - expert_traj[None, :, 2])

    return e_lon, e_lat, e_yaw


def _compute_anchor_prefilter_score(
    expert_traj_local: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    expert = expert_traj_local[:, :6]
    _, num_steps, _ = anchors.shape

    expert_end_y = expert[-1, 1]
    expert_end_yaw = float(_wrap_angle(expert[-1, 2] - expert[0, 2]))
    expert_path_len = float(np.sum(np.linalg.norm(np.diff(expert[:, :2], axis=0), axis=-1)))
    expert_turn_energy = float(np.sum(np.abs(_wrap_angle(np.diff(expert[:, 2])))))

    anchor_end_y = anchors[:, -1, 1]
    anchor_end_yaw = _wrap_angle(anchors[:, -1, 2] - anchors[:, 0, 2])
    anchor_path_len = np.sum(np.linalg.norm(np.diff(anchors[:, :, :2], axis=1), axis=-1), axis=1)
    anchor_turn_energy = np.sum(np.abs(_wrap_angle(np.diff(anchors[:, :, 2], axis=1))), axis=1)

    checkpoint_ids = np.array(
        [
            max(1, int(0.25 * (num_steps - 1))),
            max(1, int(0.50 * (num_steps - 1))),
            max(1, int(0.75 * (num_steps - 1))),
        ],
        dtype=np.int64,
    )

    expert_y_ckpt = expert[checkpoint_ids, 1]
    expert_yaw_ckpt = expert[checkpoint_ids, 2]
    anchor_y_ckpt = anchors[:, checkpoint_ids, 1]
    anchor_yaw_ckpt = anchors[:, checkpoint_ids, 2]

    y_ckpt_cost = np.mean(np.abs(anchor_y_ckpt - expert_y_ckpt[None, :]), axis=1)
    yaw_ckpt_cost = np.mean(
        np.abs(_wrap_angle(anchor_yaw_ckpt - expert_yaw_ckpt[None, :])),
        axis=1,
    )

    turn_ratio = np.clip(
        abs(float(_wrap_angle(expert[-1, 2] - expert[0, 2]))) / 0.35,
        0.0,
        1.0,
    )

    w_end_y = 3.0 - 1.8 * turn_ratio
    w_ckpt_y = 2.0 - 1.0 * turn_ratio
    w_end_yaw = 2.5 + 1.0 * turn_ratio
    w_turn = 1.5 + 0.8 * turn_ratio
    w_path = 0.5

    return (
        w_end_y * np.abs(anchor_end_y - expert_end_y)
        + w_end_yaw * np.abs(_wrap_angle(anchor_end_yaw - expert_end_yaw))
        + w_turn * np.abs(anchor_turn_energy - expert_turn_energy)
        + w_path * np.abs(anchor_path_len - expert_path_len)
        + w_ckpt_y * y_ckpt_cost
        + yaw_ckpt_cost
    )


def _rank_anchors_by_expert_shape(
    expert_traj_local: np.ndarray,
    planner_anchor: np.ndarray,
    preselect_k: int = MAX_EVAL_ANCHORS,
) -> tuple[np.ndarray, np.ndarray]:
    num_steps = expert_traj_local.shape[0]
    anchors = planner_anchor[:, :num_steps, :6]
    num_anchors = anchors.shape[0]

    prefilter_scores = _compute_anchor_prefilter_score(
        expert_traj_local=expert_traj_local,
        anchors=anchors,
    )

    preselect_k = min(max(preselect_k, 1), num_anchors)
    coarse_top_idx = np.argpartition(prefilter_scores, preselect_k - 1)[:preselect_k]
    coarse_anchors = anchors[coarse_top_idx]

    num_resample = max(2 * num_steps, 32)
    expert_rs = _resample_trajectory_by_arclength(expert_traj_local[:, :6], num_resample)

    coarse_xy = coarse_anchors[:, :, :2]
    coarse_seg_len = np.linalg.norm(np.diff(coarse_xy, axis=1), axis=-1)
    coarse_seg_len = np.maximum(coarse_seg_len, 1e-3)
    coarse_cum_s = np.concatenate(
        [np.zeros((preselect_k, 1), dtype=np.float32), np.cumsum(coarse_seg_len, axis=1, dtype=np.float32)],
        axis=1,
    )
    total_len = np.maximum(coarse_cum_s[:, -1], 1e-3)
    target_s = np.linspace(0.0, 1.0, num_resample, dtype=np.float32)[None, :] * total_len[:, None]

    coarse_rs = _resample_trajectories_by_arclength_batch(coarse_anchors, target_s)
    e_lon, e_lat, e_yaw = _compute_expert_frame_errors_batch(
        expert_traj=expert_rs,
        candidate_trajs=coarse_rs,
    )

    weights = np.linspace(0.8, 1.2, num_resample, dtype=np.float32)
    weights[-max(4, num_resample // 6):] *= 2.0
    weights = weights / weights.sum()

    lat_cost = _huber(e_lat / 0.60, delta=1.0)
    lon_cost = _huber(e_lon / 2.50, delta=1.0)
    yaw_cost = _huber((2.0 * np.sin(0.5 * e_yaw)) / 0.20, delta=1.0)

    turn_ratio = np.clip(
        abs(float(_wrap_angle(expert_traj_local[-1, 2] - expert_traj_local[0, 2]))) / 0.35,
        0.0,
        1.0,
    )

    w_lat = 2.5 - 0.7 * turn_ratio
    w_lon = 0.8
    w_yaw = 1.5 + 0.8 * turn_ratio

    w_terminal_lat = 4.0 - 1.8 * turn_ratio
    w_terminal_lon = 1.0
    w_terminal_yaw = 3.0 + 1.5 * turn_ratio

    point_cost = (
        w_lat * np.sum(lat_cost * weights[None, :], axis=1)
        + w_lon * np.sum(lon_cost * weights[None, :], axis=1)
        + w_yaw * np.sum(yaw_cost * weights[None, :], axis=1)
    )

    terminal_cost = (
        w_terminal_lat * np.abs(e_lat[:, -1])
        + w_terminal_lon * np.abs(e_lon[:, -1])
        + w_terminal_yaw * np.abs(e_yaw[:, -1])
    )

    coarse_score_norm = prefilter_scores[coarse_top_idx]
    coarse_score_norm = coarse_score_norm / np.maximum(np.median(coarse_score_norm), 1e-6)
    fine_scores = point_cost + terminal_cost + 0.15 * coarse_score_norm

    geometry_scores = np.full(num_anchors, np.inf, dtype=np.float32)
    geometry_scores[coarse_top_idx] = fine_scores
    sorted_anchor_indices = np.argsort(geometry_scores)
    return sorted_anchor_indices, geometry_scores


def _score_anchors_by_expert_traj_numpy(
    expert_traj_local: np.ndarray,
    planner_anchor: np.ndarray,
    regularization_weight: float = 0.15,
) -> np.ndarray:
    anchors = np.asarray(
        planner_anchor[:, : expert_traj_local.shape[0], :6],
        dtype=np.float32,
    )
    num_anchors, num_steps, _ = anchors.shape

    prefilter_scores = _compute_anchor_prefilter_score(
        expert_traj_local=expert_traj_local,
        anchors=anchors,
    )

    num_resample = max(2 * num_steps, 32)
    expert_rs = _resample_trajectory_by_arclength(expert_traj_local[:, :6], num_resample)

    coarse_xy = anchors[:, :, :2]
    coarse_seg_len = np.linalg.norm(np.diff(coarse_xy, axis=1), axis=-1)
    coarse_seg_len = np.maximum(coarse_seg_len, 1e-3)
    coarse_cum_s = np.concatenate(
        [np.zeros((num_anchors, 1), dtype=np.float32), np.cumsum(coarse_seg_len, axis=1, dtype=np.float32)],
        axis=1,
    )
    total_len = np.maximum(coarse_cum_s[:, -1], 1e-3)
    target_s = np.linspace(0.0, 1.0, num_resample, dtype=np.float32)[None, :] * total_len[:, None]

    anchor_rs = _resample_trajectories_by_arclength_batch(anchors, target_s)
    e_lon, e_lat, e_yaw = _compute_expert_frame_errors_batch(
        expert_traj=expert_rs,
        candidate_trajs=anchor_rs,
    )

    weights = np.linspace(0.8, 1.2, num_resample, dtype=np.float32)
    weights[-max(4, num_resample // 6):] *= 2.0
    weights = weights / weights.sum()

    lat_cost = _huber(e_lat / 0.60, delta=1.0)
    lon_cost = _huber(e_lon / 2.50, delta=1.0)
    yaw_cost = _huber((2.0 * np.sin(0.5 * e_yaw)) / 0.20, delta=1.0)

    turn_ratio = np.clip(
        abs(float(_wrap_angle(expert_traj_local[-1, 2] - expert_traj_local[0, 2]))) / 0.35,
        0.0,
        1.0,
    )

    w_lat = 2.5 - 0.7 * turn_ratio
    w_lon = 0.8
    w_yaw = 1.5 + 0.8 * turn_ratio
    w_terminal_lat = 4.0 - 1.8 * turn_ratio
    w_terminal_lon = 1.0
    w_terminal_yaw = 3.0 + 1.5 * turn_ratio

    point_cost = (
        w_lat * np.sum(lat_cost * weights[None, :], axis=1)
        + w_lon * np.sum(lon_cost * weights[None, :], axis=1)
        + w_yaw * np.sum(yaw_cost * weights[None, :], axis=1)
    )

    terminal_cost = (
        w_terminal_lat * np.abs(e_lat[:, -1])
        + w_terminal_lon * np.abs(e_lon[:, -1])
        + w_terminal_yaw * np.abs(e_yaw[:, -1])
    )

    coarse_score_norm = prefilter_scores / np.maximum(np.median(prefilter_scores), 1e-6)
    return (point_cost + terminal_cost + regularization_weight * coarse_score_norm).astype(np.float32)


def _build_expert_anchor_trajectory_segment(
    expert_trajectory: np.ndarray,
    num_anchor_steps: int,
    start_step: int,
) -> np.ndarray:
    required_steps = num_anchor_steps + start_step
    if expert_trajectory.shape[0] < required_steps:
        raise ValueError(
            f"expert trajectory too short: need at least {required_steps} points, got {expert_trajectory.shape[0]}"
        )

    end_step = start_step + num_anchor_steps
    return expert_trajectory[start_step:end_step, :][:, [
        TrajectoryState.X,
        TrajectoryState.Y,
        TrajectoryState.HEADING,
        TrajectoryState.VELOCITY_X,
        TrajectoryState.ACCELERATION_X,
        TrajectoryState.YAW_RATE,
    ]].astype(np.float32)


def _build_expert_anchor_trajectory_aligned(
    expert_trajectory: np.ndarray,
    num_anchor_steps: int,
) -> np.ndarray:
    # Planner anchors start from the first future step, so the expert slice should do the same.
    return _build_expert_anchor_trajectory_segment(
        expert_trajectory=expert_trajectory,
        num_anchor_steps=num_anchor_steps,
        start_step=1,
    )


def _build_expert_anchor_trajectory_legacy(
    expert_trajectory: np.ndarray,
    num_anchor_steps: int,
) -> np.ndarray:
    return _build_expert_anchor_trajectory_segment(
        expert_trajectory=expert_trajectory,
        num_anchor_steps=num_anchor_steps,
        start_step=0,
    )


def _build_current_state(
    expert_trajectory: np.ndarray,
    batch_size: int,
    state_dim: int,
) -> np.ndarray:
    current_state = np.zeros((batch_size, 1, state_dim), dtype=np.float32)
    current_state[:, 0, 3] = float(expert_trajectory[0, TrajectoryState.VELOCITY_X])
    current_state[:, 0, 4] = float(expert_trajectory[0, TrajectoryState.ACCELERATION_X])
    current_state[:, 0, 5] = float(expert_trajectory[0, TrajectoryState.YAW_RATE])
    return np.nan_to_num(current_state, copy=False)


def _evaluate_anchor_subset(
    planner_anchor: np.ndarray,
    expert_trajectory: np.ndarray,
    scene_feature: SceneFeature,
    agent_prediction: AgentPrediction,
    simulator: BatchSimulator,
    evaluator: BatchEvaluator,
) -> tuple[np.ndarray, int]:
    expert_traj_local = _build_expert_anchor_trajectory_aligned(
        expert_trajectory=expert_trajectory,
        num_anchor_steps=min(ANCHOR_NUM_POSES, planner_anchor.shape[1]),
    )
    sorted_anchor_indices, geometry_scores = _rank_anchors_by_expert_shape(
        expert_traj_local=expert_traj_local,
        planner_anchor=planner_anchor,
        preselect_k=MAX_EVAL_ANCHORS,
    )

    candidate_indices = sorted_anchor_indices[np.isfinite(geometry_scores[sorted_anchor_indices])]
    candidate_indices = candidate_indices[:MAX_EVAL_ANCHORS]

    if len(candidate_indices) == 0:
        return np.zeros((planner_anchor.shape[0],), dtype=np.float32), 0

    aggregated_scores = np.zeros((planner_anchor.shape[0],), dtype=np.float32)
    best_anchor_index = int(candidate_indices[0])

    for start_idx in range(0, len(candidate_indices), EVAL_BATCH_SIZE):
        batch_indices = candidate_indices[start_idx : start_idx + EVAL_BATCH_SIZE]
        batch_anchors = planner_anchor[batch_indices, : expert_traj_local.shape[0], :6].astype(np.float32)

        current_state = _build_current_state(
            expert_trajectory=expert_trajectory,
            batch_size=batch_anchors.shape[0],
            state_dim=batch_anchors.shape[-1],
        )
        extended_trajectories = np.concatenate((current_state, batch_anchors), axis=1)

        simulated_trajectories = simulator.simulate(extended_trajectories)
        scores_eval = evaluator.batch_evaluate(
            simulated_trajectories,
            scene_feature=scene_feature,
            agent_prediction_gt=agent_prediction,
            expert_trajectory=expert_trajectory,
            discount_factor=1.0,
            prediction_mode="prediction",
            aggregate_only=True,
        )

        trajectory_scores = np.asarray(scores_eval["aggregate_scores"], dtype=np.float32)
        aggregated_scores[batch_indices] = trajectory_scores
        del simulated_trajectories, scores_eval

        best_score = float(np.max(trajectory_scores)) if len(trajectory_scores) > 0 else 0.0
        if best_score > 0.0:
            best_anchor_index = int(batch_indices[int(np.argmax(trajectory_scores))])
            break

    return aggregated_scores, best_anchor_index


def _select_best_anchor_by_expert_dist(
    planner_anchor: np.ndarray,
    expert_trajectory: np.ndarray,
    start_step: int = 1,
) -> int:
    if start_step == 1:
        expert_traj_local = _build_expert_anchor_trajectory_aligned(
            expert_trajectory=expert_trajectory,
            num_anchor_steps=min(ANCHOR_NUM_POSES, planner_anchor.shape[1]),
        )
    elif start_step == 0:
        expert_traj_local = _build_expert_anchor_trajectory_legacy(
            expert_trajectory=expert_trajectory,
            num_anchor_steps=min(ANCHOR_NUM_POSES, planner_anchor.shape[1]),
        )
    else:
        expert_traj_local = _build_expert_anchor_trajectory_segment(
            expert_trajectory=expert_trajectory,
            num_anchor_steps=min(ANCHOR_NUM_POSES, planner_anchor.shape[1]),
            start_step=start_step,
        )
    expert_dist = _score_anchors_by_expert_traj_numpy(
        expert_traj_local=expert_traj_local,
        planner_anchor=planner_anchor,
    )
    return int(np.argmin(expert_dist))


def _select_best_anchor_by_expert_dist_legacy(
    planner_anchor: np.ndarray,
    expert_trajectory: np.ndarray,
) -> int:
    return _select_best_anchor_by_expert_dist(
        planner_anchor=planner_anchor,
        expert_trajectory=expert_trajectory,
        start_step=0,
    )


def _normalize_anchor_indice_strategy(strategy: Optional[str]) -> str:
    normalized = str(strategy or "expert_dist").strip().lower()
    aliases = {
        "expert_dist": "expert_dist",
        "expert": "expert_dist",
        "distance": "expert_dist",
        "expert_dist_aligned": "expert_dist",
        "aligned": "expert_dist",
        "expert_dist_legacy": "expert_dist_legacy",
        "legacy": "expert_dist_legacy",
        "scorer": "scorer",
        "score": "scorer",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported anchor indice strategy: {strategy}")
    return aliases[normalized]


def _is_cached_feature_valid(path: Path, feature_type: Any) -> bool:
    if not path.exists():
        return False

    try:
        load_feature_target_from_pickle(path, feature_type=feature_type)
        return True
    except Exception as exc:
        logger.warning("Detected corrupted cached feature %s: %s", str(path), str(exc))
        return False


def cache_data(args: List[Dict[str, Any]]) -> List[Optional[Any]]:
    def cache_data_internal(args: List[Dict[str, Any]]) -> List[Optional[Any]]:
        import warnings

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        node_id = int(os.environ.get("NODE_RANK", 0))
        thread_id = str(uuid.uuid4())

        scenario_paths = [Path(a["scenario_path"]) for a in args]
        logger.info(
            "Extracted %s cached scenarios for thread_id=%s, node_id=%s.",
            str(len(scenario_paths)),
            thread_id,
            node_id,
        )

        future_sampling = TrajectorySampling(num_poses=ANCHOR_NUM_POSES, time_horizon=ANCHOR_TIME_HORIZON)
        simulator = BatchSimulator(future_sampling)
        evaluator = BatchEvaluator(future_sampling,
                                   use_following_penalty=True,
                                   enable_valid_proposal_mask=True,
                                   )
        planner_anchor = np.load(PLANNER_ANCHOR_FILE, mmap_mode="r")
        zero_anchor_index = _get_zero_anchor_index(planner_anchor)

        num_success = 0

        for scenario_arg in args:
            scenario_cache_dir = Path(scenario_arg["scenario_path"])
            override_anchor_score = bool(scenario_arg.get("override_anchor_score", False))
            override_anchor_indice = bool(scenario_arg.get("override_anchor_indice", False))
            check_valid = bool(scenario_arg.get("check_valid", False))
            anchor_score_name = str(scenario_arg.get("anchor_score_name", ANCHOR_SCORE_FILE))
            anchor_indice_name = str(scenario_arg.get("anchor_indice_name", ANCHOR_INDICE_FILE))
            anchor_indice_strategy = _normalize_anchor_indice_strategy(scenario_arg.get("anchor_indice_strategy", "expert_dist"))

            log_name = scenario_cache_dir.parent.parent.name
            type_name = scenario_cache_dir.parent.name
            token = scenario_cache_dir.name
            score_path = scenario_cache_dir / anchor_score_name
            indice_path = scenario_cache_dir / anchor_indice_name

            score_path_valid = score_path.exists()
            indice_path_valid = indice_path.exists()
            if check_valid:
                if score_path_valid:
                    score_path_valid = _is_cached_feature_valid(score_path, AnchorScores)
                if indice_path_valid:
                    indice_path_valid = _is_cached_feature_valid(indice_path, AnchorIndice)

            need_anchor_score = override_anchor_score or not score_path_valid
            need_anchor_indice = override_anchor_indice or not indice_path_valid

            if not need_anchor_score and not need_anchor_indice:
                logger.debug(
                    "Skip %s/%s/%s because anchor score and indice already exist.",
                    log_name,
                    type_name,
                    token,
                )
                num_success += 1
                continue

            scene_feature_path = scenario_cache_dir / "scene_feature.gz"
            agent_prediction_path = scenario_cache_dir / "agent_prediction.gz"
            expert_trajectory_path = scenario_cache_dir / "expert_trajectory.gz"

            missing_paths: List[str] = []
            if need_anchor_indice and not expert_trajectory_path.exists():
                missing_paths.append("expert_trajectory.gz")
            if need_anchor_score:
                if not scene_feature_path.exists():
                    missing_paths.append("scene_feature.gz")
                if not agent_prediction_path.exists():
                    missing_paths.append("agent_prediction.gz")
                if not expert_trajectory_path.exists():
                    missing_paths.append("expert_trajectory.gz")

            if missing_paths:
                logger.info(
                    "Skip %s/%s/%s due to missing cached files: %s",
                    log_name,
                    type_name,
                    token,
                    ", ".join(sorted(set(missing_paths))),
                )
                continue

            scene_feature = None
            agent_prediction = None
            expert_trajectory_feature = None
            expert_trajectory = None
            aggregated_scores = None
            best_anchor_index_from_scorer = None
            anchor_scores = None
            anchor_indice = None

            try:
                expert_trajectory_feature = load_feature_target_from_pickle(
                    expert_trajectory_path,
                    feature_type=Trajectory,
                )
                expert_trajectory = np.asarray(expert_trajectory_feature.data, dtype=np.float32)

                expert_is_stationary, displacement_m, displacement_index = _is_expert_stationary_at_time(
                    expert_trajectory=expert_trajectory,
                )
                if expert_is_stationary:
                    if need_anchor_indice:
                        anchor_indice = AnchorIndice(indice=np.array([zero_anchor_index], dtype=np.int64))
                        dump_feature_target_to_pickle(indice_path, anchor_indice.serialize())

                    if need_anchor_score:
                        aggregated_scores = np.zeros((planner_anchor.shape[0],), dtype=np.float32)
                        anchor_scores = AnchorScores(aggregated_scores=aggregated_scores)
                        dump_feature_target_to_pickle(score_path, anchor_scores.serialize())

                    logger.debug(
                        "Assign zero anchor for stationary expert %s/%s/%s: disp@t=%.1fs(idx=%d)=%.4fm, zero_anchor_index=%d",
                        log_name,
                        type_name,
                        token,
                        EXPERT_STATIONARY_CHECK_TIME_S,
                        displacement_index,
                        displacement_m,
                        zero_anchor_index,
                    )
                    num_success += 1
                    if num_success % 32 == 0:
                        gc.collect()
                    continue

                need_scorer_eval = need_anchor_score or (need_anchor_indice and anchor_indice_strategy == "scorer")

                if need_scorer_eval:
                    scene_feature = load_feature_target_from_pickle(
                        scene_feature_path,
                        feature_type=SceneFeature,
                    )
                    agent_prediction = load_feature_target_from_pickle(
                        agent_prediction_path,
                        feature_type=AgentPrediction,
                    )

                    aggregated_scores, best_anchor_index_from_scorer = _evaluate_anchor_subset(
                        planner_anchor=planner_anchor,
                        expert_trajectory=expert_trajectory,
                        scene_feature=scene_feature,
                        agent_prediction=agent_prediction,
                        simulator=simulator,
                        evaluator=evaluator,
                    )
                    # best_anchor_index_from_scorer = int(np.argmax(aggregated_scores)) if aggregated_scores is not None else None

                if need_anchor_indice:
                    if anchor_indice_strategy == "expert_dist":
                        best_anchor_index = _select_best_anchor_by_expert_dist(
                            planner_anchor=planner_anchor,
                            expert_trajectory=expert_trajectory,
                            start_step=1,
                        )
                    elif anchor_indice_strategy == "expert_dist_legacy":
                        best_anchor_index = _select_best_anchor_by_expert_dist_legacy(
                            planner_anchor=planner_anchor,
                            expert_trajectory=expert_trajectory,
                        )
                    else:
                        if best_anchor_index_from_scorer is None:
                            raise RuntimeError("scorer-based anchor indice requested but scorer result was not computed")
                        best_anchor_index = best_anchor_index_from_scorer
                    anchor_indice = AnchorIndice(indice=np.array([best_anchor_index], dtype=np.int64))
                    dump_feature_target_to_pickle(indice_path, anchor_indice.serialize())

                if need_anchor_score:
                    if aggregated_scores is None:
                        aggregated_scores, best_anchor_index_from_scorer = _evaluate_anchor_subset(
                            planner_anchor=planner_anchor,
                            expert_trajectory=expert_trajectory,
                            scene_feature=scene_feature,
                            agent_prediction=agent_prediction,
                            simulator=simulator,
                            evaluator=evaluator,
                        )

                    anchor_scores = AnchorScores(aggregated_scores=aggregated_scores)
                    dump_feature_target_to_pickle(score_path, anchor_scores.serialize())

                num_success += 1
                if num_success % 32 == 0:
                    gc.collect()

            except Exception as exc:
                logger.info("Failed to append anchor score for %s/%s/%s: %s", log_name, type_name, token, str(exc))
                continue
            finally:
                del scene_feature, agent_prediction, expert_trajectory_feature, expert_trajectory
                del aggregated_scores, anchor_scores, anchor_indice

        logger.info(
            "Cached %d/%d scenarios for thread_id=%s, node_id=%s",
            num_success,
            len(scenario_paths),
            thread_id,
            node_id,
        )
        return [f"num_success:{num_success}/{len(scenario_paths)}"]

    result = cache_data_internal(args)
    gc.collect()
    return result


def _collect_scenario_paths_for_log(task: tuple[Path, bool]) -> List[Path]:
    log_path, expand_iteration = task
    scenario_paths: List[Path] = []
    if expand_iteration:
        for scene_path in sorted(path for path in log_path.iterdir() if path.is_dir()):
            for iter_path in sorted(path for path in scene_path.iterdir() if path.is_dir()):
                scenario_paths.append(iter_path)
    else:
        for type_path in sorted(path for path in log_path.iterdir() if path.is_dir()):
            for token_path in sorted(path for path in type_path.iterdir() if path.is_dir()):
                scenario_paths.append(token_path)
    return scenario_paths


def _collect_scenario_paths_by_scan(cache_dir: Path, expand_iteration: bool) -> List[Path]:
    log_paths = sorted(path for path in cache_dir.iterdir() if path.is_dir())
    if not log_paths:
        return []

    num_workers = max(1, int((os.cpu_count() or 1) * 0.9))
    num_workers = min(num_workers, len(log_paths))

    if num_workers == 1:
        scenario_paths: List[Path] = []
        for log_path in log_paths:
            scenario_paths.extend(_collect_scenario_paths_for_log((log_path, expand_iteration)))
        return scenario_paths

    scenario_paths: List[Path] = []
    with mp.Pool(processes=num_workers) as pool:
        for log_scenario_paths in pool.imap(
            _collect_scenario_paths_for_log,
            ((log_path, expand_iteration) for log_path in log_paths),
        ):
            scenario_paths.extend(log_scenario_paths)

    return scenario_paths


def _collect_scenario_paths_by_filters(
    cache_dir: Path,
    expand_iteration: bool,
    selected_tokens: Optional[set[str]] = None,
    selected_log_names: Optional[set[str]] = None,
) -> List[Path]:
    scenario_paths: List[Path] = []
    for log_path in sorted(path for path in cache_dir.iterdir() if path.is_dir()):
        if selected_log_names is not None and log_path.name not in selected_log_names:
            continue
        for type_path in sorted(path for path in log_path.iterdir() if path.is_dir()):
            for token_path in sorted(path for path in type_path.iterdir() if path.is_dir()):
                if selected_tokens is not None and token_path.name not in selected_tokens:
                    continue
                if expand_iteration:
                    scenario_paths.extend(sorted(path for path in token_path.iterdir() if path.is_dir()))
                else:
                    scenario_paths.append(token_path)
    return scenario_paths


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for cached dataset post-processing.
    :param cfg: omegaconf dictionary
    """
    sys.stdout.flush()
    logger.info("Global Seed set to 0")
    pl.seed_everything(0, workers=True)

    logger.info("Building Worker")
    cache_dir = Path(cfg.cache.cache_path) / cfg.job_name
    expand_iteration = bool(getattr(cfg, "expand_iteration", False))
    logger.info("Cache directory: %s (expand_iteration=%s)", str(cache_dir), expand_iteration)

    use_manifest=bool(getattr(cfg.cache, "use_manifest", True))

    selected_tokens_path = resolve_scene_tokens_path(
        cache_root=cache_dir,
        scene_tokens_path=getattr(cfg.cache, "scene_tokens_path", None),
        scene_tokens_name=str(getattr(cfg.cache, "scene_tokens_name", DEFAULT_SCENE_TOKENS_JSON_NAME)),
    )

    selected_tokens: Optional[set[str]] = None
    selected_log_names: Optional[set[str]] = None
    if selected_tokens_path.is_file() and use_manifest:
        token_selection = load_scene_token_selection(selected_tokens_path)
        selected_tokens = set(token_selection.get("scenario_tokens", []))
        loaded_log_names = token_selection.get("log_names", [])
        selected_log_names = set(loaded_log_names) if loaded_log_names else None
        scenario_paths = _collect_scenario_paths_by_filters(
            cache_dir=cache_dir,
            expand_iteration=expand_iteration,
            selected_tokens=selected_tokens,
            selected_log_names=selected_log_names,
        )
        logger.info(
            "Loaded %d selected cached scenarios from scenario token json: %s",
            len(scenario_paths),
            selected_tokens_path,
        )
    else:
        scenario_paths = _collect_scenario_paths_by_scan(cache_dir=cache_dir, expand_iteration=expand_iteration)
        logger.info("Scenario token json not found, scanned %d cached scenarios from directory tree.", len(scenario_paths))

    logger.info("Found %d cached scenarios to process.", len(scenario_paths))

    if bool(getattr(cfg, "benchmark_only", False)):
        benchmark_limit = int(getattr(cfg, "benchmark_scenario_limit", BENCHMARK_SCENARIO_LIMIT))
        _run_benchmark(scenario_paths, benchmark_limit=benchmark_limit)
        print("[INFO] Finished benchmark for current code.")
        return

    worker: WorkerPool = (
        instantiate(cfg.worker)
        if is_target_type(cfg.worker, RayDistributed)
        else instantiate(cfg.worker)
    )
    validate_type(worker, WorkerPool)
    logger.info("Building WorkerPool...DONE!")

    override_anchor_score = bool(getattr(cfg, "override_anchor_score", False))
    override_anchor_indice = bool(getattr(cfg, "override_anchor_indice", False))
    check_valid = bool(getattr(cfg, "check_valid", False))
    anchor_indice_strategy = _normalize_anchor_indice_strategy(getattr(cfg, "anchor_indice_strategy", "expert_dist"))
    anchor_score_name = str(getattr(cfg, "anchor_score_name", ANCHOR_SCORE_FILE))
    anchor_indice_name = str(getattr(cfg, "anchor_indice_name", ANCHOR_INDICE_FILE))

    data_points = [
        {
            "scenario_path": scenario_path,
            "override_anchor_score": override_anchor_score,
            "override_anchor_indice": override_anchor_indice,
            "check_valid": check_valid,
            "anchor_indice_strategy": anchor_indice_strategy,
            "anchor_score_name": anchor_score_name,
            "anchor_indice_name": anchor_indice_name,
        }
        for scenario_path in scenario_paths
    ]

    logger.info(
        "Starting appending anchor scores and indices for %s cached scenarios \n"
        "(override score=%s, override indice=%s, check valid=%s, indice strategy=%s, anchor score name=%s, anchor indice name=%s)...",
        str(len(data_points)),
        override_anchor_score,
        override_anchor_indice,
        check_valid,
        anchor_indice_strategy,
        anchor_score_name,
        anchor_indice_name,
    )

    worker_map(worker, cache_data, data_points)
    logger.info(
        "Finished appending anchor scores and indices for %d cached training/validation samples",
        len(data_points),
    )
    print("[INFO] Finished appending anchor scores for cached data.")


if __name__ == "__main__":
    main()
