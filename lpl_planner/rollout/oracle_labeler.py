from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from shapely.geometry import Polygon

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from lpl_planner.planning.planner.utils.int_enum import RoadType
from lpl_planner.planning.scene.evaluate.scene_scorer import BatchEvaluator
from lpl_planner.planning.scene.evaluate.simulator import DEFAULT_SIMULATION_DT, BatchSimulator
from lpl_planner.planning.scene.map.occupancy_map import OccupancyMap
from lpl_planner.planning.scene.scene_feature.features import (
    AgentPrediction,
    AnchorIndice,
    AnchorScores,
    SceneFeature,
)
from lpl_planner.training.dataset.dataset_utils import (
    dump_feature_target_to_pickle,
    load_feature_target_from_pickle,
)


logger = logging.getLogger(__name__)

_PLANNER_ANCHOR_CACHE: Dict[tuple[str, bool], np.ndarray] = {}
_PROFILE_STAGE_NAMES = ("load", "prefilter", "simulate", "evaluate", "write", "total")

_DRIVABLE_ROAD_TYPES = {
    int(RoadType.LANE),
    int(RoadType.CONNECTOR),
    int(RoadType.CARPARK),
    int(RoadType.INTERSECTION),
}


def _resolve_override_flag(override: Optional[bool], overwrite: Optional[bool]) -> bool:
    if override is not None:
        return bool(override)
    if overwrite is not None:
        return bool(overwrite)
    return False


def _new_profile() -> Dict[str, float]:
    return {stage: 0.0 for stage in _PROFILE_STAGE_NAMES}


def _accumulate_profile(total_profile: Dict[str, float], sample_profile: Optional[Dict[str, float]]) -> None:
    if sample_profile is None:
        return
    for stage in _PROFILE_STAGE_NAMES:
        total_profile[stage] += float(sample_profile.get(stage, 0.0))


def _build_profile_summary(
    total_profile: Dict[str, float],
    sample_count: int,
    wall_time_seconds: float,
) -> Dict[str, Any]:
    cumulative_worker_total = float(total_profile.get("total", 0.0))
    stage_seconds = {stage: float(total_profile.get(stage, 0.0)) for stage in _PROFILE_STAGE_NAMES}
    if cumulative_worker_total > 1e-9:
        stage_percent = {
            stage: (100.0 * stage_seconds[stage] / cumulative_worker_total if stage != "total" else 100.0)
            for stage in _PROFILE_STAGE_NAMES
        }
    else:
        stage_percent = {stage: 0.0 for stage in _PROFILE_STAGE_NAMES}
    return {
        "enabled": True,
        "profiled_samples": int(sample_count),
        "time_basis": "cumulative_sample_time_seconds",
        "wall_time_seconds": float(wall_time_seconds),
        "cumulative_worker_time_seconds": cumulative_worker_total,
        "stage_seconds": stage_seconds,
        "stage_percent": stage_percent,
    }


def _log_profile_summary(profile_summary: Dict[str, Any]) -> None:
    stage_seconds = profile_summary["stage_seconds"]
    stage_percent = profile_summary["stage_percent"]
    logger.info(
        "Oracle profiling across %d samples: wall=%.3fs | cumulative_worker_total=%.3fs | load=%.3fs (%.1f%%) | prefilter=%.3fs (%.1f%%) | simulate=%.3fs (%.1f%%) | evaluate=%.3fs (%.1f%%) | write=%.3fs (%.1f%%)",
        int(profile_summary["profiled_samples"]),
        float(profile_summary.get("wall_time_seconds", 0.0)),
        float(profile_summary.get("cumulative_worker_time_seconds", stage_seconds["total"])),
        stage_seconds["total"],
        stage_seconds["load"],
        stage_percent["load"],
        stage_seconds["prefilter"],
        stage_percent["prefilter"],
        stage_seconds["simulate"],
        stage_percent["simulate"],
        stage_seconds["evaluate"],
        stage_percent["evaluate"],
        stage_seconds["write"],
        stage_percent["write"],
    )


def _load_planner_anchor(planner_anchor_path: Path, use_mmap: bool) -> np.ndarray:
    resolved_path = str(Path(planner_anchor_path).resolve())
    cache_key = (resolved_path, bool(use_mmap))
    cached_anchor = _PLANNER_ANCHOR_CACHE.get(cache_key)
    if cached_anchor is not None:
        return cached_anchor

    mmap_mode = "r" if use_mmap else None
    planner_anchor = np.load(resolved_path, mmap_mode=mmap_mode)
    if planner_anchor.dtype != np.float32:
        planner_anchor = np.asarray(planner_anchor, dtype=np.float32)
    _PLANNER_ANCHOR_CACHE[cache_key] = planner_anchor
    return planner_anchor


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _resolve_ref_path(scene_feature: SceneFeature) -> np.ndarray:
    ref_path_feature = getattr(scene_feature, "ref_path_feature", None)
    ref_path_mask = getattr(scene_feature, "ref_path_feature_mask", None)
    if ref_path_feature is None:
        return np.zeros((0, 3), dtype=np.float32)

    ref_path = _to_numpy(ref_path_feature).astype(np.float32, copy=False)
    if ref_path_mask is not None:
        mask = _to_numpy(ref_path_mask).astype(bool).reshape(-1)
        if mask.shape[0] == ref_path.shape[0]:
            ref_path = ref_path[mask]
    return ref_path


def _build_drivable_area_map(scene_feature: SceneFeature) -> Optional[OccupancyMap]:
    road_feature = getattr(scene_feature, "road_feature", None)
    if road_feature is None:
        return None

    road_geometry = _to_numpy(road_feature.road_geometry)
    road_type = _to_numpy(road_feature.road_type).reshape(-1)
    if road_geometry.ndim != 3 or road_geometry.shape[0] == 0 or road_type.size == 0:
        return None

    polygons = []
    tokens = []
    for road_idx, raw_type in enumerate(road_type.tolist()):
        if int(raw_type) not in _DRIVABLE_ROAD_TYPES:
            continue

        coords = np.asarray(road_geometry[road_idx], dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] < 3 or coords.shape[1] < 2:
            continue

        finite_mask = np.isfinite(coords).all(axis=1)
        coords = coords[finite_mask]
        if coords.shape[0] < 3:
            continue

        polygon = Polygon(coords[:, :2])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue
        if polygon.geom_type == "MultiPolygon":
            polygon = max(polygon.geoms, key=lambda geom: geom.area, default=None)
        if polygon is None or polygon.is_empty or polygon.area <= 1e-3:
            continue

        tokens.append(f"drivable_area_{road_idx}")
        polygons.append(polygon)

    if not polygons:
        return None
    return OccupancyMap(tokens, np.asarray(polygons, dtype=object))


def _prefilter_anchor_indices_by_drivable_area(
    planner_anchor: np.ndarray,
    selected_indices: np.ndarray,
    drivable_area_map: Optional[OccupancyMap],
) -> np.ndarray:
    selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    if drivable_area_map is None or selected_indices.size == 0:
        return selected_indices

    anchor_endpoints = np.asarray(
        planner_anchor[selected_indices, -1, :2],
        dtype=np.float64,
    )
    if anchor_endpoints.ndim != 2 or anchor_endpoints.shape[0] == 0:
        return selected_indices

    endpoint_in_drivable = drivable_area_map.points_in_polygons(anchor_endpoints).any(axis=0)
    if np.any(endpoint_in_drivable):
        return selected_indices[endpoint_in_drivable]
    return selected_indices


def _prefilter_anchor_indices_by_ref_path(
    planner_anchor: np.ndarray,
    ref_path: np.ndarray,
    current_speed: float,
    topk: int,
) -> np.ndarray:
    anchor_num = planner_anchor.shape[0]
    topk = max(1, min(int(topk), anchor_num))
    ref_xy = np.asarray(ref_path, dtype=np.float32)
    if ref_xy.ndim != 2 or ref_xy.shape[0] < 2:
        return np.arange(topk, dtype=np.int64)

    def _compute_ref_heading(path_xy: np.ndarray, path_heading: Optional[np.ndarray]) -> np.ndarray:
        if path_heading is not None:
            return path_heading.astype(np.float32, copy=False)
        diffs = np.diff(path_xy, axis=0)
        seg_heading = np.arctan2(diffs[:, 1], diffs[:, 0]).astype(np.float32, copy=False)
        return np.concatenate([seg_heading, seg_heading[-1:]], axis=0)

    def _compute_anchor_heading(anchor_xy: np.ndarray, anchor_heading: Optional[np.ndarray], key_indices: np.ndarray) -> np.ndarray:
        if anchor_heading is not None:
            return anchor_heading[:, key_indices].astype(np.float32, copy=False)

        num_steps = anchor_xy.shape[1]
        prev_indices = np.clip(key_indices - 1, 0, num_steps - 1)
        next_indices = np.clip(key_indices + 1, 0, num_steps - 1)
        heading_list = []
        for key_idx, prev_idx, next_idx in zip(key_indices.tolist(), prev_indices.tolist(), next_indices.tolist()):
            start_xy = anchor_xy[:, prev_idx]
            end_xy = anchor_xy[:, next_idx]
            if prev_idx == next_idx and key_idx > 0:
                start_xy = anchor_xy[:, key_idx - 1]
                end_xy = anchor_xy[:, key_idx]
            heading_list.append(np.arctan2(end_xy[:, 1] - start_xy[:, 1], end_xy[:, 0] - start_xy[:, 0]))
        return np.stack(heading_list, axis=1).astype(np.float32, copy=False)

    def _select_ref_point_by_travel_distance(path_xy: np.ndarray, travel_distance: float) -> np.ndarray:
        if path_xy.shape[0] == 1:
            return path_xy[0]
        segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
        cumulative_lengths = np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)])
        clamped_distance = float(np.clip(travel_distance, 0.0, cumulative_lengths[-1]))
        upper_idx = int(np.searchsorted(cumulative_lengths, clamped_distance, side="right"))
        upper_idx = min(max(upper_idx, 1), path_xy.shape[0] - 1)
        lower_idx = upper_idx - 1
        span = cumulative_lengths[upper_idx] - cumulative_lengths[lower_idx]
        if span <= 1e-6:
            return path_xy[upper_idx]
        ratio = (clamped_distance - cumulative_lengths[lower_idx]) / span
        return (1.0 - ratio) * path_xy[lower_idx] + ratio * path_xy[upper_idx]

    def _compute_path_progress(path_xy: np.ndarray) -> np.ndarray:
        if path_xy.shape[0] <= 1:
            return np.zeros((path_xy.shape[0],), dtype=np.float32)
        segment_lengths = np.linalg.norm(np.diff(path_xy, axis=0), axis=1).astype(np.float32, copy=False)
        return np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)], axis=0)

    key_indices = np.asarray([planner_anchor.shape[1] // 2, planner_anchor.shape[1] - 1], dtype=np.int64)
    anchor_xy = planner_anchor[:, :, :2].astype(np.float32, copy=False)
    anchor_key_xy = anchor_xy[:, key_indices]
    anchor_heading = planner_anchor[:, :, 2] if planner_anchor.shape[2] > 2 else None
    anchor_key_heading = _compute_anchor_heading(anchor_xy, anchor_heading, key_indices)

    ref_path_xy = ref_xy[:, :2].astype(np.float32, copy=False)
    ref_path_heading = ref_xy[:, 2] if ref_xy.shape[1] > 2 else None
    ref_heading = _compute_ref_heading(ref_path_xy, ref_path_heading)
    ref_progress = _compute_path_progress(ref_path_xy)

    deltas = anchor_key_xy[:, :, None, :] - ref_path_xy[None, None, :, :]
    sq_dist = np.sum(deltas * deltas, axis=-1)
    nearest_idx = np.argmin(sq_dist, axis=-1)
    nearest_dist = np.sqrt(np.take_along_axis(sq_dist, nearest_idx[..., None], axis=-1)[..., 0])
    nearest_ref_heading = ref_heading[nearest_idx]
    heading_bias = np.abs(_wrap_angle(anchor_key_heading - nearest_ref_heading))

    travel_distance = max(float(current_speed), 0.0) * 2.0
    speed_target_point = _select_ref_point_by_travel_distance(ref_path_xy, travel_distance)
    midpoint_progress = ref_progress[nearest_idx[:, 0]]
    target_progress = ref_progress[np.argmin(np.sum((ref_path_xy - speed_target_point[None, :]) ** 2.0, axis=-1))]

    path_distance_scale = 2.0
    heading_scale = np.deg2rad(20.0)
    progress_scale = max(4.0, 0.5 * max(travel_distance, 0.0))
    total_score = (
        0.45 * (nearest_dist.mean(axis=1) / max(path_distance_scale, 1e-3))
        + 0.30 * (heading_bias.mean(axis=1) / max(heading_scale, 1e-3))
        + 0.20 * (np.abs(midpoint_progress - target_progress) / max(progress_scale, 1e-3))
    )
    return np.argsort(total_score)[:topk].astype(np.int64, copy=False)


def _prepend_current_state(scene_feature: SceneFeature, trajectories: np.ndarray) -> np.ndarray:
    trajectories = np.asarray(trajectories, dtype=np.float32)
    current_state = np.zeros((trajectories.shape[0], 1, trajectories.shape[2]), dtype=trajectories.dtype)
    ego_current_state = _to_numpy(scene_feature.ego_feature.ego_current_state)
    current_state[:, 0, 3] = float(ego_current_state[3])
    current_state[:, 0, 4] = float(ego_current_state[4])
    current_state[:, 0, 5] = float(ego_current_state[5])
    return np.concatenate([current_state, trajectories], axis=1)


def _build_dense_anchor_scores(
    anchor_num: int,
    sampled_indices: np.ndarray,
    sampled_scores: np.ndarray,
    fill_margin: float,
) -> np.ndarray:
    sampled_scores = np.asarray(sampled_scores, dtype=np.float32).reshape(-1)
    if sampled_scores.size == 0:
        return np.full((anchor_num,), -1.0, dtype=np.float32)
    score_span = float(np.max(sampled_scores) - np.min(sampled_scores))
    fill_value = float(np.min(sampled_scores) - max(fill_margin, score_span + 1.0))
    dense_scores = np.full((anchor_num,), fill_value, dtype=np.float32)
    dense_scores[np.asarray(sampled_indices, dtype=np.int64)] = sampled_scores
    return dense_scores


def _iter_sample_dirs(root: Path) -> Iterable[Path]:
    return sorted(path.parent for path in root.rglob("scene_feature.gz"))


def _load_sample_payload(
    sample_dir: Path,
    planner_anchor: np.ndarray,
    prefilter_topk: int,
    profile: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    load_start = perf_counter() if profile is not None else 0.0
    scene_feature = load_feature_target_from_pickle(sample_dir / "scene_feature.gz", feature_type=SceneFeature)
    agent_prediction = load_feature_target_from_pickle(sample_dir / "agent_prediction.gz", feature_type=AgentPrediction)
    if profile is not None:
        profile["load"] += perf_counter() - load_start

    prefilter_start = perf_counter() if profile is not None else 0.0
    ref_path = _resolve_ref_path(scene_feature)
    ego_current_state = _to_numpy(scene_feature.ego_feature.ego_current_state)
    selected_indices = _prefilter_anchor_indices_by_ref_path(
        planner_anchor=planner_anchor,
        ref_path=ref_path,
        current_speed=float(ego_current_state[3]),
        topk=prefilter_topk,
    )
    ref_path_selected_count = int(selected_indices.size)
    drivable_area_map = _build_drivable_area_map(scene_feature)
    selected_indices = _prefilter_anchor_indices_by_drivable_area(
        planner_anchor=planner_anchor,
        selected_indices=selected_indices,
        drivable_area_map=drivable_area_map,
    )
    if profile is not None:
        profile["prefilter"] += perf_counter() - prefilter_start
    if selected_indices.size == 0:
        return None

    return {
        "sample_dir": sample_dir,
        "scene_feature": scene_feature,
        "agent_prediction": agent_prediction,
        "ref_path": ref_path,
        "selected_indices": selected_indices,
        "selected_anchors": planner_anchor[selected_indices],
        "ref_path_selected_count": ref_path_selected_count,
    }


def _write_oracle_label(
    payload: Dict[str, Any],
    aggregate_scores: np.ndarray,
    planner_anchor: np.ndarray,
    fill_margin: float,
    anchor_indice_name: str,
    anchor_score_name: str,
) -> Dict[str, Any]:
    sample_dir = payload["sample_dir"]
    selected_indices = payload["selected_indices"]
    aggregate_scores = np.asarray(aggregate_scores, dtype=np.float32)
    best_local_idx = int(np.argmax(aggregate_scores))
    best_anchor_idx = int(selected_indices[best_local_idx])
    dense_scores = _build_dense_anchor_scores(
        anchor_num=int(planner_anchor.shape[0]),
        sampled_indices=selected_indices,
        sampled_scores=aggregate_scores,
        fill_margin=fill_margin,
    )

    dump_feature_target_to_pickle(
        sample_dir / anchor_indice_name,
        AnchorIndice(indice=np.asarray([best_anchor_idx], dtype=np.int32)).serialize(),
    )
    dump_feature_target_to_pickle(
        sample_dir / anchor_score_name,
        AnchorScores(aggregated_scores=dense_scores).serialize(),
    )
    return {
        "sample_dir": str(sample_dir),
        "oracle_anchor_index": best_anchor_idx,
        "oracle_score": float(aggregate_scores[best_local_idx]),
        "selected_anchor_count": int(selected_indices.shape[0]),
        "ref_path_selected_count": int(payload.get("ref_path_selected_count", selected_indices.shape[0])),
    }


def _evaluate_payload(
    payload: Dict[str, Any],
    simulator: BatchSimulator,
    evaluator: BatchEvaluator,
    discount_factor: float,
    profile: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    simulate_start = perf_counter() if profile is not None else 0.0
    simulated_trajs = simulator.simulate(_prepend_current_state(payload["scene_feature"], payload["selected_anchors"]))
    if profile is not None:
        profile["simulate"] += perf_counter() - simulate_start

    evaluate_start = perf_counter() if profile is not None else 0.0
    scores = evaluator.batch_evaluate(
        simulated_trajs,
        scene_feature=payload["scene_feature"],
        agent_prediction_gt=payload["agent_prediction"],
        ref_path=payload["ref_path"],
        discount_factor=discount_factor,
        aggregate_only=True,
    )
    if profile is not None:
        profile["evaluate"] += perf_counter() - evaluate_start
    return np.asarray(scores["aggregate_scores"], dtype=np.float32)


def _process_sample_dir(
    sample_dir: Path,
    planner_anchor: np.ndarray,
    proposal_sampling: TrajectorySampling,
    prefilter_topk: int,
    discount_factor: float,
    fill_margin: float,
    anchor_indice_name: str,
    anchor_score_name: str,
    simulator: BatchSimulator,
    evaluator: BatchEvaluator,
    debug_profile: bool = False,
    include_record: bool = True,
    override: bool = False,
) -> Optional[Dict[str, Any]]:
    anchor_indice_path = sample_dir / anchor_indice_name
    anchor_score_path = sample_dir / anchor_score_name
    if not override and anchor_indice_path.exists() and anchor_score_path.exists():
        return None

    sample_profile = _new_profile() if debug_profile else None
    total_start = perf_counter() if sample_profile is not None else 0.0
    payload = _load_sample_payload(
        sample_dir=sample_dir,
        planner_anchor=planner_anchor,
        prefilter_topk=prefilter_topk,
        profile=sample_profile,
    )
    if payload is None:
        return None

    aggregate_scores = _evaluate_payload(
        payload=payload,
        simulator=simulator,
        evaluator=evaluator,
        discount_factor=discount_factor,
        profile=sample_profile,
    )
    write_start = perf_counter() if sample_profile is not None else 0.0
    record = _write_oracle_label(
        payload=payload,
        aggregate_scores=aggregate_scores,
        planner_anchor=planner_anchor,
        fill_margin=fill_margin,
        anchor_indice_name=anchor_indice_name,
        anchor_score_name=anchor_score_name,
    )
    result: Dict[str, Any] = {"written": True}
    if sample_profile is not None:
        sample_profile["write"] += perf_counter() - write_start
        sample_profile["total"] += perf_counter() - total_start
        result["profiling"] = sample_profile
    if include_record:
        result["record"] = record
    return result


def _process_oracle_sample_dirs_chunk(args: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not args:
        return []

    planner_anchor = _load_planner_anchor(
        planner_anchor_path=Path(args[0]["planner_anchor_path"]),
        use_mmap=bool(args[0].get("planner_anchor_mmap", False)),
    )
    proposal_sampling = TrajectorySampling(
        num_poses=int(planner_anchor.shape[1]),
        interval_length=float(args[0]["proposal_interval_length"]),
    )
    discount_factor = float(args[0]["discount_factor"])
    simulator = BatchSimulator(
        proposal_sampling=proposal_sampling,
        default_dt=DEFAULT_SIMULATION_DT,
    )
    evaluator = BatchEvaluator(
        proposal_sampling=proposal_sampling,
        default_dt=DEFAULT_SIMULATION_DT,
    )

    results: List[Dict[str, Any]] = []
    for item in args:
        record = _process_sample_dir(
            sample_dir=Path(item["sample_dir"]),
            planner_anchor=planner_anchor,
            proposal_sampling=proposal_sampling,
            prefilter_topk=int(item["prefilter_topk"]),
            discount_factor=discount_factor,
            fill_margin=float(item["fill_margin"]),
            anchor_indice_name=item["anchor_indice_name"],
            anchor_score_name=item["anchor_score_name"],
            simulator=simulator,
            evaluator=evaluator,
            debug_profile=bool(item.get("debug_profile", False)),
            include_record=bool(item.get("include_record", True)),
            override=bool(item.get("override", False)),
        )
        if record is None:
            continue
        record["index"] = int(item["index"])
        results.append(record)
    return results


def _process_sample_dirs_worker_map(
    worker: WorkerPool,
    sample_dirs: List[Path],
    planner_anchor_path: Path,
    proposal_interval_length: float,
    prefilter_topk: int,
    discount_factor: float,
    fill_margin: float,
    anchor_indice_name: str,
    anchor_score_name: str,
    debug_profile: bool,
    include_record: bool,
    override: bool,
    planner_anchor_mmap: bool,
) -> List[Dict[str, Any]]:
    indexed_sample_dirs = [
        {
            "index": index,
            "sample_dir": str(sample_dir),
            "planner_anchor_path": str(planner_anchor_path),
            "planner_anchor_mmap": bool(planner_anchor_mmap),
            "proposal_interval_length": float(proposal_interval_length),
            "prefilter_topk": int(prefilter_topk),
            "discount_factor": discount_factor,
            "fill_margin": float(fill_margin),
            "anchor_indice_name": anchor_indice_name,
            "anchor_score_name": anchor_score_name,
            "debug_profile": bool(debug_profile),
            "include_record": bool(include_record),
            "override": bool(override),
        }
        for index, sample_dir in enumerate(sample_dirs)
    ]
    mapped_results = worker_map(worker, _process_oracle_sample_dirs_chunk, indexed_sample_dirs)
    return sorted(mapped_results, key=lambda item: int(item["index"]))


def generate_oracle_labels_for_sample_dir(
    sample_dir: Path,
    planner_anchor: np.ndarray,
    proposal_sampling: TrajectorySampling,
    prefilter_topk: int,
    discount_factor: float,
    fill_margin: float,
    anchor_indice_name: str,
    anchor_score_name: str,
    overwrite: Optional[bool] = None,
    override: Optional[bool] = None,
    debug_profile: bool = False,
    include_record: bool = True,
) -> Optional[Dict[str, Any]]:
    override = _resolve_override_flag(override=override, overwrite=overwrite)

    simulator = BatchSimulator(
        proposal_sampling=proposal_sampling,
        default_dt=DEFAULT_SIMULATION_DT,
    )
    evaluator = BatchEvaluator(
        proposal_sampling=proposal_sampling,
        default_dt=DEFAULT_SIMULATION_DT,
    )
    return _process_sample_dir(
        sample_dir=sample_dir,
        planner_anchor=planner_anchor,
        proposal_sampling=proposal_sampling,
        prefilter_topk=prefilter_topk,
        discount_factor=discount_factor,
        fill_margin=fill_margin,
        anchor_indice_name=anchor_indice_name,
        anchor_score_name=anchor_score_name,
        simulator=simulator,
        evaluator=evaluator,
        debug_profile=debug_profile,
        include_record=include_record,
        override=override,
    )


def generate_oracle_labels_for_rollout_root(
    root: Path,
    planner_anchor_path: Path,
    worker: Optional[WorkerPool] = None,
    eval_batch_size: int = 16,
    num_workers: int = 0,
    prefilter_topk: int = 1024,
    discount_factor: float = 1.0,
    fill_margin: float = 1.0,
    anchor_indice_name: str = "anchor_indice_oracle.gz",
    anchor_score_name: str = "anchor_scores_oracle.gz",
    summary_name: str = "oracle_generation_summary.json",
    overwrite: Optional[bool] = None,
    override: Optional[bool] = None,
    debug: bool = False,
    include_records: bool = False,
    planner_anchor_mmap: bool = True,
) -> Dict[str, Any]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"rollout root not found: {root}")
    wall_time_start = perf_counter()
    override = _resolve_override_flag(override=override, overwrite=overwrite)
    planner_anchor = _load_planner_anchor(planner_anchor_path=planner_anchor_path, use_mmap=planner_anchor_mmap)
    proposal_sampling = TrajectorySampling(
        num_poses=int(planner_anchor.shape[1]),
        interval_length=0.2,
    )

    created_worker = False
    if worker is None and int(num_workers) > 1:
        worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=int(num_workers))
        created_worker = True

    records: List[Dict[str, Any]] = []
    total_profile = _new_profile()
    profiled_samples = 0
    total = 0
    written = 0
    pending_sample_dirs: List[Path] = []
    for sample_dir in _iter_sample_dirs(root):
        total += 1
        anchor_indice_path = sample_dir / anchor_indice_name
        anchor_score_path = sample_dir / anchor_score_name
        if not override and anchor_indice_path.exists() and anchor_score_path.exists():
            continue
        pending_sample_dirs.append(sample_dir)

    if worker is not None and len(pending_sample_dirs) > 1:
        worker_results = _process_sample_dirs_worker_map(
            worker=worker,
            sample_dirs=pending_sample_dirs,
            planner_anchor_path=planner_anchor_path,
            proposal_interval_length=proposal_sampling.interval_length,
            prefilter_topk=prefilter_topk,
            discount_factor=discount_factor,
            fill_margin=fill_margin,
            anchor_indice_name=anchor_indice_name,
            anchor_score_name=anchor_score_name,
            debug_profile=debug,
            include_record=include_records,
            override=override,
            planner_anchor_mmap=planner_anchor_mmap,
        )
        for result in worker_results:
            written += int(bool(result.get("written", False)))
            if include_records and "record" in result:
                records.append(result["record"])
            if debug and "profiling" in result:
                _accumulate_profile(total_profile, result["profiling"])
                profiled_samples += 1
    else:
        simulator = BatchSimulator(
            proposal_sampling=proposal_sampling,
            default_dt=DEFAULT_SIMULATION_DT,
        )
        evaluator = BatchEvaluator(
            proposal_sampling=proposal_sampling,
            default_dt=DEFAULT_SIMULATION_DT,
        )
        for sample_dir in pending_sample_dirs:
            record = _process_sample_dir(
                sample_dir=sample_dir,
                planner_anchor=planner_anchor,
                proposal_sampling=proposal_sampling,
                prefilter_topk=prefilter_topk,
                discount_factor=discount_factor,
                fill_margin=fill_margin,
                anchor_indice_name=anchor_indice_name,
                anchor_score_name=anchor_score_name,
                simulator=simulator,
                evaluator=evaluator,
                debug_profile=debug,
                include_record=include_records,
                override=override,
            )
            if record is None:
                continue
            written += int(bool(record.get("written", False)))
            if include_records and "record" in record:
                records.append(record["record"])
            if debug and "profiling" in record:
                _accumulate_profile(total_profile, record["profiling"])
                profiled_samples += 1

    if created_worker and hasattr(worker, "shutdown"):
        try:
            worker.shutdown()
        except Exception:
            pass

    wall_time_seconds = perf_counter() - wall_time_start
    profile_summary = (
        _build_profile_summary(total_profile, profiled_samples, wall_time_seconds)
        if debug
        else {"enabled": False, "wall_time_seconds": float(wall_time_seconds)}
    )

    summary = {
        "root": str(root),
        "planner_anchor_path": str(planner_anchor_path),
        "total_samples": total,
        "written_samples": written,
        "pending_samples": len(pending_sample_dirs),
        "skipped_existing_samples": max(total - len(pending_sample_dirs), 0),
        "parallel_eval_enabled": worker is not None,
        "eval_batch_size": int(eval_batch_size),
        "prefilter_topk": int(prefilter_topk),
        "discount_factor": float(discount_factor),
        "anchor_indice_name": anchor_indice_name,
        "anchor_score_name": anchor_score_name,
        "override": bool(override),
        "records_included": bool(include_records),
        "planner_anchor_mmap": bool(planner_anchor_mmap),
        "debug": bool(debug),
        "profiling": profile_summary,
        "records": records if include_records else None,
    }
    summary_path = root / summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if debug:
        _log_profile_summary(profile_summary)
    logger.info("Generated oracle labels for %d/%d rollout samples under %s", written, total, root)
    return summary