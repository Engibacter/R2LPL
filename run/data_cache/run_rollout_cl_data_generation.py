from __future__ import annotations

import importlib.util
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from nuplan.common.utils.distributed_scenario_filter import DistributedMode, DistributedScenarioFilter
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.abstract_scenario_builder import RepartitionStrategy
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.script.builders.utils.utils_type import is_target_type, validate_type
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_ray import RayDistributed
from nuplan.planning.utils.multithreading.worker_utils import chunk_list

from lpl_planner.rollout.rollout_cl_generator import RolloutCLScenarioWorker
from lpl_planner.rollout.rollout_utils import extract_planner_state_dict
from lpl_planner.utils.default_paths import configure_default_paths


configure_default_paths()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

lpl_planner_spec = importlib.util.find_spec("lpl_planner")
lpl_planner_dir = os.path.dirname(lpl_planner_spec.origin)

CONFIG_PATH = os.path.join(lpl_planner_dir, "config/training")
CONFIG_NAME = "custom_rollout_caching"


def _resolve_checkpoint_path(cfg: DictConfig) -> Optional[Path]:
    candidates = [
        getattr(cfg, "rollout_ckpt_path", None),
        getattr(cfg, "ckpt_path", None),
        getattr(cfg, "start_model_path", None),
        getattr(cfg, "pretrained_ckpt", None),
    ]
    for candidate in candidates:
        if candidate in {None, "", "None"}:
            continue
        path = Path(str(candidate)).expanduser()
        if path.is_file():
            return path
    return None


def _load_rollout_state_dict(cfg: DictConfig) -> Optional[Dict[str, Any]]:
    ckpt_path = _resolve_checkpoint_path(cfg)
    if ckpt_path is None:
        logger.warning("No rollout checkpoint path resolved; workers will use freshly instantiated model weights.")
        return None
    logger.info("Loading rollout checkpoint from %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    return extract_planner_state_dict(checkpoint)


def _build_scenarios(cfg: DictConfig, worker: WorkerPool) -> List[AbstractScenario]:
    scenario_builder = instantiate(cfg.scenario_builder)
    if int(os.environ.get("NUM_NODES", 1)) > 1 and bool(getattr(cfg, "distribute_by_scenario", True)):
        repartition_strategy = scenario_builder.repartition_strategy
        if repartition_strategy == RepartitionStrategy.REPARTITION_FILE_DISK:
            scenario_filter = DistributedScenarioFilter(
                cfg=cfg,
                worker=worker,
                node_rank=int(os.environ.get("NODE_RANK", 0)),
                num_nodes=int(os.environ.get("NUM_NODES", 1)),
                synchronization_path=Path(str(getattr(cfg, "rollout_package_dir", getattr(cfg, "rollout_cache_dir", "rollout_packages")))),
                timeout_seconds=int(getattr(cfg, "distributed_timeout_seconds", 3600)),
                distributed_mode=getattr(cfg, "distributed_mode", DistributedMode.LOG_FILE_BASED),
            )
            return scenario_filter.get_scenarios()
        if repartition_strategy == RepartitionStrategy.INLINE:
            scenario_filter = build_scenario_filter(cfg.scenario_filter)
            scenarios = scenario_builder.get_scenarios(scenario_filter, worker)
            return chunk_list(scenarios, int(os.environ.get("NUM_NODES", 1)))[int(os.environ.get("NODE_RANK", 0))]
        raise ValueError(f"Unsupported repartition strategy: {repartition_strategy}")

    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    return scenario_builder.get_scenarios(scenario_filter, worker)


def _load_planner_anchor(cfg: DictConfig) -> Optional[Any]:
    anchor_path = getattr(cfg, "oracle_planner_anchor_path", None) or getattr(cfg, "planner_anchor_path", None)
    if anchor_path in {None, "", "None"}:
        return None
    path = Path(str(anchor_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Planner anchor path does not exist: {path}")
    import numpy as np

    return np.load(path, mmap_mode="r")


def _seed_everything(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_rollout_sequential(cfg: DictConfig, scenarios: List[AbstractScenario], state_dict: Optional[Dict[str, Any]], planner_anchor) -> List[Dict[str, Any]]:
    if not scenarios:
        return []
    worker = RolloutCLScenarioWorker(cfg=cfg, state_dict=state_dict, planner_anchor=planner_anchor, enable_actor=True)
    results = []
    for scenario in scenarios:
        try:
            results.append(worker.collect_rollout_package_task(scenario))
        except Exception as exc:
            logger.exception("Failed rollout package generation for scenario %s", getattr(scenario, "token", "<unknown>"))
            results.append({"scenario_token": getattr(scenario, "token", "<unknown>"), "error": str(exc)})
    return results


def _run_rollout_ray(cfg: DictConfig, scenarios: List[AbstractScenario], state_dict: Optional[Dict[str, Any]], planner_anchor) -> List[Dict[str, Any]]:
    if not scenarios:
        return []
    import ray

    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(ignore_reinit_error=True)

    worker_count = min(max(int(getattr(cfg, "num_workers", 1)), 1), len(scenarios))
    WorkerRemote = ray.remote(
        num_gpus=float(getattr(cfg, "gpus_per_worker", 0.0)),
        num_cpus=float(getattr(cfg, "cpus_per_worker", 1.0)),
    )(RolloutCLScenarioWorker)
    workers = [WorkerRemote.remote(cfg, state_dict, planner_anchor, True) for _ in range(worker_count)]

    results: List[Dict[str, Any]] = []
    pending: Dict[Any, Any] = {}
    scenario_index = 0
    for worker in workers:
        if scenario_index >= len(scenarios):
            break
        pending[worker.collect_rollout_package_task.remote(scenarios[scenario_index])] = worker
        scenario_index += 1

    while pending:
        done_refs, _ = ray.wait(list(pending.keys()), num_returns=1)
        done_ref = done_refs[0]
        worker = pending.pop(done_ref)
        try:
            results.append(ray.get(done_ref))
        except Exception as exc:
            logger.exception("A rollout CL Ray task failed")
            results.append({"error": str(exc)})
        if scenario_index < len(scenarios):
            pending[worker.collect_rollout_package_task.remote(scenarios[scenario_index])] = worker
            scenario_index += 1

    for worker in workers:
        try:
            ray.kill(worker)
        except Exception:
            pass
    if owns_ray:
        ray.shutdown()
    return results


def _build_frame_tasks(package_results: List[Dict[str, Any]], scenarios: List[AbstractScenario]) -> List[Dict[str, Any]]:
    scenario_by_token = {scenario.token: scenario for scenario in scenarios}
    tasks: List[Dict[str, Any]] = []
    for package_result in package_results:
        if "error" in package_result:
            continue
        scenario = scenario_by_token.get(str(package_result.get("scenario_token", "")))
        package_path = str(package_result.get("package_path", ""))
        if scenario is None or not package_path:
            continue
        for frame_record in package_result.get("frame_records", []) or []:
            if frame_record.get("oracle_candidate_reasons"):
                tasks.append({"scenario": scenario, "package_path": package_path, "frame_record": frame_record})
    return tasks


def _build_road_frame_tasks(package_results: List[Dict[str, Any]], scenarios: List[AbstractScenario], cfg: DictConfig) -> List[Dict[str, Any]]:
    scenario_by_token = {scenario.token: scenario for scenario in scenarios}
    stride = max(int(getattr(cfg, "road_frame_stride", 5)), 1)
    max_frames = max(int(getattr(cfg, "road_max_frames_per_scenario", 30)), 1)
    tasks: List[Dict[str, Any]] = []
    for package_result in package_results:
        if "error" in package_result:
            continue
        scenario = scenario_by_token.get(str(package_result.get("scenario_token", "")))
        package_path = str(package_result.get("package_path", ""))
        if scenario is None or not package_path:
            continue
        selected = []
        for frame_record in package_result.get("frame_records", []) or []:
            if int(frame_record.get("iteration", 0)) % stride == 0:
                selected.append(frame_record)
            if len(selected) >= max_frames:
                break
        tasks.extend({"scenario": scenario, "package_path": package_path, "frame_record": frame_record} for frame_record in selected)
    return tasks


def _run_frames_sequential(cfg: DictConfig, frame_tasks: List[Dict[str, Any]], planner_anchor) -> List[Dict[str, Any]]:
    worker = RolloutCLScenarioWorker(cfg=cfg, state_dict=None, planner_anchor=planner_anchor, enable_actor=False)
    use_road = str(getattr(cfg, "rollout_retrieval_style", "r2lpl")).strip().lower() == "road"
    results: List[Dict[str, Any]] = []
    for task in frame_tasks:
        try:
            if use_road:
                results.append(worker.process_road_frame_from_package_path(task["scenario"], task["package_path"], task["frame_record"]))
            else:
                results.append(worker.process_frame_from_package_path(task["scenario"], task["package_path"], task["frame_record"]))
        except Exception as exc:
            logger.exception("Failed frame target recovery")
            results.append({"error": str(exc)})
    return results


def _run_frames_ray(cfg: DictConfig, frame_tasks: List[Dict[str, Any]], planner_anchor) -> List[Dict[str, Any]]:
    if not frame_tasks:
        return []
    import ray

    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(ignore_reinit_error=True)

    worker_count = min(max(int(getattr(cfg, "oracle_num_workers", 0)) or int(getattr(cfg, "num_workers", 1)), 1), len(frame_tasks))
    WorkerRemote = ray.remote(
        num_gpus=float(getattr(cfg, "oracle_gpus_per_worker", 0.0)),
        num_cpus=float(getattr(cfg, "oracle_cpus_per_worker", getattr(cfg, "cpus_per_worker", 1.0))),
    )(RolloutCLScenarioWorker)
    workers = [WorkerRemote.remote(cfg, None, planner_anchor, False) for _ in range(worker_count)]
    scenario_refs: Dict[str, Any] = {}
    for task in frame_tasks:
        token = str(task["scenario"].token)
        if token not in scenario_refs:
            scenario_refs[token] = ray.put(task["scenario"])

    use_road = str(getattr(cfg, "rollout_retrieval_style", "r2lpl")).strip().lower() == "road"
    results: List[Dict[str, Any]] = []
    pending: Dict[Any, Any] = {}
    task_index = 0
    for worker in workers:
        if task_index >= len(frame_tasks):
            break
        task = frame_tasks[task_index]
        scenario_ref = scenario_refs[str(task["scenario"].token)]
        if use_road:
            pending[worker.process_road_frame_from_package_path.remote(scenario_ref, task["package_path"], task["frame_record"])] = worker
        else:
            pending[worker.process_frame_from_package_path.remote(scenario_ref, task["package_path"], task["frame_record"])] = worker
        task_index += 1

    while pending:
        done_refs, _ = ray.wait(list(pending.keys()), num_returns=1)
        done_ref = done_refs[0]
        worker = pending.pop(done_ref)
        try:
            results.append(ray.get(done_ref))
        except Exception as exc:
            logger.exception("A frame target Ray task failed")
            results.append({"error": str(exc)})
        if task_index < len(frame_tasks):
            task = frame_tasks[task_index]
            scenario_ref = scenario_refs[str(task["scenario"].token)]
            if use_road:
                pending[worker.process_road_frame_from_package_path.remote(scenario_ref, task["package_path"], task["frame_record"])] = worker
            else:
                pending[worker.process_frame_from_package_path.remote(scenario_ref, task["package_path"], task["frame_record"])] = worker
            task_index += 1

    for worker in workers:
        try:
            ray.kill(worker)
        except Exception:
            pass
    if owns_ray:
        ray.shutdown()
    return results


def _write_summary(cfg: DictConfig, package_results: List[Dict[str, Any]], frame_results: List[Dict[str, Any]]) -> Path:
    output_root = Path(str(getattr(cfg, "rollout_cl_cache_dir", getattr(cfg.cache, "cache_path", "rollout_cl_cache"))))
    output_root.mkdir(parents=True, exist_ok=True)

    def _json_safe(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            value = float(value)
            return value if np.isfinite(value) else None
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return value

    def _increment(counter: Dict[str, int], key: Any, amount: int = 1) -> None:
        key = str(key or "unknown")
        counter[key] = counter.get(key, 0) + int(amount)

    def _summarize_package_result(result: Dict[str, Any]) -> Dict[str, Any]:
        summary_result = {key: value for key, value in result.items() if key not in {"package", "frame_records"}}
        package = result.get("package")
        if isinstance(package, dict):
            ego_vectors = package.get("ego_state_vectors")
            ego_vector_count = int(np.asarray(ego_vectors).shape[0]) if ego_vectors is not None else 0
            frame_records = package.get("frame_records", []) or []
        else:
            ego_vector_count = 0
            frame_records = result.get("frame_records", []) or []
        if frame_records:
            reason_counts: Dict[str, int] = {}
            for record in frame_records:
                for reason in record.get("oracle_candidate_reasons", []) or []:
                    reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
            summary_result["package_frame_count"] = len(frame_records)
            summary_result["package_action_count"] = len(package.get("action_records", []) or []) if isinstance(package, dict) else int(result.get("frames", 0))
            summary_result["package_ego_state_count"] = ego_vector_count
            summary_result["oracle_candidate_count"] = sum(1 for record in frame_records if record.get("oracle_candidate_reasons"))
            summary_result["high_risk_frame_count"] = sum(1 for record in frame_records if record.get("is_high_risk"))
            summary_result["model_expert_disagreement_count"] = sum(1 for record in frame_records if record.get("is_model_expert_disagreement"))
            summary_result["oracle_candidate_reason_counts"] = reason_counts
        return _json_safe(summary_result)

    package_scene_type_counts: Dict[str, int] = {}
    package_failure_counts: Dict[str, int] = {}
    package_candidate_reason_counts: Dict[str, int] = {}
    package_high_risk_count = 0
    package_disagreement_count = 0
    package_oracle_candidate_count = 0
    for result in package_results:
        _increment(package_scene_type_counts, result.get("scene_type"))
        _increment(package_failure_counts, result.get("failure_type", result.get("termination_type")))
        frame_records = result.get("frame_records", []) or []
        package_high_risk_count += sum(1 for record in frame_records if record.get("is_high_risk"))
        package_disagreement_count += sum(1 for record in frame_records if record.get("is_model_expert_disagreement"))
        for record in frame_records:
            reasons = record.get("oracle_candidate_reasons", []) or []
            if reasons:
                package_oracle_candidate_count += 1
            for reason in reasons:
                _increment(package_candidate_reason_counts, reason)

    frame_scene_type_counts: Dict[str, int] = {}
    frame_state_class_counts: Dict[str, int] = {}
    frame_drop_reason_counts: Dict[str, int] = {}
    frame_candidate_reason_counts: Dict[str, int] = {}
    frame_disagreement_reason_counts: Dict[str, int] = {}
    kept_frame_count = 0
    unrecoverable_count = 0
    best_score_sum = 0.0
    best_score_count = 0
    min_ttc_values = []
    for result in frame_results:
        kept = int(result.get("kept", 0))
        kept_frame_count += kept
        _increment(frame_scene_type_counts, result.get("scene_type"))
        if kept:
            _increment(frame_state_class_counts, result.get("state_class"))
            score = result.get("best_score", None)
            if score is not None:
                score = float(score)
                if np.isfinite(score):
                    best_score_sum += score
                    best_score_count += 1
            min_ttc = result.get("min_ttc", None)
            if min_ttc is not None:
                min_ttc = float(min_ttc)
                if np.isfinite(min_ttc):
                    min_ttc_values.append(min_ttc)
        else:
            drop_reason = result.get("drop_reason", "unknown")
            _increment(frame_drop_reason_counts, drop_reason)
            if drop_reason == "unrecoverable":
                unrecoverable_count += 1
        for reason in result.get("oracle_candidate_reasons", []) or []:
            _increment(frame_candidate_reason_counts, reason)
        disagreement_reason = result.get("disagreement_reason", "")
        if disagreement_reason:
            _increment(frame_disagreement_reason_counts, disagreement_reason)

    summary = {
        "job_name": str(getattr(cfg, "job_name", "rollout_cl_generation")),
        "rollout_package_dir": str(getattr(cfg, "rollout_package_dir", getattr(cfg, "rollout_cache_dir", ""))),
        "rollout_cl_cache_dir": str(output_root),
        "num_scenarios": len(package_results),
        "num_frame_tasks": len(frame_results),
        "total_kept": int(sum(int(result.get("kept", 0)) for result in frame_results)),
        "total_dropped_unrecoverable": int(sum(1 for result in frame_results if result.get("drop_reason") == "unrecoverable")),
        "num_errors": int(sum(1 for result in package_results if "error" in result) + sum(1 for result in frame_results if "error" in result)),
        "package_results": [_summarize_package_result(result) for result in package_results],
        "frame_results": _json_safe(frame_results),
    }
    summary_path = output_root / str(getattr(cfg, "rollout_cl_summary_name", "rollout_cl_generation_summary.json"))
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    compact_summary = {
        "job_name": summary["job_name"],
        "rollout_package_dir": summary["rollout_package_dir"],
        "rollout_cl_cache_dir": summary["rollout_cl_cache_dir"],
        "num_scenarios": summary["num_scenarios"],
        "num_frame_tasks": summary["num_frame_tasks"],
        "total_kept": kept_frame_count,
        "total_dropped_unrecoverable": unrecoverable_count,
        "num_errors": summary["num_errors"],
        "package_scene_type_counts": package_scene_type_counts,
        "package_failure_counts": package_failure_counts,
        "package_oracle_candidate_count": package_oracle_candidate_count,
        "package_high_risk_frame_count": package_high_risk_count,
        "package_model_expert_disagreement_count": package_disagreement_count,
        "package_candidate_reason_counts": package_candidate_reason_counts,
        "frame_scene_type_counts": frame_scene_type_counts,
        "frame_state_class_counts": frame_state_class_counts,
        "frame_drop_reason_counts": frame_drop_reason_counts,
        "frame_candidate_reason_counts": frame_candidate_reason_counts,
        "frame_disagreement_reason_counts": frame_disagreement_reason_counts,
        "best_score_mean": best_score_sum / best_score_count if best_score_count else None,
        "min_ttc_mean": float(np.mean(min_ttc_values)) if min_ttc_values else None,
        "min_ttc_min": float(np.min(min_ttc_values)) if min_ttc_values else None,
    }
    compact_path = output_root / "rollout_cl_generation_summary_compact.json"
    compact_path.write_text(json.dumps(_json_safe(compact_summary), indent=2), encoding="utf-8")
    return summary_path


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    sys.stdout.flush()
    _seed_everything(int(getattr(cfg, "seed", 0)))
    logger.info("Rollout CL generation config:\n%s", OmegaConf.to_yaml(cfg))

    scenario_worker: WorkerPool = instantiate(cfg.worker) if is_target_type(cfg.worker, RayDistributed) else instantiate(cfg.worker)
    validate_type(scenario_worker, WorkerPool)
    try:
        scenarios = _build_scenarios(cfg, scenario_worker)
    finally:
        if hasattr(scenario_worker, "shutdown"):
            try:
                scenario_worker.shutdown()
                logger.info("Scenario worker shutdown after scenario extraction.")
            except Exception:
                logger.exception("Failed to shutdown scenario worker after scenario extraction.")
    max_scenarios = getattr(cfg, "max_rollout_scenarios", None)
    if max_scenarios not in {None, "", "None"}:
        scenarios = scenarios[: int(max_scenarios)]
    logger.info("Collected %d scenarios for rollout CL data generation", len(scenarios))

    state_dict = _load_rollout_state_dict(cfg)
    use_parallel = bool(getattr(cfg, "rollout_parallel", True)) and int(getattr(cfg, "num_workers", 1)) > 1
    if use_parallel:
        package_results = _run_rollout_ray(cfg, scenarios, state_dict, planner_anchor=None)
    else:
        planner_anchor = _load_planner_anchor(cfg)
        package_results = _run_rollout_sequential(cfg, scenarios, state_dict, planner_anchor)

    retrieval_style = str(getattr(cfg, "rollout_retrieval_style", "r2lpl")).strip().lower()
    frame_tasks = (
        _build_road_frame_tasks(package_results, scenarios, cfg)
        if retrieval_style == "road"
        else _build_frame_tasks(package_results, scenarios)
    )
    logger.info(
        "Built %d frame tasks from %d rollout packages (style=%s)",
        len(frame_tasks),
        len(package_results),
        retrieval_style,
    )
    oracle_parallel = bool(getattr(cfg, "oracle_parallel", getattr(cfg, "rollout_parallel", True))) and int(
        getattr(cfg, "oracle_num_workers", 0) or getattr(cfg, "num_workers", 1)
    ) > 1
    if oracle_parallel:
        frame_results = _run_frames_ray(cfg, frame_tasks, planner_anchor=None)
    else:
        planner_anchor = _load_planner_anchor(cfg)
        frame_results = _run_frames_sequential(cfg, frame_tasks, planner_anchor)

    summary_path = _write_summary(cfg, package_results, frame_results)
    logger.info("Finished rollout CL generation. Summary: %s", summary_path)


if __name__ == "__main__":
    main()
