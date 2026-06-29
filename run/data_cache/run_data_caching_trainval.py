# RM_data_caching.py
# This module provides functionality to cache and retrieve precomputed features

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging
import uuid
import os
import gc
import time
from collections import Counter
import pytorch_lightning as pl
import numpy as np
import importlib.util
import sys
# import psutil

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.script.builders.utils.utils_type import is_target_type, validate_type
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import chunk_list, worker_map
from nuplan.planning.utils.multithreading.worker_ray import RayDistributed
from nuplan.common.utils.distributed_scenario_filter import DistributedMode, DistributedScenarioFilter
from nuplan.planning.scenario_builder.abstract_scenario_builder import RepartitionStrategy
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter



from lpl_planner.planning.scene.trajectory_library import get_trajectory_from_scenario
from lpl_planner.planning.scene.scene_manager import SceneManager
from lpl_planner.planning.scene.scene_feature.features import (
    SceneFeature,
    Trajectory,
    AgentPrediction,
)
from lpl_planner.training.dataset import dump_feature_target_to_pickle, load_feature_target_from_pickle
from lpl_planner.training.dataset.scene_token_json import (
    DEFAULT_SCENE_TOKENS_JSON_NAME,
    load_scene_token_selection,
    resolve_scene_tokens_path,
    write_scene_token_selection,
)
from lpl_planner.utils.default_paths import configure_default_paths


configure_default_paths()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

lpl_planner_spec = importlib.util.find_spec("lpl_planner")
lpl_planner_dir = os.path.dirname(lpl_planner_spec.origin)

CONFIG_PATH = os.path.join(lpl_planner_dir, "config/training")
CONFIG_NAME = "custom_caching"



HORIZON = 8  # seconds
TIME_INTERVAL = 0.2  # second
NUM_SAMPLES = 100 # number of trajectories to sample
NEGATIVE_RATIO = 0.5


def _cast_float64_to_float32(data: Any) -> Any:
    """Recursively cast float64 numpy values to float32."""
    if isinstance(data, dict):
        return {k: _cast_float64_to_float32(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_cast_float64_to_float32(v) for v in data]
    if isinstance(data, tuple):
        return tuple(_cast_float64_to_float32(v) for v in data)
    if isinstance(data, np.ndarray) and data.dtype == np.float64:
        return data.astype(np.float32, copy=False)
    if isinstance(data, np.float64):
        return np.float32(data)
    return data


def _rewrite_cache_to_float32(cache_path: Path, feature_type: Any) -> None:
    """Load cache file, cast float64 values to float32, and save it back."""
    feature = load_feature_target_from_pickle(cache_path, feature_type=feature_type)
    serialized = feature.serialize()
    serialized_fp32 = _cast_float64_to_float32(serialized)
    dump_feature_target_to_pickle(cache_path, serialized_fp32)
    del feature


def _log_scenario_type_stats(scenarios: List[AbstractScenario]) -> None:
    """Log scenario type distribution for the currently selected scenarios."""
    type_counter = Counter(getattr(scenario, "scenario_type", None) or "<unknown>" for scenario in scenarios)

    logger.info("Scenario statistics only mode enabled. Total scenarios: %d", len(scenarios))
    print(f"Total scenarios: {len(scenarios)}")

    for scenario_type, count in sorted(type_counter.items(), key=lambda item: (-item[1], item[0])):
        logger.info("scenario_type=%s count=%d", scenario_type, count)
        print(f"{scenario_type}: {count}")


def _write_selected_scene_tokens_json(
    cache_root: Path,
    scenarios: List[AbstractScenario],
    num_nodes: int,
    node_rank: int,
    timeout_seconds: int,
    scene_tokens_path: Optional[str],
    scene_tokens_name: str,
) -> Optional[Path]:
    scenario_tokens = sorted({str(scenario.token) for scenario in scenarios})
    log_names = sorted({str(scenario.log_name) for scenario in scenarios})
    resolved_path = resolve_scene_tokens_path(
        cache_root=cache_root,
        scene_tokens_path=scene_tokens_path,
        scene_tokens_name=scene_tokens_name,
    )

    if num_nodes <= 1:
        final_path, token_count = write_scene_token_selection(
            cache_root=cache_root,
            scenario_tokens=scenario_tokens,
            log_names=log_names,
            scene_tokens_path=str(resolved_path),
            scene_tokens_name=scene_tokens_name,
        )
        logger.info("Wrote selected scenario token json with %d tokens: %s", token_count, final_path)
        return final_path

    shard_dir = cache_root / ".scene_token_json_shards"
    shard_path = shard_dir / f"{resolved_path.name}.rank_{node_rank:04d}.json"
    write_scene_token_selection(
        cache_root=cache_root,
        scenario_tokens=scenario_tokens,
        log_names=log_names,
        scene_tokens_path=str(shard_path),
        scene_tokens_name=scene_tokens_name,
        extra_metadata={
            "node_rank": node_rank,
            "num_nodes": num_nodes,
        },
    )
    logger.info("Wrote selected scenario token shard for node %d: %s", node_rank, shard_path)

    if node_rank != 0:
        return None

    deadline = time.time() + max(timeout_seconds, 1)
    shard_paths = [shard_dir / f"{resolved_path.name}.rank_{rank:04d}.json" for rank in range(num_nodes)]
    while time.time() < deadline:
        if all(path.is_file() for path in shard_paths):
            break
        time.sleep(1.0)

    missing_shards = [str(path) for path in shard_paths if not path.is_file()]
    if missing_shards:
        raise TimeoutError(f"Timed out waiting for selected scene token shards: {missing_shards}")

    merged_tokens = set()
    merged_logs = set()
    for shard_path in shard_paths:
        payload = load_scene_token_selection(shard_path)
        merged_tokens.update(payload.get("scenario_tokens", []))
        merged_logs.update(payload.get("log_names", []))

    final_path, token_count = write_scene_token_selection(
        cache_root=cache_root,
        scenario_tokens=sorted(merged_tokens),
        log_names=sorted(merged_logs),
        scene_tokens_path=str(resolved_path),
        scene_tokens_name=scene_tokens_name,
        extra_metadata={
            "num_nodes": num_nodes,
        },
    )
    logger.info("Wrote merged selected scenario token json with %d tokens: %s", token_count, final_path)
    return final_path


def cache_data(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Optional[Any]]:
    

    def cache_data_internal(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Optional[Any]]:
        
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        # process = psutil.Process(os.getpid())
        node_id = int(os.environ.get("NODE_RANK", 0))
        thread_id = str(uuid.uuid4())

        scenarios: List[AbstractScenario] = [a["scenario"] for a in args]
        cfg: DictConfig = args[0]["cfg"]

        # Create feature preprocessor
        assert cfg.cache.cache_path is not None, f"Cache path cannot be None when caching, got {cfg.cache.cache_path}"

        logger.info("Extracted %s scenarios for thread_id=%s, node_id=%s.", str(len(scenarios)), thread_id, node_id)

        scene_manager = SceneManager(time_step=TIME_INTERVAL,
                                     use_ref_path=False,  # we will extract ref path feature separately in the scene manager, set this to False to save memory)
                                    use_dynamics=False)  # we will extract dynamics feature separately in the scene manager, set this to False to save memory)
        # peak_memory = process.memory_info().rss / (1024 * 1024)  # in MB
        num_success = 0

        for job_idx, scenario in enumerate(scenarios):

            # Check if scenario is already cached
            iteration = 0
            if cfg.split_iteration:
                iteration = args[job_idx].get("iteration", None)
                if iteration is None:
                    logger.warning("Iteration index is missing for scenario %s in split_iteration mode, skipping caching for this scenario.", scenario.token)
                    continue
                scenario_cache_dir = Path(cfg.cache.cache_path) / cfg.job_name / scenario.log_name / scenario.scenario_type / scenario.token / f"iteration_{iteration:04d}"
            else:
                scenario_cache_dir = Path(cfg.cache.cache_path) / cfg.job_name / scenario.log_name / scenario.scenario_type / scenario.token
                    
            
            force_feature_computation = bool(getattr(cfg.cache, "force_feature_computation", False))
            if (not force_feature_computation) and \
               (scenario_cache_dir / "scene_feature.gz").exists() and \
               (scenario_cache_dir / "agent_prediction.gz").exists() and \
               (scenario_cache_dir / "expert_trajectory.gz").exists():
                # _rewrite_cache_to_float32(scenario_cache_dir / "scene_feature.gz", feature_type=SceneFeature)
                # _rewrite_cache_to_float32(scenario_cache_dir / "expert_trajectory.gz", feature_type=Trajectory)
                
                num_success += 1
                continue

            scene_manager.init_from_nuplan_scenario(
                scenario,
                iteration=int(iteration),
                use_route_correction=True,
                use_scenario_for_route_correction=True,
            )
            expert_trajectory = get_trajectory_from_scenario(
                scenario,
                run_step=int(iteration),
                time_horizon=HORIZON,
                time_interval=TIME_INTERVAL,
            )
            # sd = scene_manager.lane_map.cartesian_to_frenet(expert_trajectory[..., :3])
            # s, d = sd[..., 0], sd[..., 1]
            # if np.any(s[1:] - s[:-1] < -2e-1) or np.any(np.abs(d) > 10.0):
            #     logger.info("Invalid expert trajectory for scenario %s, skipping caching.", scenario.token)
            #     continue  # invalid expert trajectory, skip caching
            
            scenario_cache_dir.mkdir(parents=True, exist_ok=True)
            # cache scene feature and agent prediction
            if force_feature_computation or not (scenario_cache_dir / "scene_feature.gz").exists():
                logger.debug("Caching scenario %s ...", scenario.token)
                scene_feature, agent_target = scene_manager.extract_feature_target_from_scenario(
                    scenario,
                    iteration=int(iteration),
                    use_route_correction=True,
                )
                scene_feature = SceneFeature.deserialize(scene_feature)
                dump_feature_target_to_pickle(scenario_cache_dir / "scene_feature.gz", scene_feature.serialize())
                del scene_feature

                agent_prediction = AgentPrediction.deserialize(agent_target)
                dump_feature_target_to_pickle(scenario_cache_dir / "agent_prediction.gz", agent_prediction.serialize())
                del agent_prediction

            # cache expert trajectory
            if force_feature_computation or not (scenario_cache_dir / "expert_trajectory.gz").exists():
                expert_traj_feat = Trajectory(data=expert_trajectory)
                dump_feature_target_to_pickle(scenario_cache_dir / "expert_trajectory.gz", expert_traj_feat.serialize())
            
            # if not (scenario_cache_dir / "extened_expert_trajectory.gz").exists():
                
            #     extended_expert_trajectory = np.concatenate((expert_trajectory, sd), axis=-1)
            #     expert_traj_feat = Trajectory(data=extended_expert_trajectory)
            #     dump_feature_target_to_pickle(scenario_cache_dir / "extened_expert_trajectory.gz", expert_traj_feat.serialize())
            #     del expert_traj_feat, extended_expert_trajectory, sd, expert_trajectory
            
            gc.collect()

            num_success += 1

        logger.info("Cached %d/%d scenarios for thread_id=%s, node_id=%s", num_success, len(scenarios), thread_id, node_id)

        return [f"num_success:{num_success}/{len(scenarios)}"]

    result = cache_data_internal(args)

    # Force a garbage collection to clean up any unused resources
    gc.collect()

    return result

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for dataset caching script.
    :param cfg: omegaconf dictionary
    """
    sys.stdout.flush()
    logger.info("Global Seed set to 0")
    pl.seed_everything(0, workers=True)
    num_nodes = int(os.environ.get("NUM_NODES", 1))
    node_rank = int(os.environ.get("NODE_RANK", 0))

    logger.info("Building Worker")
    worker: WorkerPool = (
        instantiate(cfg.worker)
        if is_target_type(cfg.worker, RayDistributed)
        else instantiate(cfg.worker)
    )
    validate_type(worker, WorkerPool)
    logger.info('Building WorkerPool...DONE!')

    logger.info("Building SceneLoader")

    # Build scenario builder
    scenario_builder = instantiate(cfg.scenario_builder)

    if num_nodes > 1 and cfg.distribute_by_scenario:
        # Partition differently based on how the scenario builder loads the data
        repartition_strategy = scenario_builder.repartition_strategy
        if repartition_strategy == RepartitionStrategy.REPARTITION_FILE_DISK:
            scenario_filter = DistributedScenarioFilter(
                cfg=cfg,
                worker=worker,
                node_rank=node_rank,
                num_nodes=num_nodes,
                synchronization_path=cfg.cache.cache_path,
                timeout_seconds=cfg.get("distributed_timeout_seconds", 3600),
                distributed_mode=cfg.get("distributed_mode", DistributedMode.LOG_FILE_BASED),
            )
            scenarios = scenario_filter.get_scenarios()
        elif repartition_strategy == RepartitionStrategy.INLINE:
            scenario_filter = build_scenario_filter(cfg.scenario_filter)
            scenarios = scenario_builder.get_scenarios(scenario_filter, worker)
            scenarios = chunk_list(scenarios, num_nodes)[node_rank]
        else:
            expected_repartition_strategies = [e.value for e in RepartitionStrategy]
            raise ValueError(
                f"Expected repartition strategy to be in {expected_repartition_strategies}, got {repartition_strategy}."
            )
    else:
        logger.debug(
            "Building scenarios without distribution, if you're running on a multi-node system, make sure you aren't"
            "accidentally caching each scenario multiple times!"
        )
        scenario_filter = build_scenario_filter(cfg.scenario_filter)
        scenarios = scenario_builder.get_scenarios(scenario_filter, worker)

    if cfg.get("scenario_stats_only", False):
        logger.info("Scenario statistics only mode enabled, skipping caching and only logging scenario type distribution.")
        _log_scenario_type_stats(scenarios)
        return

    # construct data points with scenarios and iterations
    if cfg.split_iteration:
        data_points = []
        for scenario in scenarios:
            for iteration in range(scenario.get_number_of_iterations()):
                data_points.append({"scenario": scenario, "cfg": cfg, "iteration": iteration})
    else:
        data_points = [{"scenario": scenario, "cfg": cfg} for scenario in scenarios]


    logger.info("Starting dataset caching of %s files...", str(len(data_points)))

    results = worker_map(worker, cache_data, data_points)

    if bool(getattr(cfg.cache, "write_scene_tokens_json", True)):
        cache_root = Path(cfg.cache.cache_path) / cfg.job_name
        _write_selected_scene_tokens_json(
            cache_root=cache_root,
            scenarios=scenarios,
            num_nodes=num_nodes,
            node_rank=node_rank,
            timeout_seconds=int(cfg.get("distributed_timeout_seconds", 3600)),
            scene_tokens_path=getattr(cfg.cache, "scene_tokens_path", None),
            scene_tokens_name=str(getattr(cfg.cache, "scene_tokens_name", DEFAULT_SCENE_TOKENS_JSON_NAME)),
        )

    logger.info(f"Finished caching {len(data_points)} scenarios for training/validation dataset")
    total_success = 0
    total_scenarios = 0
    for r in results:
        if isinstance(r, list) and len(r) > 0 and isinstance(r[0], str) and r[0].startswith("num_success:"):
            parts = r[0].split(":")[1].split("/")
            total_success += int(parts[0])
            total_scenarios += int(parts[1])
        elif isinstance(r, str):
            parts = r.split(":")[1].split("/")
            total_success += int(parts[0])
            total_scenarios += int(parts[1])
        logger.debug(f"raw result entry: {r}")
    print(f"Total success: {total_success}/{total_scenarios}")



if __name__ == "__main__":
    main()
