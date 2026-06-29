from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from pathlib import Path
import logging
import copy
import torch
import os
import numpy as np
import multiprocessing as mp

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from lpl_planner.training.dataset.dataset_utils import (
    load_feature_target_from_pickle,
)
from lpl_planner.training.dataset.scene_token_json import (
    DEFAULT_SCENE_TOKENS_JSON_NAME,
    load_scene_token_selection,
    resolve_scene_tokens_path,
)
from lpl_planner.training.dataset.cache_manifest import (
    DEFAULT_CACHE_MANIFEST_NAME,
    iter_cache_manifest_entries,
    resolve_cache_manifest_path,
)
from lpl_planner.training.dataset.traj_cluster_artifact import (
    assign_cluster_labels,
    load_traj_cluster_artifact,
    standardize_cluster_features,
)
from lpl_planner.planning.scene.scene_feature.features import (Trajectory,
                                                                SceneFeature,
                                                                AgentPrediction,
                                                                SceneToken,
                                                                AnchorIndice,
                                                                AnchorIndiceV2,
                                                                AnchorScores,
                                                                FactorizedAnchorTarget,
                                                                RolloutTeacherMetadata,
                                                                ReplayPlannerTargets)
from tqdm import tqdm
import random

logger = logging.getLogger(__name__)

CacheEntry = Tuple[Path, ...]


def _resolve_anchor_indice_feature_type(anchor_indice_path: Path) -> Any:
    stem = anchor_indice_path.stem.lower()
    if stem.endswith("_v2") or "anchor_indice_v2" in stem:
        return AnchorIndiceV2
    return AnchorIndice


def _is_factorized_anchor_indice_name(anchor_indice_name: Optional[str]) -> bool:
    if anchor_indice_name in {None, "", "None"}:
        return False
    normalized_name = str(anchor_indice_name).lower()
    return normalized_name.endswith("_v2") or "anchor_indice_v2" in normalized_name


def _resolve_optional_cache_path(
    cache_dir: Path,
    explicit_name: Optional[str],
    default_names: List[str],
) -> Optional[Path]:
    if explicit_name is not None:
        explicit_path = cache_dir / explicit_name
        candidates = [explicit_path] if explicit_path.suffix else [explicit_path.with_suffix(".gz"), explicit_path]
    else:
        candidates = [cache_dir / name for name in default_names]

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _validate_cache_sample(task: Tuple[str, str, Path, Optional[str], str, bool, bool, bool, bool, bool, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]) -> Optional[Tuple[str, str, Path, Optional[str], str, CacheEntry]]:
    token_name, scenario_token, sample_path, scene_type, log_name, use_anchor_indice, use_anchor_score, use_factorized_anchor_target, use_rollout_teacher_metadata, use_replay_planner_targets, anchor_indice_name, anchor_score_name, factorized_anchor_target_name, rollout_teacher_metadata_name, replay_planner_targets_name = task

    scene_path = sample_path / "scene_feature.gz"
    expert_traj_path = sample_path / "expert_trajectory.gz"
    agent_prediction_path = sample_path / "agent_prediction.gz"
    if not (scene_path.is_file() and expert_traj_path.is_file() and agent_prediction_path.is_file()):
        return None

    cache_entry: List[Path] = [scene_path, expert_traj_path, agent_prediction_path]
    if use_anchor_indice:
        anchor_indice_path = _resolve_optional_cache_path(
            sample_path,
            anchor_indice_name,
            ["anchor_indice.gz", "anchor_indice_t1_512.gz"],
        )
        if anchor_indice_path is None:
            return None
        cache_entry.append(anchor_indice_path)

    if use_anchor_score:
        anchor_scores_path = _resolve_optional_cache_path(
            sample_path,
            anchor_score_name,
            ["anchor_scores_k4_4k.gz", "anchor_scores.gz"],
        )
        if anchor_scores_path is None:
            return None
        cache_entry.append(anchor_scores_path)

    if use_factorized_anchor_target:
        factorized_anchor_target_path = _resolve_optional_cache_path(
            sample_path,
            factorized_anchor_target_name,
            ["factorized_anchor_target.gz"],
        )
        if factorized_anchor_target_path is None:
            return None
        cache_entry.append(factorized_anchor_target_path)

    if use_rollout_teacher_metadata:
        rollout_teacher_metadata_path = _resolve_optional_cache_path(
            sample_path,
            rollout_teacher_metadata_name,
            ["rollout_teacher_metadata.gz"],
        )
        if rollout_teacher_metadata_path is None:
            return None
        cache_entry.append(rollout_teacher_metadata_path)

    if use_replay_planner_targets:
        replay_planner_targets_path = _resolve_optional_cache_path(
            sample_path,
            replay_planner_targets_name,
            ["replay_planner_targets.gz"],
        )
        if replay_planner_targets_path is None:
            return None
        cache_entry.append(replay_planner_targets_path)

    return token_name, scenario_token, sample_path, scene_type, log_name, tuple(cache_entry)


@dataclass
class SampleMeta:
    token: str
    scenario_token: str
    sample_path: Path
    log_name: str
    scene_type: Optional[str] = None

def safe_load_feature(file_path: Path, feature_type: Any) -> Any:
    try:
        return load_feature_target_from_pickle(file_path, feature_type=feature_type)
    except Exception as e:
        logger.error(f"Error loading feature from {file_path}: {e}")
        raise

class MVDataset(torch.utils.data.Dataset):
    """Dataset wrapper for planning model datasets from cache only."""

    def __init__(
        self,
        cache_path: str,
        log_names: Optional[List[str]] = None,
        future_sampling : Optional[TrajectorySampling] = None,
        scene_tokens: Optional[List[str]] = None,
        scene_tokens_path: Optional[str] = None,
        manifest_path: Optional[str] = None,
        expand_iteration: bool = False,
        max_samples: Optional[int] = None,
        use_anchor_indice: bool = False,
        use_anchor_score: bool = False,
        use_factorized_anchor_target: bool = False,
        anchor_indice_name: Optional[str] = None,
        anchor_score_name: Optional[str] = None,
        factorized_anchor_target_name: Optional[str] = None,
        use_rollout_teacher_metadata: bool = False,
        rollout_teacher_metadata_name: Optional[str] = None,
        use_replay_planner_targets: bool = False,
        replay_planner_targets_name: Optional[str] = None,
        memory_ratio: Optional[float] = None,
        memory_method: Optional[str] = None,
        memory_random_seed: int = 0,
        traj_cluster_num_groups: int = 32,
        traj_cluster_artifact_path: Optional[str] = None,
        as_source_memory: bool = False,
        use_manifest: bool = False,
    ):
        """
        Initializes the dataset module.
        :param cache_path: directory to cache folder
        :param sample_num: number of samples to load
        :param sample_ratio: ratio of samples to load
        :param log_names: optional list of log folder to consider, defaults to None
        """
        super().__init__()
        assert Path(cache_path).is_dir(), f"Cache path {cache_path} does not exist!"
        self._cache_path = Path(cache_path)
        token_selection = self._resolve_scene_token_selection(scene_tokens_path)
        if token_selection is not None and use_manifest:
            if scene_tokens is None:
                scene_tokens = token_selection.get("scenario_tokens", [])
            if log_names is None and token_selection.get("log_names"):
                log_names = token_selection["log_names"]
        self._used_scene_tokens: Optional[Set[str]] = None if scene_tokens is None else {str(token) for token in scene_tokens}
        self._requested_log_names: Optional[Set[str]] = None if log_names is None else {Path(log_name).name for log_name in log_names}
        self._max_samples = max_samples
        self.use_anchor_indice = use_anchor_indice
        self.use_anchor_score = bool(use_anchor_score or (use_anchor_indice and not _is_factorized_anchor_indice_name(anchor_indice_name)))
        self.use_factorized_anchor_target = bool(use_factorized_anchor_target)
        self.anchor_indice_name = anchor_indice_name
        self.anchor_score_name = anchor_score_name
        self.factorized_anchor_target_name = factorized_anchor_target_name
        self.use_rollout_teacher_metadata = bool(use_rollout_teacher_metadata)
        self.rollout_teacher_metadata_name = rollout_teacher_metadata_name
        self.use_replay_planner_targets = bool(use_replay_planner_targets)
        self.replay_planner_targets_name = replay_planner_targets_name
        self.future_sampling = future_sampling
        self.memory_ratio = memory_ratio
        self.memory_method = memory_method
        self.memory_random_seed = int(memory_random_seed)
        self.traj_cluster_num_groups = max(int(traj_cluster_num_groups), 1)
        self.traj_cluster_artifact_path = None if traj_cluster_artifact_path in {None, "", "None"} else str(Path(traj_cluster_artifact_path).expanduser())
        self._traj_cluster_artifact: Optional[Dict[str, Any]] = None
        self.as_source_memory = bool(as_source_memory)
        self.memory_tokens: Optional[List[str]] = None
        self._sample_meta: Dict[str, SampleMeta] = {}
        self._manifest_path = self._resolve_manifest_path(manifest_path)
        self._val_split_applied = False
        self._is_val_split_dataset = False
        self._val_split_source_token_count: Optional[int] = None

        if log_names is not None:
            self.log_names = [Path(log_name) for log_name in log_names if (self._cache_path / log_name).is_dir()]
        else:
            self.log_names = [log_name for log_name in self._cache_path.iterdir() if log_name.is_dir()]

        if self._manifest_path is not None and use_manifest:
            self._valid_cache_paths = self._load_valid_caches_from_manifest(cache_path=self._cache_path)
        else:
            self._valid_cache_paths = self._load_valid_caches(
                cache_path=self._cache_path,
                log_names=self.log_names,
                expand_iteration=expand_iteration,
            )
        
        self.tokens = list(self._valid_cache_paths.keys())
        if self.memory_ratio is not None or self.memory_method is not None:
            if self.memory_ratio is None or self.memory_method is None:
                raise ValueError("memory_ratio and memory_method must be set together")
            self.extract_memory(
                memory_ratio=self.memory_ratio,
                method=self.memory_method,
                random_seed=self.memory_random_seed,
                traj_cluster_num_groups=self.traj_cluster_num_groups,
                as_source_memory=self.as_source_memory,
            )

        logger.info(f"Initialized MVDataset with anchor_indice enabled={self.use_anchor_indice} (name={self.anchor_indice_name}), \n"
                    f"anchor_score enabled={self.use_anchor_score} (name={self.anchor_score_name}), \n"
                    f"factorized_anchor_target enabled={self.use_factorized_anchor_target} (name={self.factorized_anchor_target_name}), \n"
                    f"rollout_teacher_metadata enabled={self.use_rollout_teacher_metadata} (name={self.rollout_teacher_metadata_name}).\n"
                    f"replay_planner_targets enabled={self.use_replay_planner_targets} (name={self.replay_planner_targets_name}).\n"
                    f"total valid samples found: {len(self._valid_cache_paths)}. Tokens retained after filtering: {len(self.tokens)}."
                    )

    def _resolve_scene_token_selection(self, scene_tokens_path: Optional[str]) -> Optional[Dict[str, Any]]:
        resolved_scene_tokens_path = resolve_scene_tokens_path(
            cache_root=self._cache_path,
            scene_tokens_path=scene_tokens_path,
            scene_tokens_name=DEFAULT_SCENE_TOKENS_JSON_NAME,
        )
        selection_was_explicit = scene_tokens_path not in {None, "", "None"}
        if resolved_scene_tokens_path.is_file():
            logger.info("Using scene token json: %s", resolved_scene_tokens_path)
            return load_scene_token_selection(resolved_scene_tokens_path)
        if selection_was_explicit:
            raise FileNotFoundError(f"Scene token json does not exist: {resolved_scene_tokens_path}")
        return None

    def _resolve_manifest_path(self, manifest_path: Optional[str]) -> Optional[Path]:
        resolved_manifest_path = resolve_cache_manifest_path(
            cache_root=self._cache_path,
            manifest_path=manifest_path,
            manifest_name=DEFAULT_CACHE_MANIFEST_NAME,
        )
        manifest_was_explicit = manifest_path not in {None, "", "None"}
        if resolved_manifest_path.is_file():
            logger.info("Using selected manifest: %s", resolved_manifest_path)
            return resolved_manifest_path
        if manifest_was_explicit:
            raise FileNotFoundError(f"Selected manifest does not exist: {resolved_manifest_path}")
        return None


    def _resolve_anchor_indice_path(self, cache_dir: Path) -> Optional[Path]:
        candidates = self._build_optional_feature_candidates(
            cache_dir=cache_dir,
            explicit_name=self.anchor_indice_name,
            default_names=["anchor_indice.gz", "anchor_indice_t1_512.gz"],
        )
        for path in candidates:
            if path.is_file():
                return path
        return None

    def _resolve_anchor_scores_path(self, cache_dir: Path) -> Optional[Path]:
        candidates = self._build_optional_feature_candidates(
            cache_dir=cache_dir,
            explicit_name=self.anchor_score_name,
            default_names=["anchor_scores_k4_4k.gz", "anchor_scores.gz"],
        )
        for path in candidates:
            if path.is_file():
                return path
        return None

    def _resolve_rollout_teacher_metadata_path(self, cache_dir: Path) -> Optional[Path]:
        candidates = self._build_optional_feature_candidates(
            cache_dir=cache_dir,
            explicit_name=self.rollout_teacher_metadata_name,
            default_names=["rollout_teacher_metadata.gz"],
        )
        for path in candidates:
            if path.is_file():
                return path
        return None

    def _resolve_replay_planner_targets_path(self, cache_dir: Path) -> Optional[Path]:
        candidates = self._build_optional_feature_candidates(
            cache_dir=cache_dir,
            explicit_name=self.replay_planner_targets_name,
            default_names=["replay_planner_targets.gz"],
        )
        for path in candidates:
            if path.is_file():
                return path
        return None

    def _resolve_factorized_anchor_target_path(self, cache_dir: Path) -> Optional[Path]:
        candidates = self._build_optional_feature_candidates(
            cache_dir=cache_dir,
            explicit_name=self.factorized_anchor_target_name,
            default_names=["factorized_anchor_target.gz"],
        )
        for path in candidates:
            if path.is_file():
                return path
        return None

    def _build_optional_feature_candidates(
        self,
        cache_dir: Path,
        explicit_name: Optional[str],
        default_names: List[str],
    ) -> List[Path]:
        if explicit_name is not None:
            explicit_path = cache_dir / explicit_name
            if explicit_path.suffix:
                return [explicit_path]
            return [explicit_path.with_suffix(".gz"), explicit_path]
        return [cache_dir / name for name in default_names]

    def __len__(self) -> int:
        """
        :return: number of samples to load
        """
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Loads and returns pair of feature and target dict from data.
        :param idx: index of sample to load.
        :return: tuple of feature and target dictionary
        """
        return self._load_scene_with_token(self.tokens[idx])

    def _iter_sample_dirs(
        self,
        cache_path: Path,
        log_names: List[Path],
        expand_iteration: bool,
    ):
        for log_name in tqdm(log_names, desc="Loading Valid Caches"):
            log_path = cache_path / log_name
            if not log_path.is_dir():
                continue
            if expand_iteration:
                for first_level_path in log_path.iterdir():
                    if not first_level_path.is_dir():
                        continue

                    direct_iteration_paths = [
                        child for child in first_level_path.iterdir()
                        if child.is_dir() and child.name.startswith("iteration_")
                    ]
                    if direct_iteration_paths:
                        token_path = first_level_path
                        if self._used_scene_tokens is not None and token_path.name not in self._used_scene_tokens:
                            continue
                        for iteration_path in sorted(direct_iteration_paths):
                            yield f"{token_path.name}_{iteration_path.name}", token_path.name, iteration_path, None, log_name.name
                        continue

                    scene_type = first_level_path.name
                    for iteration_path in sorted(first_level_path.rglob("iteration_*")):
                        if not iteration_path.is_dir():
                            continue
                        token_path = iteration_path.parent
                        if self._used_scene_tokens is not None and token_path.name not in self._used_scene_tokens:
                            continue
                        yield f"{token_path.name}_{iteration_path.name}", token_path.name, iteration_path, scene_type, log_name.name
            else:
                for type_path in log_path.iterdir():
                    if not type_path.is_dir():
                        continue
                    for token_path in type_path.iterdir():
                        if not token_path.is_dir():
                            continue
                        if self._used_scene_tokens is not None and token_path.name not in self._used_scene_tokens:
                            continue
                        yield token_path.name, token_path.name, token_path, type_path.name, log_name.name

    def _iter_manifest_sample_dirs(self, cache_path: Path):
        assert self._manifest_path is not None
        for entry in iter_cache_manifest_entries(self._manifest_path):
            log_name = str(entry.get("log_name", ""))
            if self._requested_log_names is not None and log_name not in self._requested_log_names:
                continue

            scenario_token = str(entry.get("scenario_token", ""))
            if self._used_scene_tokens is not None and scenario_token not in self._used_scene_tokens:
                continue

            relative_path = entry.get("relative_path")
            if not relative_path:
                continue

            sample_path = cache_path / str(relative_path)
            if not sample_path.is_dir():
                continue

            sample_token = str(entry.get("sample_token") or scenario_token)
            scene_type = entry.get("scene_type")
            yield sample_token, scenario_token, sample_path, scene_type, log_name

    def _load_valid_caches(
        self,
        cache_path: Path,
        log_names: List[Path],
        expand_iteration: bool = False,
        num_workers: int = 4,
    ) -> Dict[str, CacheEntry]:
        """
        Helper method to load valid cache paths.
        :param cache_path: directory of training cache folder
        :param feature_builders: list of feature builders
        :param target_builders: list of target builders
        :param log_names: list of log paths to load
        :return: dictionary of tokens and sample paths as keys / values
        """

        valid_cache_paths: Dict[str, CacheEntry] = {}

        tasks = [
            (
                token_name,
                scenario_token,
                sample_path,
                scene_type,
                log_name,
                self.use_anchor_indice,
                self.use_anchor_score,
                self.use_factorized_anchor_target,
                self.use_rollout_teacher_metadata,
                self.use_replay_planner_targets,
                self.anchor_indice_name,
                self.anchor_score_name,
                self.factorized_anchor_target_name,
                self.rollout_teacher_metadata_name,
                self.replay_planner_targets_name,
            )
            for token_name, scenario_token, sample_path, scene_type, log_name in self._iter_sample_dirs(
                cache_path=cache_path,
                log_names=log_names,
                expand_iteration=expand_iteration,
            )
        ]

        if not tasks:
            return valid_cache_paths

        if num_workers <= 1:
            results = map(_validate_cache_sample, tasks)
            progress_desc = "Validating cache entries"
            for result in tqdm(results, total=len(tasks), desc=progress_desc):
                if result is None:
                    continue
                token_name, scenario_token, sample_path, scene_type, log_name, cache_entry = result
                valid_cache_paths[token_name] = cache_entry
                self._sample_meta[token_name] = SampleMeta(
                    token=token_name,
                    scenario_token=scenario_token,
                    sample_path=sample_path,
                    log_name=log_name,
                    scene_type=scene_type,
                )
                if self._max_samples is not None and len(valid_cache_paths) >= self._max_samples:
                    break
            return valid_cache_paths
        
        num_workers = max(num_workers, int((os.cpu_count() or 1) * 0.9))
        worker_count = min(max(1, int(num_workers)), len(tasks))
        chunk_size = max(1, len(tasks) // (worker_count * 8))
        with mp.Pool(processes=worker_count) as pool:
            for result in tqdm(
                pool.imap(_validate_cache_sample, tasks, chunksize=chunk_size),
                total=len(tasks),
                desc="Validating cache entries",
            ):
                if result is None:
                    continue
                token_name, scenario_token, sample_path, scene_type, log_name, cache_entry = result
                valid_cache_paths[token_name] = cache_entry
                self._sample_meta[token_name] = SampleMeta(
                    token=token_name,
                    scenario_token=scenario_token,
                    sample_path=sample_path,
                    log_name=log_name,
                    scene_type=scene_type,
                )
                if self._max_samples is not None and len(valid_cache_paths) >= self._max_samples:
                    break

        return valid_cache_paths

    def _load_valid_caches_from_manifest(
        self,
        cache_path: Path,
    ) -> Dict[str, CacheEntry]:
        valid_cache_paths: Dict[str, CacheEntry] = {}

        for token_name, scenario_token, sample_path, scene_type, log_name in self._iter_manifest_sample_dirs(cache_path=cache_path):
            if self._max_samples is not None and len(valid_cache_paths) >= self._max_samples:
                break

            self._append_valid_cache_entry(
                valid_cache_paths=valid_cache_paths,
                token_name=token_name,
                scenario_token=scenario_token,
                sample_path=sample_path,
                scene_type=scene_type,
                log_name=log_name,
            )

        return valid_cache_paths

    def _append_valid_cache_entry(
        self,
        valid_cache_paths: Dict[str, CacheEntry],
        token_name: str,
        scenario_token: str,
        sample_path: Path,
        scene_type: Optional[str],
        log_name: str,
    ) -> None:
        if self._max_samples is not None and len(valid_cache_paths) >= self._max_samples:
            return

        scene_path = sample_path / "scene_feature.gz"
        expert_traj_path = sample_path / "expert_trajectory.gz"
        agent_prediction_path = sample_path / "agent_prediction.gz"

        if not (scene_path.is_file() and expert_traj_path.is_file() and agent_prediction_path.is_file()):
            return

        cache_entry: List[Path] = [scene_path, expert_traj_path, agent_prediction_path]
        if self.use_anchor_indice:
            anchor_indice_path = self._resolve_anchor_indice_path(sample_path)
            if anchor_indice_path is None:
                return
            cache_entry.append(anchor_indice_path)
        if self.use_anchor_score:
            anchor_scores_path = self._resolve_anchor_scores_path(sample_path)
            if anchor_scores_path is None:
                return
            cache_entry.append(anchor_scores_path)
        if self.use_factorized_anchor_target:
            factorized_anchor_target_path = self._resolve_factorized_anchor_target_path(sample_path)
            if factorized_anchor_target_path is None:
                return
            cache_entry.append(factorized_anchor_target_path)
        if self.use_rollout_teacher_metadata:
            rollout_teacher_metadata_path = self._resolve_rollout_teacher_metadata_path(sample_path)
            if rollout_teacher_metadata_path is None:
                return
            cache_entry.append(rollout_teacher_metadata_path)
        if self.use_replay_planner_targets:
            replay_planner_targets_path = self._resolve_replay_planner_targets_path(sample_path)
            if replay_planner_targets_path is None:
                return
            cache_entry.append(replay_planner_targets_path)

        valid_cache_paths[token_name] = tuple(cache_entry)
        self._sample_meta[token_name] = SampleMeta(
            token=token_name,
            scenario_token=scenario_token,
            sample_path=sample_path,
            log_name=log_name,
            scene_type=scene_type,
        )

    def _normalize_memory_method(self, method: str) -> str:
        normalized = method.strip().lower()
        aliases = {
            "scene": "scene_type",
            "task": "scene_type",
            "task_type": "scene_type",
            "scene_type": "scene_type",
            "traj": "traj_cluster",
            "traj_cluster": "traj_cluster",
            "trajectory_cluster": "traj_cluster",
        }
        if normalized not in aliases:
            raise ValueError(f"unsupported memory method: {method}")
        return aliases[normalized]

    def _validate_memory_ratio(self, memory_ratio: float) -> float:
        ratio = float(memory_ratio)
        if ratio <= 0.0 or ratio > 1.0:
            raise ValueError(f"memory_ratio must be in (0, 1], got {memory_ratio}")
        return ratio

    def _retain_tokens(self, selected_tokens: List[str]) -> None:
        selected_set = set(selected_tokens)
        self._valid_cache_paths = {
            token: self._valid_cache_paths[token]
            for token in selected_tokens
            if token in self._valid_cache_paths
        }
        self._sample_meta = {
            token: self._sample_meta[token]
            for token in selected_tokens
            if token in self._sample_meta
        }
        self.tokens = [token for token in selected_tokens if token in selected_set and token in self._valid_cache_paths]

        if self.memory_tokens is not None:
            self.memory_tokens = [token for token in self.memory_tokens if token in self._valid_cache_paths]

    def _build_token_subset_dataset(self, selected_tokens: List[str]) -> "MVDataset":
        subset_dataset = object.__new__(MVDataset)
        subset_dataset.__dict__ = copy.copy(self.__dict__)
        subset_dataset._valid_cache_paths = {
            token: self._valid_cache_paths[token]
            for token in selected_tokens
            if token in self._valid_cache_paths
        }
        subset_dataset._sample_meta = {
            token: self._sample_meta[token]
            for token in selected_tokens
            if token in self._sample_meta
        }
        subset_dataset.tokens = [token for token in selected_tokens if token in subset_dataset._valid_cache_paths]
        subset_dataset.memory_tokens = None
        subset_dataset._val_split_applied = False
        subset_dataset._is_val_split_dataset = False
        subset_dataset._val_split_source_token_count = None
        return subset_dataset

    def _validate_val_split_ratio(self, val_ratio: float) -> float:
        ratio = float(val_ratio)
        if ratio < 0.0 or ratio >= 1.0:
            raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
        return ratio

    def split_val_dataset(self, val_ratio: float, random_seed: int = 0) -> Optional["MVDataset"]:
        ratio = self._validate_val_split_ratio(val_ratio)
        if ratio == 0.0:
            logger.info("val_ratio=0.0, skipping validation split")
            return None

        if self._is_val_split_dataset:
            raise RuntimeError("cannot split a dataset that is already produced by split_val_dataset")

        if self._val_split_applied:
            raise RuntimeError("split_val_dataset has already been called on this dataset")

        token_list = list(self.tokens)
        if len(token_list) <= 1:
            raise ValueError("split_val_dataset requires at least 2 samples when val_ratio > 0")

        rng = random.Random(int(random_seed))
        shuffled_tokens = list(token_list)
        rng.shuffle(shuffled_tokens)

        val_count = int(round(len(shuffled_tokens) * ratio))
        val_count = min(max(val_count, 1), len(shuffled_tokens) - 1)
        val_token_set = set(shuffled_tokens[:val_count])

        train_tokens = [token for token in token_list if token not in val_token_set]
        val_tokens = [token for token in token_list if token in val_token_set]

        val_dataset = self._build_token_subset_dataset(val_tokens)
        val_dataset._is_val_split_dataset = True
        val_dataset._val_split_source_token_count = len(token_list)

        self._retain_tokens(train_tokens)
        self._val_split_applied = True
        self._val_split_source_token_count = len(token_list)

        logger.info(
            "Split dataset into train=%s / val=%s with val_ratio=%s and seed=%s",
            len(train_tokens),
            len(val_tokens),
            ratio,
            random_seed,
        )
        return val_dataset

    def get_traj_cluster_artifact(self) -> Optional[Dict[str, Any]]:
        if self.traj_cluster_artifact_path is None:
            return None
        if self._traj_cluster_artifact is None:
            self._traj_cluster_artifact = load_traj_cluster_artifact(self.traj_cluster_artifact_path)
            artifact_cluster_num = int(self._traj_cluster_artifact["cluster_num_groups"])
            if artifact_cluster_num != self.traj_cluster_num_groups:
                logger.info(
                    "Loaded traj cluster artifact with cluster_num_groups=%s (config requested %s)",
                    artifact_cluster_num,
                    self.traj_cluster_num_groups,
                )
        return self._traj_cluster_artifact

    def _compute_traj_cluster_features(self, tokens: List[str]) -> Tuple[List[str], np.ndarray]:
        token_list = list(tokens)
        if not token_list:
            return [], np.zeros((0, 0), dtype=np.float32)
        trajs = np.stack([self._load_expert_trajectory_array(token) for token in token_list], axis=0)
        feats = self._featureize_xy_yaw_for_cluster(trajs).astype(np.float32)
        return token_list, feats

    def build_traj_cluster_partition_context(
        self,
        tokens: List[str],
        random_seed: int,
        traj_cluster_num_groups: Optional[int] = None,
    ) -> Tuple[Dict[str, str], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        token_list, feats = self._compute_traj_cluster_features(list(tokens))
        if not token_list:
            return {}, {}, {}

        artifact = self.get_traj_cluster_artifact()
        if artifact is not None:
            feats_std = standardize_cluster_features(
                feats,
                feature_mean=artifact["feature_mean"],
                feature_std=artifact["feature_std"],
            )
            token_to_cluster = artifact.get("token_to_cluster") or {}
            missing_indices = [idx for idx, token in enumerate(token_list) if token not in token_to_cluster]
            labels = np.full((len(token_list),), -1, dtype=np.int64)
            for idx, token in enumerate(token_list):
                if token in token_to_cluster:
                    labels[idx] = int(token_to_cluster[token])
            if missing_indices:
                assigned = assign_cluster_labels(feats_std[missing_indices], artifact["cluster_centers"])
                labels[np.asarray(missing_indices, dtype=np.int64)] = assigned
            partition_map = {token: f"cluster_{int(label)}" for token, label in zip(token_list, labels.tolist())}
            partition_centers = {
                f"cluster_{cluster_idx}": center
                for cluster_idx, center in enumerate(np.asarray(artifact["cluster_centers"], dtype=np.float32))
            }
            token_features = {token: feature for token, feature in zip(token_list, feats_std)}
            return partition_map, partition_centers, token_features

        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            raise ImportError("traj_cluster memory extraction requires scikit-learn") from exc

        feats_std = standardize_cluster_features(feats)
        cluster_num = min(max(int(traj_cluster_num_groups or self.traj_cluster_num_groups), 1), len(token_list))
        labels = KMeans(n_clusters=cluster_num, n_init=10, random_state=random_seed).fit_predict(feats_std)
        partition_map = {token: f"cluster_{int(label)}" for token, label in zip(token_list, labels.tolist())}
        partition_centers: Dict[str, np.ndarray] = {}
        for cluster_idx in range(cluster_num):
            member_mask = labels == cluster_idx
            if not np.any(member_mask):
                continue
            partition_centers[f"cluster_{cluster_idx}"] = feats_std[member_mask].mean(axis=0)
        token_features = {token: feature for token, feature in zip(token_list, feats_std)}
        return partition_map, partition_centers, token_features

    def _load_expert_trajectory_array(self, token: str) -> np.ndarray:
        cache_paths = self._valid_cache_paths[token]
        expert_traj_path = cache_paths[1]
        expert_trajectory_feature: Trajectory = safe_load_feature(expert_traj_path, feature_type=Trajectory)
        if self.future_sampling is not None:
            expert_trajectory_feature = Trajectory(expert_trajectory_feature.data[..., 1:, :])
        return np.asarray(expert_trajectory_feature.data, dtype=np.float32)

    def _featureize_xy_yaw_for_cluster(self, trajs: np.ndarray) -> np.ndarray:
        if trajs.ndim != 3 or trajs.shape[2] < 3:
            raise ValueError(f"Expected trajectory array with shape [N, T, D>=3], got {trajs.shape}")
        xy = np.stack([trajs[:, :, 0], trajs[:, :, 1]], axis=-1).astype(np.float32)
        xy = np.clip(xy / 100.0, -100.0, 100.0)
        yaw = trajs[:, :, 2].astype(np.float32)
        yaw_feat = np.stack([np.sin(yaw), np.cos(yaw)], axis=-1).astype(np.float32)
        feats = np.concatenate([xy, yaw_feat], axis=-1)
        return feats.reshape(feats.shape[0], -1)

    def _balanced_sample_from_buckets(
        self,
        buckets: Dict[Any, List[str]],
        total_keep: int,
        rng: random.Random,
    ) -> List[str]:
        prepared = {
            key: list(tokens)
            for key, tokens in buckets.items()
            if len(tokens) > 0
        }
        for tokens in prepared.values():
            rng.shuffle(tokens)

        selected: List[str] = []
        while len(selected) < total_keep and prepared:
            bucket_keys = list(prepared.keys())
            rng.shuffle(bucket_keys)
            empty_keys: List[Any] = []
            for key in bucket_keys:
                tokens = prepared[key]
                if not tokens:
                    empty_keys.append(key)
                    continue
                selected.append(tokens.pop())
                if not tokens:
                    empty_keys.append(key)
                if len(selected) >= total_keep:
                    break
            for key in empty_keys:
                prepared.pop(key, None)

        rng.shuffle(selected)
        return selected

    def _select_memory_tokens_by_scene_type(self, memory_ratio: float, random_seed: int) -> List[str]:
        rng = random.Random(random_seed)
        buckets: Dict[str, List[str]] = {}
        for token in self.tokens:
            meta = self._sample_meta.get(token)
            scene_type = meta.scene_type if meta is not None and meta.scene_type is not None else "__unknown__"
            buckets.setdefault(scene_type, []).append(token)

        total_keep = max(1, int(round(len(self.tokens) * memory_ratio)))
        return self._balanced_sample_from_buckets(buckets, total_keep, rng)

    def _select_memory_tokens_by_traj_cluster(
        self,
        memory_ratio: float,
        traj_cluster_num_groups: int,
        random_seed: int,
    ) -> List[str]:
        total_keep = max(1, int(round(len(self.tokens) * memory_ratio)))
        if total_keep >= len(self.tokens):
            return list(self.tokens)

        rng = random.Random(random_seed)
        token_list = list(self.tokens)
        partition_map, _, _ = self.build_traj_cluster_partition_context(
            token_list,
            random_seed=random_seed,
            traj_cluster_num_groups=traj_cluster_num_groups,
        )

        buckets: Dict[int, List[str]] = {}
        for token in token_list:
            partition = partition_map.get(token, "cluster_0")
            label = int(str(partition).split("cluster_")[-1]) if str(partition).startswith("cluster_") else 0
            buckets.setdefault(label, []).append(token)

        return self._balanced_sample_from_buckets(buckets, total_keep, rng)

    def select_memory_tokens(
        self,
        memory_ratio: float,
        method: str,
        random_seed: int = 0,
        traj_cluster_num_groups: Optional[int] = None,
    ) -> List[str]:
        ratio = self._validate_memory_ratio(memory_ratio)
        normalized_method = self._normalize_memory_method(method)
        if normalized_method == "scene_type":
            return self._select_memory_tokens_by_scene_type(ratio, random_seed)
        return self._select_memory_tokens_by_traj_cluster(
            memory_ratio=ratio,
            traj_cluster_num_groups=traj_cluster_num_groups or self.traj_cluster_num_groups,
            random_seed=random_seed,
        )

    def extract_memory(
        self,
        memory_ratio: float,
        method: str,
        random_seed: int = 0,
        traj_cluster_num_groups: Optional[int] = None,
        as_source_memory: bool = False,
    ) -> List[str]:
        selected_tokens = self.select_memory_tokens(
            memory_ratio=memory_ratio,
            method=method,
            random_seed=random_seed,
            traj_cluster_num_groups=traj_cluster_num_groups,
        )
        self.memory_tokens = list(selected_tokens)
        if as_source_memory:
            self.as_source_memory = True
            self._retain_tokens(self.memory_tokens)
        return self.memory_tokens

    def _load_scene_with_token(self, token: str) -> Tuple[FeaturesType, TargetsType]:
        """
        Helper method to load sample tensors given token
        :param token: unique string identifier of sample
        :return: tuple of feature and target dictionaries
        """

        cache_paths = self._valid_cache_paths[token]
        cursor = 0
        scene_path = cache_paths[cursor]
        cursor += 1
        expert_traj_path = cache_paths[cursor]
        cursor += 1
        agent_prediction_path = cache_paths[cursor]
        cursor += 1
        anchor_indice_path = cache_paths[cursor] if self.use_anchor_indice else None
        if self.use_anchor_indice:
            cursor += 1
        anchor_scores_path = cache_paths[cursor] if self.use_anchor_score else None
        if self.use_anchor_score:
            cursor += 1
        factorized_anchor_target_path = cache_paths[cursor] if self.use_factorized_anchor_target else None
        if self.use_factorized_anchor_target:
            cursor += 1
        rollout_teacher_metadata_path = cache_paths[cursor] if self.use_rollout_teacher_metadata else None
        if self.use_rollout_teacher_metadata:
            cursor += 1
        replay_planner_targets_path = cache_paths[cursor] if self.use_replay_planner_targets else None

        features: FeaturesType = {}
        targets: TargetsType = {}
        scene_feature = safe_load_feature(scene_path, feature_type=SceneFeature)
        agent_feature = safe_load_feature(agent_prediction_path, feature_type=AgentPrediction)
        
        expert_trajectory_feature: Trajectory = safe_load_feature(expert_traj_path, feature_type=Trajectory)

        features[SceneFeature.get_feature_unique_name()] = scene_feature.to_feature_tensor()
        scenario_token = self._sample_meta.get(token).scenario_token if token in self._sample_meta else token
        features['scenario_token'] = SceneToken(scenario_token)
        
        # expert trajectory and scores as targets
        if self.future_sampling is not None:
            traj_time_step = 0.1
            time_idx = int(self.future_sampling.time_horizon / traj_time_step) + 1
            desired_time_step = self.future_sampling.interval_length
            idx_interval = int(desired_time_step//traj_time_step)
            expert_trajectory_feature = Trajectory(expert_trajectory_feature.data[...,1:,:])
        targets['expert_trajectory'] = expert_trajectory_feature.to_feature_tensor()
        targets[AgentPrediction.get_feature_unique_name()] = agent_feature.to_feature_tensor()
        if self.use_anchor_indice and anchor_indice_path is not None:
            anchor_indice_feature_type = _resolve_anchor_indice_feature_type(anchor_indice_path)
            anchor_indice_feature = safe_load_feature(anchor_indice_path, feature_type=anchor_indice_feature_type)
            targets[anchor_indice_feature_type.get_feature_unique_name()] = anchor_indice_feature.to_feature_tensor()
        if self.use_anchor_score and anchor_scores_path is not None:
            anchor_scores_feature = safe_load_feature(anchor_scores_path, feature_type=AnchorScores)
            targets[AnchorScores.get_feature_unique_name()] = anchor_scores_feature.to_feature_tensor()
        if self.use_factorized_anchor_target and factorized_anchor_target_path is not None:
            factorized_anchor_target = safe_load_feature(
                factorized_anchor_target_path,
                feature_type=FactorizedAnchorTarget,
            )
            targets[FactorizedAnchorTarget.get_feature_unique_name()] = factorized_anchor_target.to_feature_tensor()
        if self.use_rollout_teacher_metadata and rollout_teacher_metadata_path is not None:
            rollout_teacher_metadata = safe_load_feature(
                rollout_teacher_metadata_path,
                feature_type=RolloutTeacherMetadata,
            )
            targets[RolloutTeacherMetadata.get_feature_unique_name()] = rollout_teacher_metadata.to_feature_tensor()
        if self.use_replay_planner_targets and replay_planner_targets_path is not None:
            replay_planner_targets = safe_load_feature(
                replay_planner_targets_path,
                feature_type=ReplayPlannerTargets,
            )
            targets[ReplayPlannerTargets.get_feature_unique_name()] = replay_planner_targets.to_feature_tensor()
        
        return (features, targets)
    
    def add_dataset(self, other_dataset: "MVDataset") -> None:
        """
        Merge another MVDataset into the current dataset.

        The two datasets must have compatible loading semantics. Duplicate tokens are
        rejected to avoid silently overriding cached samples.
        :param other_dataset: another MVDataset instance to merge with the current dataset
        """
        if other_dataset is None:
            raise ValueError("other_dataset must not be None")

        if not isinstance(other_dataset, MVDataset):
            raise TypeError(f"expected MVDataset, got {type(other_dataset)!r}")

        if self is other_dataset:
            return

        if self._val_split_applied or self._is_val_split_dataset:
            logger.warning(
                "add_dataset called after split_val_dataset; existing train/val split is now stale. "
                "If you need a refreshed validation split, rebuild from the unsplit dataset and call split_val_dataset after merging."
            )

        if self.use_anchor_indice != other_dataset.use_anchor_indice:
            raise ValueError(
                "cannot merge datasets with different use_anchor_indice settings: "
                f"{self.use_anchor_indice} vs {other_dataset.use_anchor_indice}"
            )

        if self.use_anchor_score != other_dataset.use_anchor_score:
            raise ValueError(
                "cannot merge datasets with different use_anchor_score settings: "
                f"{self.use_anchor_score} vs {other_dataset.use_anchor_score}"
            )

        if self.use_factorized_anchor_target != other_dataset.use_factorized_anchor_target:
            raise ValueError(
                "cannot merge datasets with different use_factorized_anchor_target settings: "
                f"{self.use_factorized_anchor_target} vs {other_dataset.use_factorized_anchor_target}"
            )

        if self.anchor_indice_name != other_dataset.anchor_indice_name:
            raise ValueError(
                "cannot merge datasets with different anchor_indice_name settings: "
                f"{self.anchor_indice_name} vs {other_dataset.anchor_indice_name}"
            )

        if self.anchor_score_name != other_dataset.anchor_score_name:
            raise ValueError(
                "cannot merge datasets with different anchor_score_name settings: "
                f"{self.anchor_score_name} vs {other_dataset.anchor_score_name}"
            )

        if self.factorized_anchor_target_name != other_dataset.factorized_anchor_target_name:
            raise ValueError(
                "cannot merge datasets with different factorized_anchor_target_name settings: "
                f"{self.factorized_anchor_target_name} vs {other_dataset.factorized_anchor_target_name}"
            )

        if self.use_rollout_teacher_metadata != other_dataset.use_rollout_teacher_metadata:
            raise ValueError(
                "cannot merge datasets with different use_rollout_teacher_metadata settings: "
                f"{self.use_rollout_teacher_metadata} vs {other_dataset.use_rollout_teacher_metadata}"
            )

        if self.rollout_teacher_metadata_name != other_dataset.rollout_teacher_metadata_name:
            raise ValueError(
                "cannot merge datasets with different rollout_teacher_metadata_name settings: "
                f"{self.rollout_teacher_metadata_name} vs {other_dataset.rollout_teacher_metadata_name}"
            )

        if self.use_replay_planner_targets != other_dataset.use_replay_planner_targets:
            raise ValueError(
                "cannot merge datasets with different use_replay_planner_targets settings: "
                f"{self.use_replay_planner_targets} vs {other_dataset.use_replay_planner_targets}"
            )

        if self.replay_planner_targets_name != other_dataset.replay_planner_targets_name:
            raise ValueError(
                "cannot merge datasets with different replay_planner_targets_name settings: "
                f"{self.replay_planner_targets_name} vs {other_dataset.replay_planner_targets_name}"
            )

        if (self.future_sampling is None) != (other_dataset.future_sampling is None):
            raise ValueError("cannot merge datasets with inconsistent future_sampling presence")

        if self.future_sampling is not None and other_dataset.future_sampling is not None:
            if (
                self.future_sampling.time_horizon != other_dataset.future_sampling.time_horizon
                or self.future_sampling.interval_length != other_dataset.future_sampling.interval_length
                or self.future_sampling.num_poses != other_dataset.future_sampling.num_poses
            ):
                raise ValueError(
                    "cannot merge datasets with different future_sampling settings: "
                    f"{self.future_sampling} vs {other_dataset.future_sampling}"
                )

        duplicate_tokens = set(self._valid_cache_paths).intersection(other_dataset._valid_cache_paths)
        if duplicate_tokens:
            duplicate_preview = sorted(duplicate_tokens)[:10]
            raise ValueError(
                f"cannot merge datasets with duplicate tokens: {duplicate_preview} "
                f"(total={len(duplicate_tokens)})"
            )

        self._valid_cache_paths.update(other_dataset._valid_cache_paths)
        self._sample_meta.update(other_dataset._sample_meta)

        if self._max_samples is not None:
            items = list(self._valid_cache_paths.items())[:self._max_samples]
            self._valid_cache_paths = dict(items)
            self._sample_meta = {
                token: self._sample_meta[token]
                for token in self._valid_cache_paths.keys()
                if token in self._sample_meta
            }

        self.tokens = list(self._valid_cache_paths.keys())
