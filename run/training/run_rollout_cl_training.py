from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from datetime import timedelta
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import hydra
import lightning.pytorch as pl
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from lpl_planner.training.dataset.dataset_utils import FeatureCollate
from lpl_planner.training.dataset.mv_dataset import MVDataset
from lpl_planner.training.lightning_module.mvcl_lightning_module import MVCLLightningModule
from lpl_planner.utils.default_paths import configure_default_paths

configure_default_paths()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

torch.set_float32_matmul_precision("high")

lpl_planner_spec = importlib.util.find_spec("lpl_planner")
lpl_planner_dir = os.path.dirname(lpl_planner_spec.origin)

CONFIG_PATH = os.path.join(lpl_planner_dir, "config/training")
CONFIG_NAME = "custom_rollout_cl_training"


@dataclass(frozen=True)
class CacheRef:
    dataset_id: int
    token: str

    @property
    def key(self) -> str:
        return f"{self.dataset_id}:{self.token}"


class MultiCacheTokenDataset(Dataset):
    def __init__(self, datasets: Sequence[MVDataset], refs: Sequence[CacheRef], name: str) -> None:
        self.datasets = list(datasets)
        self.refs = list(refs)
        self.name = name

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, idx: int):
        ref = self.refs[idx]
        return self.datasets[ref.dataset_id]._load_scene_with_token(ref.token)


def _safe_task_name(task_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_name)).strip("_")
    return sanitized or "task"


def _find_resume_ckpt(ckpt_dir: str) -> Optional[str]:
    last = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last):
        return last
    candidates = glob(os.path.join(ckpt_dir, "*.ckpt"))
    return max(candidates, key=os.path.getmtime) if candidates else None


def _save_manifest(manifest_path: str, manifest: Dict[str, Any]) -> None:
    OmegaConf.save(config=OmegaConf.create(manifest), f=manifest_path)


def _load_manifest(manifest_path: str) -> Dict[str, Any]:
    loaded = OmegaConf.load(manifest_path)
    return OmegaConf.to_container(loaded, resolve=True)


def _normalize_start_task_index(value: Any) -> Optional[int]:
    if value in {None, "", "None", "null", "auto"}:
        return None
    return max(int(value), 0)


def _refs_from_manifest_task(task_entry: Dict[str, Any]) -> List[CacheRef]:
    return [
        CacheRef(int(item["dataset_id"]), str(item["token"]))
        for item in (task_entry.get("memory_refs", []) or [])
    ]


def _validate_refs_exist(datasets: Sequence[MVDataset], refs: Sequence[CacheRef], label: str) -> None:
    missing: List[str] = []
    for ref in refs:
        if ref.dataset_id < 0 or ref.dataset_id >= len(datasets):
            missing.append(ref.key)
            continue
        if ref.token not in datasets[ref.dataset_id]._valid_cache_paths:
            missing.append(ref.key)
    if missing:
        preview = missing[:10]
        raise ValueError(
            f"{label} contains refs that are not available in rollout.cache_roots: {preview} "
            f"(total={len(missing)}). When bootstrapping memory, include previous rollout cache roots in the same order."
        )


def _build_tensorboard_logger(cfg: DictConfig) -> TensorBoardLogger:
    tb_log_dir = os.path.abspath(os.path.join("results", "tb_logs"))
    os.makedirs(tb_log_dir, exist_ok=True)
    return TensorBoardLogger(save_dir=tb_log_dir, name=cfg.job_name, default_hp_metric=False)


def _use_ddp_strategy(trainer_params: DictConfig) -> bool:
    devices = getattr(trainer_params, "devices", 1)
    accelerator = str(getattr(trainer_params, "accelerator", "auto")).lower()
    if accelerator == "cpu":
        return False
    if devices in {None, "auto"}:
        return False
    try:
        return int(devices) != 1
    except (TypeError, ValueError):
        return True


def _build_trainer(cfg: DictConfig, task_ckpt_dir: str, disable_checkpointing: bool, tb_logger: TensorBoardLogger) -> pl.Trainer:
    callbacks = []
    if not disable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                dirpath=task_ckpt_dir,
                filename="last",
                save_last=True,
                monitor=None,
                save_top_k=0,
            )
        )

    trainer_kwargs = dict(OmegaConf.to_container(cfg.lightning.trainer.params, resolve=True))
    if int(trainer_kwargs.get("accumulate_grad_batches", 1) or 1) != 1:
        logger.warning("MVCL uses manual optimization; overriding accumulate_grad_batches=%s to 1", trainer_kwargs["accumulate_grad_batches"])
        trainer_kwargs["accumulate_grad_batches"] = 1
    if disable_checkpointing:
        trainer_kwargs["enable_checkpointing"] = False

    strategy = "auto"
    if _use_ddp_strategy(cfg.lightning.trainer.params):
        strategy = DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=False,
            timeout=timedelta(seconds=int(getattr(cfg, "distributed_timeout_seconds", 3600))),
        )

    return pl.Trainer(strategy=strategy, callbacks=callbacks, logger=tb_logger, **trainer_kwargs)


def _build_dataloader(dataset, loader_cfg: DictConfig, shuffle: bool, drop_last: bool):
    return DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=FeatureCollate(),
        **loader_cfg,
    )


def _load_model_weights_from_checkpoint(model, ckpt_path: Path, label: str) -> None:
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    raw_state = ckpt.get("state_dict", ckpt)
    state_dict = {str(k).replace("model.", "", 1): v for k, v in raw_state.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info("%s load: missing=%s unexpected=%s from %s", label, len(missing), len(unexpected), ckpt_path)


def _normalize_method(method: str) -> str:
    aliases = {
        "ft": "ft",
        "finetune": "ft",
        "replay": "replay",
        "er": "replay",
        "derpp": "derpp",
        "der++": "derpp",
        "joint": "joint",
    }
    normalized = str(method or "derpp").strip().lower().replace("-", "_")
    if normalized not in aliases:
        raise ValueError(f"Unsupported rollout CL method: {method}")
    return aliases[normalized]


def _module_method(method: str) -> str:
    if method == "ft":
        return "ft"
    if method == "replay":
        return "er"
    if method == "derpp":
        return "derpp"
    if method == "joint":
        return "ft"
    raise ValueError(f"Unsupported method: {method}")


def _resolve_rollout_cache_roots(cfg: DictConfig) -> List[Path]:
    explicit_roots = [Path(str(path)).expanduser() for path in (getattr(cfg.rollout, "cache_roots", []) or [])]
    glob_pattern = getattr(cfg.rollout, "cache_root_glob", None)
    glob_roots = []
    if glob_pattern not in {None, "", "None"}:
        glob_roots = [Path(path).expanduser() for path in sorted(glob(str(glob_pattern)))]

    roots = []
    seen = set()
    for root in explicit_roots + glob_roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_dir():
            raise FileNotFoundError(f"Rollout CL cache root does not exist: {resolved}")
        roots.append(resolved)
    if not roots:
        raise ValueError("No rollout CL cache roots were provided. Set rollout.cache_roots or rollout.cache_root_glob.")
    return roots


def _dataset_kwargs(cfg: DictConfig, cache_path: Path, replay_targets: bool) -> Dict[str, Any]:
    return {
        "cache_path": str(cache_path),
        "future_sampling": instantiate(cfg.model.future_sampling),
        "expand_iteration": bool(getattr(cfg, "expand_iteration", False)),
        "max_samples": getattr(cfg, "max_samples", None),
        "use_anchor_indice": bool(getattr(cfg, "use_anchor_indice", True)),
        "use_anchor_score": bool(getattr(cfg, "use_anchor_score", True)),
        "anchor_indice_name": getattr(cfg, "anchor_indice_name", "anchor_indice.gz"),
        "anchor_score_name": getattr(cfg, "anchor_score_name", "anchor_scores.gz"),
        "use_replay_planner_targets": bool(replay_targets),
        "replay_planner_targets_name": getattr(cfg, "replay_target_name", "replay_planner_targets.gz"),
    }


def _expert_dataset_kwargs(cfg: DictConfig, cache_path: Path) -> Dict[str, Any]:
    kwargs = _dataset_kwargs(cfg, cache_path, replay_targets=False)
    expert_cfg = getattr(cfg, "expert_mix", OmegaConf.create({}))
    kwargs["expand_iteration"] = bool(getattr(expert_cfg, "expand_iteration", True))
    kwargs["max_samples"] = getattr(expert_cfg, "max_cache_samples", kwargs.get("max_samples", None))
    return kwargs


def _build_rollout_task_datasets(cfg: DictConfig, roots: Sequence[Path]) -> Tuple[List[MVDataset], List[List[CacheRef]], List[Dict[str, Any]]]:
    datasets: List[MVDataset] = []
    task_refs: List[List[CacheRef]] = []
    task_infos: List[Dict[str, Any]] = []
    for dataset_id, root in enumerate(roots):
        dataset = MVDataset(log_names=None, **_dataset_kwargs(cfg, root, replay_targets=True))
        if len(dataset) <= 0:
            raise ValueError(f"Rollout CL cache root has no valid samples: {root}")
        datasets.append(dataset)
        refs = [CacheRef(dataset_id=dataset_id, token=token) for token in dataset.tokens]
        task_name = root.name
        task_refs.append(refs)
        task_infos.append({"task_id": dataset_id, "task_name": task_name, "cache_path": str(root), "sample_count": len(refs)})
        logger.info("Rollout task %02d | name=%s | samples=%s | cache=%s", dataset_id, task_name, len(refs), root)
    return datasets, task_refs, task_infos


def _load_rollout_frame_metadata(cache_root: Path, summary_name: str) -> Dict[str, Dict[str, Any]]:
    summary_path = cache_root / summary_name
    if not summary_path.is_file():
        logger.warning("Rollout summary not found for memory scoring: %s", summary_path)
        return {}
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    metadata_by_token: Dict[str, Dict[str, Any]] = {}
    for result in summary.get("frame_results", []) or []:
        if not isinstance(result, dict) or int(result.get("kept", 0) or 0) <= 0:
            continue
        sample_dir = result.get("sample_dir")
        if not sample_dir:
            continue
        token = Path(str(sample_dir)).name
        metadata_by_token[token] = result
    return metadata_by_token


def _primary_candidate_reason(reasons: Sequence[Any]) -> str:
    priority = ["failure_window", "high_risk_context", "model_expert_disagreement"]
    reason_set = {str(reason) for reason in reasons}
    for reason in priority:
        if reason in reason_set:
            return reason
    return sorted(reason_set)[0] if reason_set else "other"


def _minmax_normalize_score_map(raw_scores: Dict[str, float]) -> Dict[str, float]:
    if not raw_scores:
        return {}
    values = [float(value) for value in raw_scores.values() if math.isfinite(float(value))]
    if not values:
        return {key: 0.0 for key in raw_scores}
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-8:
        return {key: 1.0 for key in raw_scores}
    return {
        key: (float(value) - lo) / (hi - lo) if math.isfinite(float(value)) else 0.0
        for key, value in raw_scores.items()
    }


def _build_rollout_priority_context(
    cfg: DictConfig,
    datasets: Sequence[MVDataset],
    roots: Sequence[Path],
) -> Tuple[Dict[str, float], Dict[str, str]]:
    memory_cfg = getattr(cfg.continual, "memory", OmegaConf.create({}))
    priority_cfg = getattr(memory_cfg, "rollout_priority", OmegaConf.create({}))
    if not bool(getattr(priority_cfg, "enabled", True)):
        return {}, {}

    summary_name = str(getattr(priority_cfg, "summary_name", "rollout_cl_generation_summary.json"))
    reason_weights = dict(OmegaConf.to_container(getattr(priority_cfg, "reason_weights", OmegaConf.create({})), resolve=True) or {})
    state_weights = dict(OmegaConf.to_container(getattr(priority_cfg, "state_class_weights", OmegaConf.create({})), resolve=True) or {})
    ttc_weight = float(getattr(priority_cfg, "ttc_weight", 0.4))
    ttc_threshold = max(float(getattr(priority_cfg, "ttc_threshold_s", 1.0)), 1e-6)
    low_score_weight = float(getattr(priority_cfg, "low_best_score_weight", 0.3))
    best_score_floor = float(getattr(priority_cfg, "best_score_floor", 0.15))
    base_weight = float(getattr(priority_cfg, "base_weight", 0.0))

    priority_scores: Dict[str, float] = {}
    partition_map: Dict[str, str] = {}
    for dataset_id, (dataset, root) in enumerate(zip(datasets, roots)):
        metadata_by_token = _load_rollout_frame_metadata(root, summary_name)
        if not metadata_by_token:
            continue
        for token in dataset.tokens:
            metadata = metadata_by_token.get(token)
            if metadata is None:
                continue
            ref = CacheRef(dataset_id=dataset_id, token=token)
            meta = dataset._sample_meta[token]
            state_class = str(metadata.get("state_class", "unknown"))
            primary_reason = _primary_candidate_reason(metadata.get("oracle_candidate_reasons", []) or [])
            partition_map[ref.key] = f"{meta.scene_type or '__unknown__'}|{state_class}|{primary_reason}"

            score = base_weight
            for reason in metadata.get("oracle_candidate_reasons", []) or []:
                score += float(reason_weights.get(str(reason), 0.0))
            score += float(state_weights.get(state_class, 0.0))

            min_ttc = metadata.get("min_ttc")
            if isinstance(min_ttc, (int, float)) and math.isfinite(float(min_ttc)):
                score += ttc_weight * max(0.0, 1.0 - float(min_ttc) / ttc_threshold)

            best_score = metadata.get("best_score")
            if isinstance(best_score, (int, float)) and math.isfinite(float(best_score)):
                score += low_score_weight * max(0.0, 1.0 - float(best_score) / max(best_score_floor, 1e-6))

            priority_scores[ref.key] = float(score)
    logger.info("Loaded rollout priority scores for %s memory candidates", len(priority_scores))
    return priority_scores, partition_map


def _split_refs(refs: Sequence[CacheRef], eval_ratio: float, seed: int, max_train: Optional[int], max_eval: Optional[int]) -> Tuple[List[CacheRef], List[CacheRef]]:
    ref_list = list(refs)
    rng = random.Random(seed)
    rng.shuffle(ref_list)
    if len(ref_list) <= 1 or eval_ratio <= 0.0:
        train_refs, eval_refs = ref_list, []
    else:
        eval_count = int(round(len(ref_list) * float(eval_ratio)))
        eval_count = min(max(eval_count, 1), len(ref_list) - 1)
        eval_refs = ref_list[:eval_count]
        train_refs = ref_list[eval_count:]
    if max_train not in {None, 0, "0", "None", ""}:
        train_refs = train_refs[: int(max_train)]
    if max_eval not in {None, 0, "0", "None", ""}:
        eval_refs = eval_refs[: int(max_eval)]
    return train_refs, eval_refs


def _ref_meta(datasets: Sequence[MVDataset], ref: CacheRef):
    return datasets[ref.dataset_id]._sample_meta[ref.token]


def _base_token_from_rollout_token(token: str) -> str:
    return re.sub(r"_iter_\d+$", "", str(token))


def _optional_limit(value: Any, fallback: int) -> int:
    if value in {None, 0, "0", "None", "null", ""}:
        return int(fallback)
    return max(int(value), 0)


def _train_drop_last(cfg: DictConfig) -> bool:
    return bool(getattr(cfg.dataloader, "drop_last", False))


def _build_expert_dataset_and_lookup(cfg: DictConfig) -> Tuple[List[MVDataset], Dict[Tuple[str, Optional[str], str], str]]:
    expert_cfg = getattr(cfg, "expert_mix", OmegaConf.create({}))
    if not bool(getattr(expert_cfg, "enabled", False)):
        return [], {}
    cache_path = getattr(expert_cfg, "cache_path", None)
    if cache_path in {None, "", "None"}:
        raise ValueError("expert_mix.enabled=true requires expert_mix.cache_path to be set.")
    cache_root = Path(str(cache_path)).expanduser()
    if not cache_root.exists():
        raise FileNotFoundError(f"expert_mix.cache_path does not exist: {cache_root}")

    expert_dataset = MVDataset(
        log_names=None,
        **_expert_dataset_kwargs(cfg, cache_root),
    )
    expert_lookup: Dict[Tuple[str, Optional[str], str], str] = {}
    for token in expert_dataset.tokens:
        meta = expert_dataset._sample_meta[token]
        expert_lookup[(meta.log_name, meta.scene_type, token)] = token

    logger.info("Loaded expert mix cache: samples=%s cache=%s", len(expert_dataset), cache_path)
    return [expert_dataset], expert_lookup


def _build_expert_refs(
    cfg: DictConfig,
    rollout_datasets: Sequence[MVDataset],
    expert_lookup: Dict[Tuple[str, Optional[str], str], str],
    current_refs: Sequence[CacheRef],
) -> List[CacheRef]:
    expert_cfg = getattr(cfg, "expert_mix", OmegaConf.create({}))
    ratio = float(getattr(expert_cfg, "ratio", 0.0))
    if ratio <= 0.0 or not expert_lookup:
        return []
    sample_basis = str(getattr(expert_cfg, "sample_basis", "max")).strip().lower()
    if sample_basis not in {"current_rollout", "rollout", "expert_pool", "expert", "max"}:
        raise ValueError(f"Unsupported expert_mix.sample_basis: {sample_basis}")

    matched_tokens: List[str] = []
    seen = set()
    for ref in current_refs:
        meta = _ref_meta(rollout_datasets, ref)
        key = (meta.log_name, meta.scene_type, _base_token_from_rollout_token(meta.scenario_token))
        token = expert_lookup.get(key)
        if token is None or token in seen:
            continue
        seen.add(token)
        matched_tokens.append(token)

    if not matched_tokens:
        logger.warning("expert_mix enabled but no matching original expert samples were found")
        return []

    rollout_based = int(math.ceil(len(current_refs) * ratio))
    expert_pool_based = int(math.ceil(len(matched_tokens) * ratio))
    if sample_basis in {"current_rollout", "rollout"}:
        keep = rollout_based
    elif sample_basis in {"expert_pool", "expert"}:
        keep = expert_pool_based
    else:
        keep = max(rollout_based, expert_pool_based)

    min_samples = getattr(expert_cfg, "min_samples_per_task", None)
    max_samples = getattr(expert_cfg, "max_samples_per_task", None)
    if min_samples not in {None, 0, "0", "None", "null", ""}:
        keep = max(keep, int(min_samples))
    if max_samples not in {None, 0, "0", "None", "null", ""}:
        keep = min(keep, int(max_samples))
    keep = min(max(keep, 1), len(matched_tokens))

    rng = random.Random(int(getattr(expert_cfg, "seed", 0)))
    rng.shuffle(matched_tokens)
    refs = [CacheRef(dataset_id=0, token=token) for token in matched_tokens[:keep]]
    logger.info(
        "Expert mix matched %s samples, using %s (basis=%s ratio=%s rollout_based=%s expert_pool_based=%s)",
        len(matched_tokens),
        len(refs),
        sample_basis,
        ratio,
        rollout_based,
        expert_pool_based,
    )
    return refs


def _partition_key(
    datasets: Sequence[MVDataset],
    ref: CacheRef,
    strategy: str,
    rollout_partition_map: Optional[Dict[str, str]] = None,
) -> str:
    strategy = str(strategy or "scene_type").strip().lower()
    if strategy in {"task", "dataset", "round"}:
        return f"task_{ref.dataset_id}"
    if strategy in {"rollout_bucket", "sar", "scenario_aware"}:
        if rollout_partition_map is not None and ref.key in rollout_partition_map:
            return rollout_partition_map[ref.key]
        meta = _ref_meta(datasets, ref)
        return meta.scene_type or "__unknown__"
    if strategy in {"none", "global"}:
        return "__global__"
    meta = _ref_meta(datasets, ref)
    return meta.scene_type or "__unknown__"


def _update_memory(
    datasets: Sequence[MVDataset],
    old_memory: Sequence[CacheRef],
    current_refs: Sequence[CacheRef],
    old_scores: Dict[str, float],
    importance_scores: Dict[str, float],
    rollout_priority_scores: Dict[str, float],
    rollout_partition_map: Dict[str, str],
    cfg: DictConfig,
) -> Tuple[List[CacheRef], Dict[str, float]]:
    memory_cfg = getattr(cfg.continual, "memory", OmegaConf.create({}))
    capacity = int(getattr(memory_cfg, "capacity", 0))
    if capacity <= 0:
        return [], {}

    merged: List[CacheRef] = []
    seen = set()
    for ref in list(old_memory) + list(current_refs):
        if ref.key in seen:
            continue
        seen.add(ref.key)
        merged.append(ref)

    partition_strategy = str(getattr(memory_cfg, "partition_strategy", "scene_type"))
    priority_cfg = getattr(memory_cfg, "rollout_priority", OmegaConf.create({}))
    train_loss_weight = float(getattr(priority_cfg, "train_loss_weight", 1.0))
    rollout_score_weight = float(getattr(priority_cfg, "rollout_score_weight", 1.0))
    normalize_components = bool(getattr(priority_cfg, "normalize_components", True))

    train_raw = {
        ref.key: float(importance_scores.get(ref.token, old_scores.get(ref.key, 0.0)))
        for ref in merged
    }
    rollout_raw = {
        ref.key: float(rollout_priority_scores.get(ref.key, 0.0))
        for ref in merged
    }
    train_component = _minmax_normalize_score_map(train_raw) if normalize_components else train_raw
    rollout_component = _minmax_normalize_score_map(rollout_raw) if normalize_components else rollout_raw

    def _memory_score(ref: CacheRef) -> float:
        train_loss_score = float(train_component.get(ref.key, 0.0))
        rollout_score = float(rollout_component.get(ref.key, 0.0))
        return train_loss_weight * train_loss_score + rollout_score_weight * rollout_score

    if len(merged) <= capacity:
        scores = {ref.key: _memory_score(ref) for ref in merged}
        return merged, scores

    partitions: Dict[str, List[CacheRef]] = {}
    for ref in merged:
        partitions.setdefault(_partition_key(datasets, ref, partition_strategy, rollout_partition_map), []).append(ref)

    selected: List[CacheRef] = []
    selected_scores: Dict[str, float] = {}
    active = [key for key, refs in partitions.items() if refs]
    base_quota = max(1, capacity // max(len(active), 1))
    remainder = max(capacity - base_quota * len(active), 0)

    for part_idx, key in enumerate(sorted(active)):
        refs = partitions[key]
        quota = min(len(refs), base_quota + (1 if part_idx < remainder else 0))
        scored = []
        for ref in refs:
            score = _memory_score(ref)
            scored.append((score, ref))
        scored.sort(key=lambda item: (item[0], item[1].key), reverse=True)
        for score, ref in scored[:quota]:
            selected.append(ref)
            selected_scores[ref.key] = score

    if len(selected) > capacity:
        selected = sorted(selected, key=lambda ref: selected_scores.get(ref.key, 0.0), reverse=True)[:capacity]
        selected_scores = {ref.key: selected_scores.get(ref.key, 0.0) for ref in selected}
    return selected, selected_scores


def _build_val_loaders(
    datasets: Sequence[MVDataset],
    task_eval_refs: Sequence[CacheRef],
    history_eval_refs: Sequence[CacheRef],
    memory_refs: Sequence[CacheRef],
    cfg: DictConfig,
) -> Tuple[Any, List[str]]:
    loaders = []
    names = []
    for name, refs in [
        ("current_eval", task_eval_refs),
        ("history_eval", history_eval_refs),
        ("memory_eval", memory_refs),
    ]:
        if not refs:
            continue
        dataset = MultiCacheTokenDataset(datasets, refs, name=name)
        loaders.append(_build_dataloader(dataset, cfg.val_dataloader.params, shuffle=False, drop_last=False))
        names.append(name)
    if not loaders:
        raise ValueError("No rollout CL validation loaders are available")
    return (loaders[0] if len(loaders) == 1 else loaders), names


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    method = _normalize_method(str(getattr(cfg.continual, "method", "derpp")))
    rollout_roots = _resolve_rollout_cache_roots(cfg)
    rollout_datasets, all_task_refs, task_infos = _build_rollout_task_datasets(cfg, rollout_roots)
    rollout_priority_scores, rollout_partition_map = _build_rollout_priority_context(cfg, rollout_datasets, rollout_roots)

    model = instantiate(cfg.model)
    start_model_path = getattr(cfg, "start_model_path", None)
    if start_model_path not in {None, "", "None"}:
        _load_model_weights_from_checkpoint(model, Path(str(start_model_path)).expanduser(), "Start checkpoint")

    module = MVCLLightningModule(
        model,
        cfg=cfg,
        learning_rate=float(cfg.learning_rate),
        warmup_steps=int(cfg.warmup_steps),
        check_invalid_grad=bool(cfg.check_invalid_grad),
        continual_method=_module_method(method),
        replay_loss_weight=float(getattr(cfg.continual, "replay_loss_weight", 1.0)),
        derpp_distill_weight=float(getattr(cfg.continual, "derpp_distill_weight", 0.25)),
        derpp_temperature=float(getattr(cfg.continual, "derpp_temperature", 1.0)),
        derpp_rank_margin=float(getattr(cfg.continual, "derpp_rank_margin", 0.5)),
        prediction_distill_weight=float(getattr(cfg.continual, "prediction_distill_weight", 0.0)),
        distill_source=str(getattr(cfg.continual, "distill_source", "stored_logits")),
        expert_loss_weight=float(getattr(cfg.expert_mix, "loss_weight", 0.0)) if bool(getattr(cfg.expert_mix, "enabled", False)) else 0.0,
        mas_enabled=bool(getattr(cfg.continual, "mas_enabled", False)),
        mas_lambda=float(getattr(cfg.continual, "mas_lambda", 0.0)),
        mas_update_alpha=float(getattr(cfg.continual, "mas_update_alpha", 0.5)),
        mas_max_batches=int(getattr(cfg.continual, "mas_max_batches", 32)),
    )

    checkpoint_root = Path(str(getattr(cfg, "checkpoint_root", "results/checkpoints"))).expanduser()
    base_ckpt_dir = str(checkpoint_root / str(cfg.job_name))
    os.makedirs(base_ckpt_dir, exist_ok=True)
    tb_logger = _build_tensorboard_logger(cfg)
    manifest_path = os.path.join(base_ckpt_dir, "rollout_continual_manifest.yaml")
    manifest = {
        "job_name": str(cfg.job_name),
        "method": method,
        "start_model_path": None if start_model_path in {None, "", "None"} else str(start_model_path),
        "rollout_tasks": task_infos,
        "memory_capacity": int(getattr(cfg.continual.memory, "capacity", 0)),
        "expert_mix": OmegaConf.to_container(cfg.expert_mix, resolve=True),
        "tasks": [],
    }

    memory_refs: List[CacheRef] = []
    memory_scores: Dict[str, float] = {}
    start_task_index = 0
    restored_from_current_manifest = False
    if bool(getattr(cfg.continual, "resume", True)) and os.path.exists(manifest_path):
        loaded = _load_manifest(manifest_path)
        if loaded.get("rollout_tasks") == task_infos:
            manifest.update(loaded)
            completed = manifest.get("tasks", []) or []
            start_task_index = len(completed)
            if completed:
                last_ckpt = completed[-1].get("final_checkpoint")
                if last_ckpt and os.path.exists(last_ckpt):
                    _load_model_weights_from_checkpoint(module.model, Path(last_ckpt), "Resume rollout CL")
                memory_refs = _refs_from_manifest_task(completed[-1])
                memory_scores = {str(key): float(value) for key, value in (completed[-1].get("memory_scores") or {}).items()}
            restored_from_current_manifest = True
        else:
            logger.warning("Ignoring incompatible rollout CL manifest: %s", manifest_path)

    requested_start_task_index = _normalize_start_task_index(getattr(cfg.continual, "start_task_index", None))
    bootstrap_manifest_path = getattr(cfg.continual.memory, "bootstrap_manifest", None)
    if not restored_from_current_manifest:
        if requested_start_task_index is not None:
            start_task_index = min(requested_start_task_index, len(all_task_refs))
        if bootstrap_manifest_path not in {None, "", "None", "null"}:
            bootstrap_manifest_path = str(Path(str(bootstrap_manifest_path)).expanduser())
            bootstrap_manifest = _load_manifest(bootstrap_manifest_path)
            bootstrap_tasks = bootstrap_manifest.get("tasks", []) or []
            if bootstrap_tasks:
                last_bootstrap_task = bootstrap_tasks[-1]
                memory_refs = _refs_from_manifest_task(last_bootstrap_task)
                memory_scores = {str(key): float(value) for key, value in (last_bootstrap_task.get("memory_scores") or {}).items()}
                if start_task_index == 0:
                    start_task_index = min(len(bootstrap_tasks), len(all_task_refs))
                manifest["tasks"] = list(bootstrap_tasks[:start_task_index])
                logger.info(
                    "Bootstrapped %s memory refs and %s prior task entries from %s; start_task_index=%s",
                    len(memory_refs),
                    len(manifest["tasks"]),
                    bootstrap_manifest_path,
                    start_task_index,
                )

    eval_ratio = float(getattr(cfg.continual, "eval_ratio", 0.1))
    max_train = getattr(cfg.continual, "max_samples_per_task", None)
    max_eval = getattr(cfg.continual, "current_eval_max_samples", None)
    train_eval_splits = [
        _split_refs(refs, eval_ratio=eval_ratio, seed=int(getattr(cfg.continual, "eval_seed", 0)) + task_idx, max_train=max_train, max_eval=max_eval)
        for task_idx, refs in enumerate(all_task_refs)
    ]
    if memory_refs:
        _validate_refs_exist(rollout_datasets, memory_refs, label="rollout CL memory")
    history_eval_refs: List[CacheRef] = []
    for prior_task_index in range(start_task_index):
        history_eval_refs.extend(train_eval_splits[prior_task_index][1])

    expert_datasets, expert_lookup = _build_expert_dataset_and_lookup(cfg)

    if method == "joint":
        joint_train_refs = [ref for train_refs, _ in train_eval_splits for ref in train_refs]
        joint_eval_refs = [ref for _, eval_refs in train_eval_splits for ref in eval_refs]
        module.set_task_context("joint_rollout", -1)
        module.set_validation_sources(["joint_eval"])
        train_loader = _build_dataloader(MultiCacheTokenDataset(rollout_datasets, joint_train_refs, "joint_train"), cfg.dataloader.params, shuffle=True, drop_last=_train_drop_last(cfg))
        val_loader = _build_dataloader(MultiCacheTokenDataset(rollout_datasets, joint_eval_refs, "joint_eval"), cfg.val_dataloader.params, shuffle=False, drop_last=False)
        task_ckpt_dir = os.path.join(base_ckpt_dir, "joint_rollout")
        os.makedirs(task_ckpt_dir, exist_ok=True)
        trainer = _build_trainer(cfg, task_ckpt_dir, bool(getattr(cfg, "disable_checkpointing", False)), tb_logger)
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=_find_resume_ckpt(task_ckpt_dir))
        final_ckpt = os.path.join(task_ckpt_dir, "last.ckpt")
        trainer.save_checkpoint(final_ckpt)
        manifest["tasks"] = [{"task_id": -1, "task_name": "joint_rollout", "train_size": len(joint_train_refs), "eval_size": len(joint_eval_refs), "final_checkpoint": final_ckpt}]
        _save_manifest(manifest_path, manifest)
        return

    for task_index in range(start_task_index, len(all_task_refs)):
        task_name = task_infos[task_index]["task_name"]
        train_refs, eval_refs = train_eval_splits[task_index]
        module.set_task_context(task_name, task_index)

        task_ckpt_dir = os.path.join(base_ckpt_dir, f"task_{task_index:02d}_{_safe_task_name(task_name)}")
        os.makedirs(task_ckpt_dir, exist_ok=True)
        trainer = _build_trainer(cfg, task_ckpt_dir, bool(getattr(cfg, "disable_checkpointing", False)), tb_logger)

        current_loader = _build_dataloader(MultiCacheTokenDataset(rollout_datasets, train_refs, f"{task_name}_train"), cfg.dataloader.params, shuffle=True, drop_last=_train_drop_last(cfg))
        loader_dict: Dict[str, Any] = {"current": current_loader}
        if method in {"replay", "derpp"} and memory_refs:
            memory_loader = _build_dataloader(MultiCacheTokenDataset(rollout_datasets, memory_refs, "memory"), cfg.dataloader.params, shuffle=True, drop_last=False)
            loader_dict["memory"] = memory_loader

        expert_refs = _build_expert_refs(cfg, rollout_datasets, expert_lookup, train_refs)
        if expert_datasets and expert_refs:
            expert_loader = _build_dataloader(MultiCacheTokenDataset(expert_datasets, expert_refs, "expert_mix"), cfg.dataloader.params, shuffle=True, drop_last=False)
            loader_dict["expert"] = expert_loader

        train_loader = CombinedLoader(loader_dict, mode="max_size_cycle") if len(loader_dict) > 1 else current_loader
        val_loaders, val_names = _build_val_loaders(
            rollout_datasets,
            task_eval_refs=eval_refs,
            history_eval_refs=history_eval_refs[: _optional_limit(getattr(cfg.continual, "history_eval_max_samples", None), len(history_eval_refs))],
            memory_refs=memory_refs[: _optional_limit(getattr(cfg.continual, "memory_eval_max_samples", None), len(memory_refs))],
            cfg=cfg,
        )
        module.set_validation_sources(val_names)

        logger.info(
            "Starting rollout CL task %s/%s: %s train=%s eval=%s memory=%s expert=%s",
            task_index,
            len(all_task_refs),
            task_name,
            len(train_refs),
            len(eval_refs),
            len(memory_refs),
            len(expert_refs),
        )
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loaders, ckpt_path=_find_resume_ckpt(task_ckpt_dir))
        final_ckpt = os.path.join(task_ckpt_dir, "last.ckpt")
        if not os.path.exists(final_ckpt):
            trainer.save_checkpoint(final_ckpt)

        task_stats = module.pop_task_statistics()
        if bool(getattr(cfg.continual, "mas_enabled", False)):
            mas_loader = _build_dataloader(MultiCacheTokenDataset(rollout_datasets, train_refs, f"{task_name}_mas"), cfg.val_dataloader.params, shuffle=False, drop_last=False)
            module.update_mas_state(mas_loader, max_batches=int(getattr(cfg.continual, "mas_max_batches", 32)))

        memory_refs, memory_scores = _update_memory(
            datasets=rollout_datasets,
            old_memory=memory_refs,
            current_refs=train_refs,
            old_scores=memory_scores,
            importance_scores=task_stats.get("importance", {}),
            rollout_priority_scores=rollout_priority_scores,
            rollout_partition_map=rollout_partition_map,
            cfg=cfg,
        )
        history_eval_refs.extend(eval_refs)
        manifest["tasks"].append(
            {
                "task_id": task_index,
                "task_name": task_name,
                "cache_path": task_infos[task_index]["cache_path"],
                "train_size": len(train_refs),
                "eval_size": len(eval_refs),
                "history_eval_size": len(history_eval_refs),
                "memory_eval_size": len(memory_refs),
                "expert_mix_size": len(expert_refs),
                "final_checkpoint": final_ckpt,
                "memory_size_after_task": len(memory_refs),
                "memory_refs": [{"dataset_id": ref.dataset_id, "token": ref.token} for ref in memory_refs],
                "memory_scores": memory_scores,
            }
        )
        _save_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
