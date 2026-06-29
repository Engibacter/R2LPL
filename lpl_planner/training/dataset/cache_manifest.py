from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import json


DEFAULT_CACHE_MANIFEST_NAME = "selected_manifest.jsonl"
_REQUIRED_CACHE_FILES = (
    "scene_feature.gz",
    "agent_prediction.gz",
    "expert_trajectory.gz",
)


def resolve_cache_manifest_path(
    cache_root: Path,
    manifest_path: Optional[str] = None,
    manifest_name: str = DEFAULT_CACHE_MANIFEST_NAME,
) -> Path:
    if manifest_path not in {None, "", "None"}:
        return Path(str(manifest_path)).expanduser()
    return Path(cache_root).expanduser() / manifest_name


def has_required_cache_files(sample_path: Path) -> bool:
    return all((sample_path / file_name).is_file() for file_name in _REQUIRED_CACHE_FILES)


def iter_cache_entries(cache_root: Path, split_iteration: bool = False) -> Iterator[Dict[str, Any]]:
    cache_root = Path(cache_root).expanduser()
    if not cache_root.is_dir():
        return

    for log_path in sorted(cache_root.iterdir()):
        if not log_path.is_dir():
            continue

        for type_path in sorted(log_path.iterdir()):
            if not type_path.is_dir():
                continue

            for token_path in sorted(type_path.iterdir()):
                if not token_path.is_dir():
                    continue

                if split_iteration:
                    for iteration_path in sorted(token_path.iterdir()):
                        if not iteration_path.is_dir() or not has_required_cache_files(iteration_path):
                            continue
                        yield {
                            "record_type": "entry",
                            "log_name": log_path.name,
                            "scene_type": type_path.name,
                            "scenario_token": token_path.name,
                            "sample_token": f"{token_path.name}_{iteration_path.name}",
                            "iteration": iteration_path.name,
                            "relative_path": iteration_path.relative_to(cache_root).as_posix(),
                        }
                else:
                    if not has_required_cache_files(token_path):
                        continue
                    yield {
                        "record_type": "entry",
                        "log_name": log_path.name,
                        "scene_type": type_path.name,
                        "scenario_token": token_path.name,
                        "sample_token": token_path.name,
                        "iteration": None,
                        "relative_path": token_path.relative_to(cache_root).as_posix(),
                    }


def iter_cache_manifest_entries(manifest_path: Path) -> Iterator[Dict[str, Any]]:
    manifest_path = Path(manifest_path).expanduser()
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        for line in manifest_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("record_type") != "entry":
                continue
            yield record


def build_manifest_entry(
    cache_root: Path,
    log_name: str,
    scene_type: str,
    scenario_token: str,
    iteration: Optional[int] = None,
) -> Dict[str, Any]:
    cache_root = Path(cache_root).expanduser()
    sample_path = cache_root / str(log_name) / str(scene_type) / str(scenario_token)
    sample_token = str(scenario_token)
    iteration_name: Optional[str] = None

    if iteration is not None:
        iteration_name = f"iteration_{int(iteration):04d}"
        sample_path = sample_path / iteration_name
        sample_token = f"{scenario_token}_{iteration_name}"

    return {
        "record_type": "entry",
        "log_name": str(log_name),
        "scene_type": str(scene_type),
        "scenario_token": str(scenario_token),
        "sample_token": sample_token,
        "iteration": iteration_name,
        "relative_path": sample_path.relative_to(cache_root).as_posix(),
    }


def _normalize_manifest_entries(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped_entries: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        if entry.get("record_type") != "entry":
            continue
        relative_path = entry.get("relative_path")
        if not relative_path:
            continue
        normalized_entry = dict(entry)
        normalized_entry["record_type"] = "entry"
        deduped_entries[str(relative_path)] = normalized_entry
    return [deduped_entries[key] for key in sorted(deduped_entries.keys())]


def write_manifest_entries(
    cache_root: Path,
    entries: Iterable[Dict[str, Any]],
    split_iteration: bool = False,
    manifest_path: Optional[str] = None,
    manifest_name: str = DEFAULT_CACHE_MANIFEST_NAME,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, int]:
    cache_root = Path(cache_root).expanduser()
    resolved_manifest_path = resolve_cache_manifest_path(
        cache_root=cache_root,
        manifest_path=manifest_path,
        manifest_name=manifest_name,
    )
    resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_entries = _normalize_manifest_entries(entries)

    with resolved_manifest_path.open("w", encoding="utf-8") as manifest_file:
        meta = {
            "record_type": "meta",
            "version": 2,
            "cache_root": str(cache_root),
            "job_name": cache_root.name,
            "split_iteration": bool(split_iteration),
            "manifest_kind": "selected",
        }
        if extra_metadata:
            meta.update(extra_metadata)
        manifest_file.write(json.dumps(meta, sort_keys=True) + "\n")

        for entry in normalized_entries:
            manifest_file.write(json.dumps(entry, sort_keys=True) + "\n")

    return resolved_manifest_path, len(normalized_entries)


def write_cache_manifest(
    cache_root: Path,
    split_iteration: bool = False,
    manifest_path: Optional[str] = None,
    manifest_name: str = DEFAULT_CACHE_MANIFEST_NAME,
) -> Tuple[Path, int]:
    return write_manifest_entries(
        cache_root=cache_root,
        entries=iter_cache_entries(cache_root=cache_root, split_iteration=split_iteration),
        split_iteration=split_iteration,
        manifest_path=manifest_path,
        manifest_name=manifest_name,
        extra_metadata={"manifest_kind": "inventory"},
    )