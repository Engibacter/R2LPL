from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json


DEFAULT_SCENE_TOKENS_JSON_NAME = "selected_scenario_tokens.json"


def resolve_scene_tokens_path(
    cache_root: Path,
    scene_tokens_path: Optional[str] = None,
    scene_tokens_name: str = DEFAULT_SCENE_TOKENS_JSON_NAME,
) -> Path:
    if scene_tokens_path not in {None, "", "None"}:
        return Path(str(scene_tokens_path)).expanduser()
    return Path(cache_root).expanduser() / scene_tokens_name


def load_scene_token_selection(scene_tokens_path: Path) -> Dict[str, Any]:
    scene_tokens_path = Path(scene_tokens_path).expanduser()
    with scene_tokens_path.open("r", encoding="utf-8") as scene_tokens_file:
        payload = json.load(scene_tokens_file)

    if isinstance(payload, list):
        payload = {"scenario_tokens": payload}

    scenario_tokens = sorted({str(token) for token in payload.get("scenario_tokens", [])})
    log_names = sorted({str(log_name) for log_name in payload.get("log_names", [])})

    normalized_payload = dict(payload)
    normalized_payload["scenario_tokens"] = scenario_tokens
    normalized_payload["log_names"] = log_names
    return normalized_payload


def write_scene_token_selection(
    cache_root: Path,
    scenario_tokens: Iterable[str],
    log_names: Optional[Iterable[str]] = None,
    scene_tokens_path: Optional[str] = None,
    scene_tokens_name: str = DEFAULT_SCENE_TOKENS_JSON_NAME,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, int]:
    cache_root = Path(cache_root).expanduser()
    resolved_path = resolve_scene_tokens_path(
        cache_root=cache_root,
        scene_tokens_path=scene_tokens_path,
        scene_tokens_name=scene_tokens_name,
    )
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_tokens = sorted({str(token) for token in scenario_tokens})
    normalized_logs = sorted({str(log_name) for log_name in (log_names or [])})

    payload: Dict[str, Any] = {
        "version": 1,
        "cache_root": str(cache_root),
        "job_name": cache_root.name,
        "scenario_tokens": normalized_tokens,
        "log_names": normalized_logs,
    }
    if extra_metadata:
        payload.update(extra_metadata)

    with resolved_path.open("w", encoding="utf-8") as scene_tokens_file:
        json.dump(payload, scene_tokens_file, ensure_ascii=True, indent=2, sort_keys=True)

    return resolved_path, len(normalized_tokens)