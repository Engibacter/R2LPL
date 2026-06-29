from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def configure_default_paths(create_results: bool = True) -> Dict[str, str]:
    """Set R2LPL output/cache defaults without overriding user-provided paths."""

    repo_root = Path(os.environ.get("R2LPL_ROOT", get_repo_root())).expanduser().resolve()
    results_root = Path(os.environ.get("R2LPL_RESULTS_ROOT", repo_root / "results")).expanduser()
    cache_root = Path(os.environ.get("R2LPL_CACHE_ROOT", results_root / "cache")).expanduser()

    os.environ.setdefault("R2LPL_ROOT", str(repo_root))
    os.environ.setdefault("R2LPL_RESULTS_ROOT", str(results_root))
    os.environ.setdefault("R2LPL_CACHE_ROOT", str(cache_root))

    if create_results:
        for path in (
            results_root,
            cache_root,
            results_root / "checkpoints",
            results_root / "planner_anchors",
            results_root / "rollout",
            results_root / "rollout_data",
            results_root / "logs",
        ):
            path.mkdir(parents=True, exist_ok=True)

    return {
        "R2LPL_ROOT": os.environ["R2LPL_ROOT"],
        "R2LPL_RESULTS_ROOT": os.environ["R2LPL_RESULTS_ROOT"],
        "R2LPL_CACHE_ROOT": os.environ["R2LPL_CACHE_ROOT"],
    }
