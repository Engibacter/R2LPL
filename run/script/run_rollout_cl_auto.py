from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lpl_planner.utils.default_paths import configure_default_paths

DEFAULT_PATHS = configure_default_paths()
DEFAULT_RESULTS_ROOT = Path(DEFAULT_PATHS["R2LPL_RESULTS_ROOT"])
DEFAULT_CACHE_ROOT = Path(DEFAULT_PATHS["R2LPL_CACHE_ROOT"])
DEFAULT_WORKSPACE_ROOT = Path(DEFAULT_PATHS["R2LPL_ROOT"])
DEFAULT_PRETRAINED_CKPT = str(
    DEFAULT_RESULTS_ROOT
    / "checkpoints"
    / "pm_muvo_v4_t4_4096_full_noap_h4s_30_lw_anchor_score_softce02"
    / "last.ckpt"
)
DEFAULT_EXPERT_CACHE = str(DEFAULT_CACHE_ROOT / "cl_expert_caching")


def _run_command(command: List[str], dry_run: bool = False, env: Optional[Dict[str, str]] = None) -> None:
    printable = " ".join(command)
    print(f"\n[rollout-cl-auto] {printable}", flush=True)
    if env:
        preview_keys = ["CKPT_PATH", "ANCHOR_ROOT", "FILTER", "JOB_NAME", "REPLAY_IMAGE_SIZE_PX", "SIM_OUTPUT_DIR", "VIDEO_SAVE_DIR"]
        preview = " ".join(f"{key}={env[key]}" for key in preview_keys if key in env)
        if preview:
            print(f"[rollout-cl-auto-env] {preview}", flush=True)
    if dry_run:
        return
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(command, check=True, env=merged_env)


def _find_round_checkpoint(checkpoint_root: Path, job_name: str, task_index: int) -> Path:
    task_prefix = f"task_{task_index:02d}_"
    job_ckpt_dir = checkpoint_root / job_name
    candidates = sorted(job_ckpt_dir.glob(f"{task_prefix}*/last.ckpt"))
    if candidates:
        return candidates[-1]
    fallback = job_ckpt_dir / "joint_rollout" / "last.ckpt"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"No checkpoint found for job={job_name}, task_index={task_index} under {job_ckpt_dir}")


def _normalize_method(method: str) -> str:
    normalized = str(method).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "ft": "FT",
        "finetune": "FT",
        "derpp": "DERPP",
        "derplusplus": "DERPP",
        "derppsar": "DERPPSAR",
        "sar": "DERPPSAR",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported method: {method}. Expected FT, DERPP, or DERPPSAR.")
    return aliases[normalized]


def _round_job_name(iteration: int) -> str:
    return f"rollout_muvo_V4_{iteration:02d}"


def _format_experiment_dir_name(scenario_filter: str, method_name: str, experiment_suffix: Optional[str]) -> str:
    base_name = f"{scenario_filter}_{method_name}"
    suffix = str(experiment_suffix or "").strip()
    if not suffix:
        return base_name
    return f"{base_name}_{suffix.lstrip('_')}"


def _parse_round_index(job_name: str) -> int:
    try:
        return int(str(job_name).rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Cannot infer round index from job name: {job_name}") from exc


def _build_rollout_command(
    python_executable: str,
    job_name: str,
    scenario_filter: str,
    ckpt_path: Path,
    round_root: Path,
    args: argparse.Namespace,
) -> List[str]:
    command = [
        python_executable,
        "run/data_cache/run_rollout_cl_data_generation.py",
        f"job_name={job_name}",
        f"scenario_filter={scenario_filter}",
        f"scenario_builder={args.scenario_builder}",
        f"output_dir={round_root / 'logs' / 'rollout'}",
        f"rollout_ckpt_path={ckpt_path}",
        f"rollout_package_dir={round_root / 'rollout_packages'}",
        f"rollout_cl_cache_dir={round_root / 'rollout_cl_cache'}",
        f"rollout_debug_dir={round_root / 'rollout_cl_cache' / '_debug'}",
        f"num_workers={args.rollout_num_workers}",
        f"oracle_num_workers={args.oracle_num_workers}",
        f"gpus_per_worker={args.gpus_per_worker}",
        f"oracle_gpus_per_worker={args.oracle_gpus_per_worker}",
        f"max_rollout_scenarios={args.max_rollout_scenarios}",
        f"debug_rollout_cache={str(args.debug_rollout_cache).lower()}",
    ]
    if args.rollout_worker:
        command.append(f"worker={args.rollout_worker}")
    rollout_style = "road" if args.road else str(args.rollout_retrieval_style)
    if rollout_style == "road":
        command.extend(
            [
                "rollout_retrieval_style=road",
                f"road_frame_stride={args.road_frame_stride}",
                f"road_max_frames_per_scenario={args.road_max_frames_per_scenario}",
                f"top_k={args.road_top_k}",
            ]
        )
    command.extend(args.rollout_overrides)
    return command


def _format_hydra_list(paths: List[Path]) -> str:
    return "[" + ",".join(str(path) for path in paths) + "]"


def _build_cl_command(
    python_executable: str,
    job_name: str,
    current_ckpt: Path,
    cache_roots: List[Path],
    checkpoint_root: Path,
    args: argparse.Namespace,
    iteration: int,
    bootstrap_manifest: Optional[Path],
    method_name: str,
) -> List[str]:
    cl_method = "ft" if method_name == "FT" else "derpp"
    memory_capacity = 0 if method_name == "FT" else args.memory_capacity
    use_sar = method_name == "DERPPSAR"
    partition_strategy = "rollout_bucket" if use_sar else "scene_type"
    command = [
        python_executable,
        "run/training/run_rollout_cl_training.py",
        f"job_name={job_name}",
        f"start_model_path={current_ckpt}",
        f"checkpoint_root={checkpoint_root}",
        f"lightning.trainer.params.default_root_dir={checkpoint_root / job_name / 'lightning'}",
        f"rollout.cache_roots={_format_hydra_list(cache_roots)}",
        f"continual.method={cl_method}",
        f"continual.start_task_index={iteration}",
        f"continual.memory.capacity={memory_capacity}",
        f"continual.memory.partition_strategy={partition_strategy}",
        f"continual.memory.rollout_priority.enabled={str(use_sar).lower()}",
        f"expert_mix.enabled={str(args.expert_mix).lower()}",
        f"max_epoch={args.max_epoch}",
        f"max_steps={args.max_steps}",
        f"use_device_num={args.use_device_num}",
        f"dataloader.params.batch_size={args.batch_size}",
        f"val_dataloader.params.batch_size={args.val_batch_size}",
    ]
    if args.expert_mix:
        if args.expert_cache_path in {None, "", "None"}:
            raise ValueError("--expert-cache-path must be provided when --expert-mix is enabled.")
        command.extend(
            [
                f"expert_mix.cache_path={args.expert_cache_path}",
                "expert_mix.expand_iteration=true",
                "expert_mix.sample_basis=max",
                f"expert_mix.ratio={args.expert_mix_ratio}",
                f"expert_mix.loss_weight={args.expert_mix_loss_weight}",
                f"expert_mix.max_samples_per_task={args.expert_mix_max_samples}",
            ]
        )
    if bootstrap_manifest is not None:
        command.append(f"continual.memory.bootstrap_manifest={bootstrap_manifest}")
    command.extend(args.cl_overrides)
    return command


def _build_sim_command(
    ckpt_path: Path,
    round_root: Path,
    job_name: str,
    scenario_filter: str,
    args: argparse.Namespace,
) -> Tuple[List[str], Dict[str, str]]:
    sim_filter = str(args.sim_scenario_filter or scenario_filter)
    use_side_results = args.sim_scenario_filter is not None and sim_filter != str(scenario_filter)
    sim_result_root = (
        round_root / "side_results" / f"{args.sim_challenge}_{sim_filter}"
        if use_side_results
        else round_root
    )
    sim_logs_dir = sim_result_root / "simulation_logs"
    sim_videos_dir = sim_result_root / "simulation_videos"
    sim_output_dir = sim_logs_dir / str(args.sim_challenge)
    anchor_root = Path(
        os.environ.get(
            "R2LPL_RESULTS_ROOT",
            str(Path(args.workspace_root).expanduser().resolve() / "results"),
        )
    ) / "planner_anchors"
    env = {
        "RESULTS_DIR": str(round_root),
        "ANCHOR_ROOT": str(anchor_root),
        "LOG_ROOT": str(sim_logs_dir),
        "VIDEO_SAVE_DIR": str(sim_videos_dir),
        "SIM_OUTPUT_DIR": str(sim_output_dir),
        "CKPT_PATH": str(ckpt_path),
        "FILTER": sim_filter,
        "JOB_NAME": str(job_name),
        "BUILDER": str(args.sim_scenario_builder or args.scenario_builder),
        "WORKER": str(args.sim_worker),
        "CHALLENGE": str(args.sim_challenge),
        "NUM_GPU": str(args.sim_gpus_per_worker),
        "NUM_CPU": str(args.sim_cpus_per_worker),
        "SAVE_REPLAY": str(args.sim_save_replay).lower(),
        "REPLAY_IMAGE_SIZE_PX": str(args.sim_replay_image_size_px),
        "SAVE_NUBOARD_DATA": str(args.sim_save_nuboard_data).lower(),
    }
    env.update(dict(item.split("=", 1) for item in args.sim_env if "=" in item))
    return ["bash", "run/script/run_muvo_planner_ray_v4_noap.sh"], env


def _simulation_logs_root(round_root: Path, scenario_filter: str, sim_challenge: str, sim_scenario_filter: Optional[str]) -> Path:
    sim_filter = str(sim_scenario_filter or scenario_filter)
    if sim_scenario_filter is not None and sim_filter != str(scenario_filter):
        return round_root / "side_results" / f"{sim_challenge}_{sim_filter}" / "simulation_logs"
    return round_root / "simulation_logs"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    return loaded if isinstance(loaded, dict) else None


def _latest_file(root: Path, patterns: List[str]) -> Optional[Path]:
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(root.rglob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _summarize_scores(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _read_simulation_scores(sim_logs_root: Path, sim_challenge: str) -> Dict[str, Any]:
    aggregator_root = sim_logs_root / sim_challenge / "aggregator_metric"
    metric_path = _latest_file(aggregator_root, ["*.csv", "*.parquet"]) if aggregator_root.exists() else None
    if metric_path is None:
        metric_path = _latest_file(sim_logs_root, ["*.csv", "*.parquet"])
    if metric_path is None:
        return {"metric_path": None, "num_scenarios": 0, "score": _summarize_scores([])}

    rows: List[Dict[str, Any]] = []
    if metric_path.suffix == ".csv":
        with metric_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    else:
        try:
            import pandas as pd
        except Exception:
            return {"metric_path": str(metric_path), "num_scenarios": None, "score": _summarize_scores([]), "warning": "pandas unavailable for parquet summary"}
        rows = pd.read_parquet(metric_path).to_dict(orient="records")

    scores: List[float] = []
    for row in rows:
        if "score" not in row or row["score"] in {None, ""}:
            continue
        try:
            scores.append(float(row["score"]))
        except (TypeError, ValueError):
            continue
    return {
        "metric_path": str(metric_path),
        "num_scenarios": len(rows),
        "score": _summarize_scores(scores),
        "zero_score_count": int(sum(1 for score in scores if score <= 1e-6)),
    }


def _write_round_summary(
    round_root: Path,
    job_name: str,
    method_name: str,
    scenario_filter: str,
    checkpoint_path: Optional[Path],
    checkpoint_root: Path,
    sim_challenge: str,
    sim_scenario_filter: Optional[str],
) -> Path:
    rollout_cache_root = round_root / "rollout_cl_cache"
    rollout_summary = _read_json(rollout_cache_root / "rollout_cl_generation_summary_compact.json")
    if rollout_summary is None:
        sample_count = len(list(rollout_cache_root.rglob("anchor_indice.gz")))
        package_count = len(list((round_root / "rollout_packages").rglob("rollout_package.gz")))
        rollout_summary = {
            "num_scenarios": package_count or None,
            "num_frame_tasks": None,
            "total_kept": sample_count,
            "total_dropped_unrecoverable": None,
            "num_errors": None,
            "note": "compact summary missing; counted cache files without opening the large full summary",
        }

    manifest_path = checkpoint_root / job_name / "rollout_continual_manifest.yaml"
    cl_summary: Dict[str, Any] = {"manifest_path": str(manifest_path) if manifest_path.exists() else None}
    if manifest_path.exists():
        manifest = None
        try:
            from omegaconf import OmegaConf

            manifest = OmegaConf.to_container(OmegaConf.load(manifest_path), resolve=True)
        except Exception:
            try:
                import yaml

                with manifest_path.open("r", encoding="utf-8") as file:
                    manifest = yaml.safe_load(file)
            except Exception as exc:
                cl_summary["warning"] = f"failed to parse manifest: {exc}"
        manifest = manifest or {}
        tasks = list((manifest or {}).get("tasks", []) or [])
        last_task = tasks[-1] if tasks else {}
        cl_summary.update(
            {
                "method": (manifest or {}).get("method"),
                "num_completed_tasks": len(tasks),
                "memory_capacity": (manifest or {}).get("memory_capacity"),
                "last_task": {
                    key: last_task.get(key)
                    for key in [
                        "task_id",
                        "task_name",
                        "train_size",
                        "eval_size",
                        "history_eval_size",
                        "memory_eval_size",
                        "expert_mix_size",
                        "memory_size_after_task",
                        "final_checkpoint",
                    ]
                },
            }
        )

    sim_logs_root = _simulation_logs_root(round_root, scenario_filter, sim_challenge, sim_scenario_filter)
    summary = {
        "job_name": job_name,
        "method": method_name,
        "scenario_filter": scenario_filter,
        "sim_scenario_filter": str(sim_scenario_filter or scenario_filter),
        "simulation_logs_root": str(sim_logs_root),
        "round_root": str(round_root),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "rollout": rollout_summary,
        "continual_learning": cl_summary,
        "simulation": _read_simulation_scores(sim_logs_root, sim_challenge),
    }
    summary_path = round_root / "rollout_cl_round_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run N rounds of Muvo v4 rollout data generation + rollout CL training.")
    parser.add_argument("--rounds", type=int, required=True, help="Number of rollout+CL iterations to run.")
    parser.add_argument("--scenario-filter", type=str, default="test14-hard", help="Hydra scenario_filter config name and output directory name.")
    parser.add_argument("--scenario-builder", type=str, default="nuplan_test", help="Hydra scenario_builder config name and output directory name.")
    parser.add_argument("--method", type=str, default="DERPPSAR", help="FT, DERPP, or DERPPSAR. Default output root is ${scenario_filter}_${method}.")
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--workspace-root", type=str, default=str(DEFAULT_WORKSPACE_ROOT))
    parser.add_argument("--initial-ckpt", type=str, default=DEFAULT_PRETRAINED_CKPT)
    parser.add_argument("--expert-cache-path", type=str, default=DEFAULT_EXPERT_CACHE)
    parser.add_argument("--base-output-root", type=str, default=None)
    parser.add_argument(
        "--experiment-suffix",
        type=str,
        default=None,
        help="Optional suffix appended to the default experiment directory name. No effect when --base-output-root is set.",
    )
    parser.add_argument("--resume-from", type=int, default=0, help="First iteration index to execute.")
    parser.add_argument("--sim-only", action="store_true", help="Only run simulation for existing rollout+CL jobs.")
    parser.add_argument("--sim-job-name", type=str, default=None, help="When --sim-only is set, simulate only this job name.")
    parser.add_argument("--skip-sim", action="store_true", help="Run rollout+CL without post-CL simulation.")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--rollout-num-workers", type=int, default=128)
    parser.add_argument(
        "--rollout-worker",
        type=str,
        default=None,
        help="Hydra worker config used to extract scenarios before rollout generation, e.g. custom_ray_distributed.",
    )
    parser.add_argument("--oracle-num-workers", type=int, default=192)
    parser.add_argument("--gpus-per-worker", type=float, default=0.03)
    parser.add_argument("--oracle-gpus-per-worker", type=float, default=0.0)
    parser.add_argument("--max-rollout-scenarios", type=str, default="null")
    parser.add_argument("--debug-rollout-cache", action="store_true")
    parser.add_argument("--road", action="store_true", help="Use RoaD-style rollout-as-demonstration cache generation.")
    parser.add_argument(
        "--rollout-retrieval-style",
        type=str,
        choices=["r2lpl", "road"],
        default="r2lpl",
        help="Rollout cache generation style. r2lpl keeps recoverability-aware retrieval; road uses rollout-as-demonstration SFT targets.",
    )
    parser.add_argument("--road-frame-stride", type=int, default=5)
    parser.add_argument("--road-max-frames-per-scenario", type=int, default=30)
    parser.add_argument("--road-top-k", type=int, default=16, help="Hydra top_k used during RoaD-style rollout action sampling.")

    parser.add_argument("--memory-capacity", type=int, default=4096)
    parser.add_argument(
        "--expert-mix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable original expert-cache mixing during rollout CL training. Disabled by default.",
    )
    parser.add_argument("--expert-mix-ratio", type=float, default=0.25)
    parser.add_argument("--expert-mix-loss-weight", type=float, default=0.25)
    parser.add_argument("--expert-mix-max-samples", type=int, default=80000)
    parser.add_argument("--max-epoch", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--use-device-num", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--val-batch-size", type=int, default=16)

    parser.add_argument("--sim-scenario-filter", type=str, default=None, help="Scenario filter for simulation. Defaults to --scenario-filter.")
    parser.add_argument("--sim-scenario-builder", type=str, default=None, help="Scenario builder for simulation. Defaults to --scenario-builder.")
    parser.add_argument("--sim-worker", type=str, default="custom_ray_distributed_server_128")
    parser.add_argument("--sim-challenge", type=str, default="closed_loop_nonreactive_agents")
    parser.add_argument("--sim-gpus-per-worker", type=float, default=0.03)
    parser.add_argument("--sim-cpus-per-worker", type=int, default=1)
    parser.add_argument("--sim-save-replay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sim-replay-image-size-px", type=int, default=768)
    parser.add_argument("--sim-save-nuboard-data", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sim-env", nargs="*", default=[], help="Extra KEY=VALUE environment overrides for run_muvo_planner_ray_v4_noap.sh.")

    parser.add_argument("--rollout-overrides", nargs="*", default=[], help="Extra Hydra overrides for rollout generation.")
    parser.add_argument("--cl-overrides", nargs="*", default=[], help="Extra Hydra overrides for rollout CL training.")
    args = parser.parse_args()

    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.resume_from < 0 or args.resume_from >= args.rounds:
        raise ValueError("--resume-from must be in [0, rounds)")
    method_name = _normalize_method(args.method)

    workspace_root = Path(args.workspace_root).expanduser().resolve()
    os.chdir(workspace_root)

    home = Path(os.environ.get("HOME", str(Path.home()))).expanduser()
    if args.base_output_root is None:
        experiment_dir_name = _format_experiment_dir_name(args.scenario_filter, method_name, args.experiment_suffix)
        base_output_root = DEFAULT_RESULTS_ROOT / "rollout" / experiment_dir_name
    else:
        base_output_root = Path(args.base_output_root).expanduser()
    base_output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = base_output_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    if args.sim_only:
        if args.sim_job_name:
            sim_jobs = [(args.sim_job_name, _parse_round_index(args.sim_job_name))]
        else:
            sim_jobs = [(_round_job_name(iteration), iteration) for iteration in range(args.resume_from, args.rounds)]
        for job_name, iteration in sim_jobs:
            round_root = base_output_root / job_name
            ckpt_path = (
                checkpoint_root / job_name / f"task_{iteration:02d}_rollout_cl_cache" / "last.ckpt"
                if args.dry_run
                else _find_round_checkpoint(checkpoint_root, job_name, iteration)
            )
            sim_command, sim_env = _build_sim_command(
                ckpt_path=ckpt_path,
                round_root=round_root,
                job_name=job_name,
                scenario_filter=args.scenario_filter,
                args=args,
            )
            _run_command(sim_command, dry_run=args.dry_run, env=sim_env)
            if not args.dry_run and args.sim_scenario_filter is None:
                summary_path = _write_round_summary(
                    round_root=round_root,
                    job_name=job_name,
                    method_name=method_name,
                    scenario_filter=args.scenario_filter,
                    checkpoint_path=ckpt_path,
                    checkpoint_root=checkpoint_root,
                    sim_challenge=args.sim_challenge,
                    sim_scenario_filter=args.sim_scenario_filter,
                )
                print(f"[rollout-cl-auto] wrote round summary: {summary_path}", flush=True)
        return

    cache_roots: List[Path] = []
    for iteration in range(args.resume_from):
        cache_roots.append(base_output_root / _round_job_name(iteration) / "rollout_cl_cache")

    current_ckpt = Path(args.initial_ckpt).expanduser()
    bootstrap_manifest: Optional[Path] = None
    if args.resume_from > 0:
        previous_job = _round_job_name(args.resume_from - 1)
        current_ckpt = _find_round_checkpoint(checkpoint_root, previous_job, args.resume_from - 1)
        bootstrap_manifest = checkpoint_root / previous_job / "rollout_continual_manifest.yaml"
        if not bootstrap_manifest.is_file():
            raise FileNotFoundError(f"Previous memory manifest does not exist: {bootstrap_manifest}")

    for iteration in range(args.resume_from, args.rounds):
        job_name = _round_job_name(iteration)
        round_root = base_output_root / job_name
        round_root.mkdir(parents=True, exist_ok=True)

        rollout_command = _build_rollout_command(
            python_executable=args.python_executable,
            job_name=job_name,
            scenario_filter=args.scenario_filter,
            ckpt_path=current_ckpt,
            round_root=round_root,
            args=args,
        )
        _run_command(rollout_command, dry_run=args.dry_run)

        current_cache_root = round_root / "rollout_cl_cache"
        cache_roots.append(current_cache_root)

        cl_command = _build_cl_command(
            python_executable=args.python_executable,
            job_name=job_name,
            current_ckpt=current_ckpt,
            cache_roots=cache_roots,
            checkpoint_root=checkpoint_root,
            args=args,
            iteration=iteration,
            bootstrap_manifest=bootstrap_manifest,
            method_name=method_name,
        )
        _run_command(cl_command, dry_run=args.dry_run)

        if not args.dry_run:
            current_ckpt = _find_round_checkpoint(checkpoint_root, job_name, iteration)
            bootstrap_manifest = checkpoint_root / job_name / "rollout_continual_manifest.yaml"
            if not bootstrap_manifest.is_file():
                raise FileNotFoundError(f"Current memory manifest does not exist after CL: {bootstrap_manifest}")
        else:
            current_ckpt = checkpoint_root / job_name / f"task_{iteration:02d}_rollout_cl_cache" / "last.ckpt"
            bootstrap_manifest = checkpoint_root / job_name / "rollout_continual_manifest.yaml"

        if not args.skip_sim:
            sim_command, sim_env = _build_sim_command(
                ckpt_path=current_ckpt,
                round_root=round_root,
                job_name=job_name,
                scenario_filter=args.scenario_filter,
                args=args,
            )
            _run_command(sim_command, dry_run=args.dry_run, env=sim_env)

        if not args.dry_run and args.sim_scenario_filter is None:
            summary_path = _write_round_summary(
                round_root=round_root,
                job_name=job_name,
                method_name=method_name,
                scenario_filter=args.scenario_filter,
                checkpoint_path=current_ckpt,
                checkpoint_root=checkpoint_root,
                sim_challenge=args.sim_challenge,
                sim_scenario_filter=args.sim_scenario_filter,
            )
            print(f"[rollout-cl-auto] iteration {iteration} complete: next_ckpt={current_ckpt}", flush=True)
            print(f"[rollout-cl-auto] wrote round summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
