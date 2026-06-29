# R2LPL

## Open-source Pipeline

This repository uses `results/` under the repository root as the default runtime workspace.
The Python entrypoints call `hybrid_planner.utils.default_paths.configure_default_paths()` before Hydra loads configs.
By default it sets:

- `R2LPL_ROOT=<repo>`
- `R2LPL_RESULTS_ROOT=<repo>/results`
- `R2LPL_CACHE_ROOT=<repo>/results/cache`
- `NUPLAN_DATA_ROOT=~/nuplan/dataset` when not already set
- `NUPLAN_MAPS_ROOT=$NUPLAN_DATA_ROOT/maps` when not already set
- `NUPLAN_EXP_ROOT=<repo>/results/nuplan_exp`

You can override any of these environment variables before running scripts.

### Stage 1: Data Caching

Build nuPlan feature cache:

```bash
conda activate hybrid
python run/data_cache/run_data_caching_trainval.py \
  job_name=trainval_caching \
  cache.cache_path=results/cache
```

This writes cached samples to:

```text
results/cache/trainval_caching/
```

Append anchor indices and dense anchor scores for MV training:

```bash
python run/data_cache/run_data_caching_trainval_anchor_score.py \
  job_name=trainval_caching \
  cache.cache_path=results/cache \
  override_anchor_indice=true \
  override_anchor_score=true
```

This reads:

```text
results/planner_anchors/planner_anchors_M4096s_T4.0_step20_full.npy
results/cache/trainval_caching/
```

and writes per-scenario files such as `anchor_indice.gz` and `anchor_scores.gz` into the cache.

### Stage 2: MV Training

Run MV planner training from the cached data:

```bash
bash run/script/run_mv_training.sh
```

The wrapper defaults to:

```text
input cache: results/cache/trainval_caching/
input anchor: results/planner_anchors/planner_anchors_M4096s_T4.0_step20_full.npy
output checkpoints: results/checkpoints/<JOB_NAME>/
```

For scratch training, `START_MODEL_PATH=None` is used by default. To warm-start, pass:

```bash
START_MODEL_PATH=results/checkpoints/<pretrained-job>/last.ckpt bash run/script/run_mv_training.sh
```

### Stage 3: Rollout CL

Generate rollout CL cache from an MV checkpoint:

```bash
python run/data_cache/run_rollout_cl_data_generation.py \
  job_name=rollout_cache_test14_hard_ro0 \
  rollout_ckpt_path=results/checkpoints/<mv-job>/last.ckpt \
  rollout_package_dir=results/rollout/rollout_cache_test14_hard_ro0/rollout_packages \
  rollout_cl_cache_dir=results/rollout/rollout_cache_test14_hard_ro0/rollout_cl_cache
```

Train continual learning on one or more rollout cache roots:

```bash
python run/training/run_rollout_cl_training.py \
  job_name=rollout_cl_derpp \
  start_model_path=results/checkpoints/<mv-job>/last.ckpt \
  checkpoint_root=results/checkpoints \
  rollout.cache_roots=[results/rollout/rollout_cache_test14_hard_ro0/rollout_cl_cache]
```

The automated N-round wrapper connects rollout generation, rollout CL training, and optional simulation:

```bash
python run/script/run_rollout_cl_auto.py \
  --rounds 1 \
  --initial-ckpt results/checkpoints/<mv-job>/last.ckpt
```

Expert-cache mixing is disabled by default, so the automated rollout CL path does not require an expert cache.
To mix original expert-cache samples into CL training, enable it explicitly:

```bash
python run/script/run_rollout_cl_auto.py \
  --rounds 1 \
  --initial-ckpt results/checkpoints/<mv-job>/last.ckpt \
  --expert-mix \
  --expert-cache-path results/cache/cl_expert_caching
```

### Minimal External Files

For a minimal reproduction, prepare these files/directories:

- nuPlan DB and maps under the system defaults:
  - `$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/mini` or `$NUPLAN_DATA_ROOT/nuplan-v1.1/trainval`
  - `$NUPLAN_DATA_ROOT/nuplan-v1.1/test` if using `scenario_builder=nuplan_test`
  - `$NUPLAN_MAPS_ROOT`
- planner anchor:
  - `results/planner_anchors/planner_anchors_M4096s_T4.0_step20_full.npy`
- optional warm-start checkpoint:
  - `results/checkpoints/muvo_base_model/last.ckpt`
- optional prebuilt MV cache if skipping Stage 1:
  - `results/cache/trainval_caching/`
- optional expert cache only when using `--expert-mix`:
  - `results/cache/cl_expert_caching/`
- optional rollout cache if skipping rollout generation:
  - `results/rollout/<rollout-job>/rollout_cl_cache/`

  
