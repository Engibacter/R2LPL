export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cwd=$(pwd)
export R2LPL_ROOT=${R2LPL_ROOT:-"$cwd"}
export R2LPL_RESULTS_ROOT=${R2LPL_RESULTS_ROOT:-"$cwd/results"}
export R2LPL_CACHE_ROOT=${R2LPL_CACHE_ROOT:-"$R2LPL_RESULTS_ROOT/cache"}
RESULTS_DIR=${RESULTS_DIR:-"$R2LPL_RESULTS_ROOT"}
CKPT_ROOT=${CKPT_ROOT:-"$RESULTS_DIR/checkpoints"}
ANCHOR_ROOT=${ANCHOR_ROOT:-"$RESULTS_DIR/planner_anchors"}
LOG_ROOT=${LOG_ROOT:-"$RESULTS_DIR/simulation_logs"}
mkdir -p "$CKPT_ROOT" "$ANCHOR_ROOT" "$LOG_ROOT" "$RESULTS_DIR/rollout_data"
SIM_OUTPUT_DIR=${SIM_OUTPUT_DIR:-""}
DEVICE=${DEVICE:-"auto"}  # or "auto" | "cpu" | "cuda:1"
USE_HAUSDORFF=${USE_HAUSDORFF:-false}  # whether to use hausdorff distance in scoring
HAUSDORFF_WEIGHT=${HAUSDORFF_WEIGHT:-0.1}  # weight for hausdorff distance in scoring
USE_ANCHOR_VELOCITY=${USE_ANCHOR_VELOCITY:-false}  # whether to use anchor velocity in scoring


PLANNER=${PLANNER:-muvo_abstract_planner_v4_noap}
USE_EVAL=${USE_EVAL:-false}  # whether to use trajectory evaluator to evaluate the trajectories
NUM_SAMPLES=${NUM_SAMPLES:-32}  # number of trajectory samples to generate
NUM_GPU=${NUM_GPU:-0.03}  # number of GPU to allocate per simulation
NUM_CPU=${NUM_CPU:-1}
TOP_K=${TOP_K:-0.0} # number of top trajectories to consider
TOP_P=${TOP_P:-0.0} # cumulative probability for nucleus sampling
PRED_LOGPROB_WEIGHT=${PRED_LOGPROB_WEIGHT:-1.0}  # weight for prediction loss in scoring
EVAL_LOGPROB_WEIGHT=${EVAL_LOGPROB_WEIGHT:-0.0}  # weight for evaluation loss in scoring
SCORE_CHUNK_SIZE=${SCORE_CHUNK_SIZE:-512}  # chunk size for scoring anchors in inference (reduce if OOM, increase if under-utilized GPU)
PREDICTION_MODE=${PREDICTION_MODE:-'CYAW'}  # 'CV', 'CA', 'CYAW', 'prediction'
EVAL_SAMPLING_DT=${EVAL_SAMPLING_DT:-0.2}  # sampling interval for evaluation, only used when use_eval is true

INTERMEDIATE_DIM=${INTERMEDIATE_DIM:-256}  # hidden dimension for the model

USE_PREDICTION=${USE_PREDICTION:-false}  # whether to use agent prediction
PREDICTION_USE_CV_DELTA=${PREDICTION_USE_CV_DELTA:-false}  # whether to use CV-based delta (current implementation) or simple future-expert delta for prediction supervision

BUILDER=${BUILDER:-nuplan_test} # nuplan_trainval | nuplan_test
FILTER=${FILTER:-test14-hard} # 'val14', 'val14-reduced', 'test14-random', 'test14-hard', 'test14-hard-test'
WORKER=${WORKER:-custom_ray_distributed}
WORKER_THREADS_PER_NODE=${WORKER_THREADS_PER_NODE:-128}
CHALLENGE=${CHALLENGE:-"closed_loop_nonreactive_agents"}
# CHALLENGE="closed_loop_reactive_agents"
# CHALLENGE="open_loop_boxes"


# PLANNER_ANCHOR=planner_anchors_M1024s_T1.0_step5_k1024.npy
# PLANNER_ANCHOR="planner_anchors_M8192s_T8.0_step40_k8192.npy"
# PLANNER_ANCHOR="planner_anchors_M8192s_T4.0_step20_k8192.npy"
# PLANNER_ANCHOR="planner_anchors_M4096s_T4.0_step20_k4096.npy"
# PLANNER_ANCHOR="planner_anchors_M4096s_T2.0_step10_k4096.npy"
# PLANNER_ANCHOR="planner_anchors_M2048s_T2.0_step10_k2048.npy"
# PLANNER_ANCHOR="planner_anchors_M1024s_T2.0_step10_k1024.npy"
# PLANNER_ANCHOR="planner_anchors_M1024s_T1.0_step5_k1024.npy"
# PLANNER_ANCHOR="planner_anchors_M4096s_T2.0_step10_dynamic.npy"
# PLANNER_ANCHOR="planner_anchors_M8192s_T2.0_step10_dynamic.npy"
# PLANNER_ANCHOR="planner_anchors_M512s_T1.0_step5_full.npy"
PLANNER_ANCHOR=${PLANNER_ANCHOR:-"planner_anchors_M4096s_T4.0_step20_full.npy"}
# PLANNER_ANCHOR="planner_anchors_M1024s_T2.0_step10_full.npy"
# PLANNER_ANCHOR="planner_anchors_M16384s_T4.0_step20_full_sep_speed.npy"
# PLANNER_ANCHOR="planner_anchors_M128s_T4.0_step20_full.npy"

MAX_SPEED_DIFF=${MAX_SPEED_DIFF:-20} # maximum speed difference for dynamic scoring
TRAIN_ANCHOR_NUM=${TRAIN_ANCHOR_NUM:-4096}
TEST_ANCHOR_NUM=${TEST_ANCHOR_NUM:-4096} # number of anchors to sample during testing |512 for t > 2

# CKPT=pm_muvo_t1_1024_noreg_ap_100/last.ckpt
# CKPT=pm_muvo_t2_4096_noreg_ap_100/last.ckpt
# CKPT=pm_muvo_t8_8192_noreg_ap_50/last.ckpt
# CKPT="rl_muvo_t8_8192_noreg_noap_s100_v3/steps-step-001190.ckpt"
# CKPT="pm_muvo_v4_t4_4096_full_ap_h4s_30_lw_anchor_score_softce02_expert/last.ckpt"
CKPT=${CKPT:-"pm_muvo_v4_t4_4096_full_noap_h4s_30_lw_anchor_score_softce02_testrl_ft/last.ckpt"}  

# CKPT="mv_cl_single_agem_task_mem24000_r005/task_00_single_increment/last.ckpt"  # CL single increment ckpt
JOB_NAME=${JOB_NAME:-pm_muvo_v4_t4_4096_full_noap_h4s_30_lw_anchor_score_softce02_cl_cyaw_32_noeval/$FILTER}
VIDEO_SAVE_DIR=${VIDEO_SAVE_DIR:-"$RESULTS_DIR/simulation_videos/$CHALLENGE/$JOB_NAME"}
SAVE_REPLAY=${SAVE_REPLAY:-false}  # whether to save replay video during simulation
REPLAY_IMAGE_SIZE_PX=${REPLAY_IMAGE_SIZE_PX:-2048}  # replay video frame resolution
SAVE_NUBOARD_DATA=${SAVE_NUBOARD_DATA:-false}  # no dedicated run_simulation_ray flag exists; when false, prune job artifacts after aggregation
NUM_POSES=${NUM_POSES:-20}

SAVE_ROLLOUT_DATA=${SAVE_ROLLOUT_DATA:-false}  # whether to save rollout data for future training
ROLLOUT_CACHE_DIR=${ROLLOUT_CACHE_DIR:-"$RESULTS_DIR/rollout_data"} # path to save rollout data
SIM_OUTPUT_DIR=${SIM_OUTPUT_DIR:-"$LOG_ROOT/$CHALLENGE/$JOB_NAME"}
CKPT_PATH=${CKPT_PATH:-"$CKPT_ROOT/$CKPT"}

python ./run/simulation/run_simulation_ray.py \
    +simulation=$CHALLENGE \
    ego_controller/tracker=pplqr_tracker \
    planner=$PLANNER \
    scenario_builder=$BUILDER \
    scenario_filter=$FILTER \
    worker=$WORKER \
    worker.threads_per_node=$WORKER_THREADS_PER_NODE \
    verbose=true \
    experiment_uid="" \
    output_dir="$SIM_OUTPUT_DIR" \
    number_of_gpus_allocated_per_simulation=$NUM_GPU \
    number_of_cpus_allocated_per_simulation=$NUM_CPU \
    planner.muvo_planner.save_replay=$SAVE_REPLAY \
    planner.muvo_planner.ckpt_path="$CKPT_PATH" \
    planner.muvo_planner.video_dir="$VIDEO_SAVE_DIR" \
    planner.muvo_planner.replay_image_size_px=$REPLAY_IMAGE_SIZE_PX \
    planner.muvo_planner.use_eval=$USE_EVAL \
    planner.muvo_planner.num_samples=$NUM_SAMPLES \
    planner.muvo_planner.top_k=$TOP_K \
    planner.muvo_planner.top_p=$TOP_P \
    planner.muvo_planner.device=$DEVICE \
    planner.muvo_planner.use_hausdorff=$USE_HAUSDORFF \
    planner.muvo_planner.hausdorff_weight=$HAUSDORFF_WEIGHT \
    planner.muvo_planner.use_anchor_velocity=$USE_ANCHOR_VELOCITY \
    planner.muvo_planner.model.future_sampling.num_poses=$NUM_POSES \
    planner.muvo_planner.future_sampling.num_poses=$NUM_POSES \
    planner.muvo_planner.model.planner_anchor_path="$ANCHOR_ROOT/$PLANNER_ANCHOR"\
    planner.muvo_planner.model.train_anchor_num=$TRAIN_ANCHOR_NUM \
    planner.muvo_planner.model.test_anchor_num=$TEST_ANCHOR_NUM \
    planner.muvo_planner.model.use_prediction=$USE_PREDICTION \
    planner.muvo_planner.model.intermidiate_dim=$INTERMEDIATE_DIM \
    planner.muvo_planner.model.score_chunk_size=$SCORE_CHUNK_SIZE \
    planner.muvo_planner.model.prediction_use_cv_delta=$PREDICTION_USE_CV_DELTA \
    planner.muvo_planner.pred_logprob_weight=$PRED_LOGPROB_WEIGHT \
    planner.muvo_planner.eval_logprob_weight=$EVAL_LOGPROB_WEIGHT \
    planner.muvo_planner.save_rollout_data=$SAVE_ROLLOUT_DATA \
    planner.muvo_planner.rollout_cache_dir="$ROLLOUT_CACHE_DIR" \
    planner.muvo_planner.prediction_mode=$PREDICTION_MODE \
    planner.muvo_planner.eval_sampling_dt=$EVAL_SAMPLING_DT

SIM_EXIT_CODE=$?

python - "$SIM_OUTPUT_DIR" <<'PY'
from pathlib import Path
import sys

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    print(f"[post] skip parquet->csv conversion: {exc}", file=sys.stderr)
    sys.exit(0)

output_dir = Path(sys.argv[1])
aggregator_dir = output_dir / "aggregator_metric"
if not aggregator_dir.exists():
    sys.exit(0)

for parquet_path in sorted(aggregator_dir.glob("*.parquet")):
    csv_path = parquet_path.with_suffix(".csv")
    df = pd.read_parquet(parquet_path)
    df.to_csv(csv_path, index=False)
    print(f"[post] wrote {csv_path}")
PY

if [[ "$SAVE_REPLAY" == "true" && -d "$VIDEO_SAVE_DIR" ]]; then
    python - "$SIM_OUTPUT_DIR" "$VIDEO_SAVE_DIR" <<'PY'
from pathlib import Path
import re
import sys

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    print(f"[post] skip video rename: {exc}", file=sys.stderr)
    sys.exit(0)

output_dir = Path(sys.argv[1])
video_dir = Path(sys.argv[2])
aggregator_dir = output_dir / "aggregator_metric"
if not aggregator_dir.exists() or not video_dir.exists():
    sys.exit(0)

metric_files = sorted(aggregator_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime)
if not metric_files:
    metric_files = sorted(aggregator_dir.glob("*.parquet"), key=lambda path: path.stat().st_mtime)
if not metric_files:
    sys.exit(0)

metric_path = metric_files[-1]
if metric_path.suffix == ".csv":
    df = pd.read_csv(metric_path)
else:
    df = pd.read_parquet(metric_path)

required_cols = {"scenario", "log_name", "score"}
if not required_cols.issubset(df.columns):
    print(f"[post] skip video rename: missing columns in {metric_path}", file=sys.stderr)
    sys.exit(0)

score_map = {}
for row in df.itertuples(index=False):
    scenario = str(getattr(row, "scenario", ""))
    log_name = str(getattr(row, "log_name", ""))
    score = getattr(row, "score", None)
    if not scenario or not log_name or pd.isna(score):
        continue
    score_int = int(round(max(0.0, min(1.0, float(score))) * 100.0))
    score_map[(log_name, scenario)] = f"{score_int:03d}"

renamed = 0
for video_path in sorted(video_dir.rglob("*.mp4")):
    stem = re.sub(r"_\d{3}$", "", video_path.stem)
    matched_score = None
    for (log_name, scenario), score_suffix in score_map.items():
        prefix = f"{log_name}_{scenario}"
        if stem == prefix or stem.startswith(prefix + "_"):
            matched_score = score_suffix
            break
    if matched_score is None:
        continue

    new_path = video_path.with_name(f"{stem}_{matched_score}{video_path.suffix}")
    if new_path == video_path:
        continue
    if new_path.exists():
        new_path.unlink()
    video_path.rename(new_path)
    renamed += 1

print(f"[post] renamed {renamed} video files with score suffixes")
PY
fi

if [[ "$SIM_EXIT_CODE" -eq 0 && "$SAVE_NUBOARD_DATA" != "true" && -d "$SIM_OUTPUT_DIR" ]]; then
    python - "$SIM_OUTPUT_DIR" <<'PY'
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1])
aggregator = root / "aggregator_metric"
if not root.exists() or not aggregator.exists():
    sys.exit(0)

for child in list(root.iterdir()):
    if child.name == "aggregator_metric":
        continue
    if child.is_dir():
        shutil.rmtree(child)
    else:
        child.unlink()

print(f"[post] pruned non-aggregator artifacts under {root}")
PY
fi

exit $SIM_EXIT_CODE
