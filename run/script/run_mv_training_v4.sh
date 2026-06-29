cwd=$(pwd)
export R2LPL_ROOT=${R2LPL_ROOT:-"$cwd"}
export R2LPL_RESULTS_ROOT=${R2LPL_RESULTS_ROOT:-"$cwd/results"}
export R2LPL_CACHE_ROOT=${R2LPL_CACHE_ROOT:-"$R2LPL_RESULTS_ROOT/cache"}
RESULTS_DIR="$R2LPL_RESULTS_ROOT"
mkdir -p "$RESULTS_DIR/checkpoints" "$RESULTS_DIR/planner_anchors" "$R2LPL_CACHE_ROOT"

model=muvo_planner_v4_noap

# dataset
SPLIT=log_splits # log_splits | test_splitter
TRAINDATA='all' # 'all' | 'train_only' | 'val_only'
CACHE_PATH=${CACHE_PATH:-"$R2LPL_CACHE_ROOT/trainval_caching"}
EXPAND_ITERATION=true  # whether to expand cache iterations as separate samples
USE_MANIFEST=false  # whether to use manifest files for training instead of scanning directories; if true, will look for manifest files at $CACHE_PATH/{train,val}_manifest.json
VAL_RATIO=0.05  # ratio split from the loaded train dataset to build validation dataset
VAL_SPLIT_SEED=0  # random seed used when splitting validation samples from train dataset
SCENE_TOKENS_PATH="None"
if [[ "$USE_MANIFEST" == true ]]; then
    SCENE_TOKENS_PATH="$CACHE_PATH/selected_scenario_tokens.json"
fi

# training hyperparameters
EPOCHS=30  # maximum number of training epochs
BATCHSIZE=72  # training batch size
INTERMEDIATE_DIM=256  # hidden dimension for the model
ENCODER_DEPTH=4
PLANNING_DECODER_DEPTHS=8
PREDICTION_DECODER_DEPTHS=5
DEVICE_NUM="[0,1,2,3]"  # which GPU to use, -1 for CPU, >=0 for GPU id
LR=1e-4  # learning rate for optimizer
WARMUP_STEPS=1000  # number of warmup steps for learning rate scheduler
CHECK_INVALID_GRAD=false  # whether to check for invalid gradients during training

REGRESS_LOSS_WEIGHT=0.0  # weight for regression loss
REGRESS_YAW_LOSS_WEIGHT=0.0  # extra wrapped-yaw SmoothL1 weight inside regression; 0 keeps legacy xy-only behavior

CLASSY_LOSS_WEIGHT=10.0  # weight for classification loss
USE_ANCHOR_INDICE=true  # whether to use anchor indice as input
ANCHOR_INDICE_NAME="anchor_indice" #"anchor_indice_t4_16384" #"anchor_indice_t2_4096" #"anchor_indice_t1_512"
ANCHOR_SCORE_NAME="anchor_scores" #"dynamic_score_t1_512" #"None"
TEACHER_ANCHOR_RATIO=0.1  # ratio of teacher anchors kept in sampled candidates during training
TEACHER_CE_WEIGHT=1.0  # weight on soft teacher-top1 cross entropy within sampled candidates
TEACHER_CE_LABEL_SMOOTHING=0.2  # smooth teacher target onto other positive anchors before fallback to non-teacher anchors
ANCHOR_SCORE_KL_WEIGHT=1.0  # weight on anchor-score KL when enabled
ANCHOR_SCORE_NEG_LOSS_WEIGHT=1.0  # weight on focal-style suppression for illegal but high-score candidates
USE_ANCHOR_SCORE_KL=true  # whether to match sampled candidate distribution to anchor scores

TRAIN_ANCHOR_NUM=256  # number of anchors to sample during training (256 safe for t=1 in bf16mixed)
TEST_ANCHOR_NUM=256  # number of anchors to sample during testing |512 for t > 2
SCORE_CHUNK_SIZE=512

USE_PREDICTION=false  # whether to use agent prediction
PREDICTION_USE_CV_DELTA=false  # whether to use CV-based delta (current implementation) or simple future-expert delta for prediction supervision
PREDICTION_LOSS_WEIGHT=1.0  # weight for prediction loss

PRECISION_MODE='bf16-mixed'  # precision mode: full precision (32, '32' or '32-true'), 16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed')
VISUALIZE=true  # whether to visualize training samples

START_MODEL_PATH=${START_MODEL_PATH:-"None"}  # path to a checkpoint to start training from

# PLANNER_ANCHOR="planner_anchors_M1024s_T2.0_step10_full.npy"
# PLANNER_ANCHOR="planner_anchors_M512s_T1.0_step5_full.npy"
PLANNER_ANCHOR=${PLANNER_ANCHOR:-"planner_anchors_M4096s_T4.0_step20_full.npy"}
# PLANNER_ANCHOR="planner_anchors_M16384s_T4.0_step20_full_sep_speed.npy"
NUM_POSE=20

# JOB_NAME=pm_muvo_v3_t1_512_full_ap_ce_sm05_h8s_50_cl
JOB_NAME=pm_muvo_v4_t4_4096_full_noap_h4s_30_lw_anchor_score_softce02_testrl_ft
DISABLE_CHECKPOINTING=false

python ./run/training/run_muvo_training.py \
    job_name="$JOB_NAME" \
    model=$model \
    splitter=$SPLIT \
    max_epoch=$EPOCHS \
    use_device_num=$DEVICE_NUM \
    learning_rate=$LR \
    warmup_steps=$WARMUP_STEPS \
    check_invalid_grad=$CHECK_INVALID_GRAD \
    train_data=$TRAINDATA \
    visualize=$VISUALIZE \
    +start_model_path="$START_MODEL_PATH" \
    cache.expand_iteration=$EXPAND_ITERATION \
    cache.use_anchor_indice=$USE_ANCHOR_INDICE \
    cache.anchor_indice_name="$ANCHOR_INDICE_NAME" \
    cache.anchor_score_name="$ANCHOR_SCORE_NAME" \
    +disable_checkpointing=$DISABLE_CHECKPOINTING \
    dataloader.params.batch_size=$BATCHSIZE \
    lightning.trainer.params.precision=$PRECISION_MODE \
    lightning.trainer.params.limit_val_batches=1.0 \
    cache.cache_path="$CACHE_PATH" \
    cache.scene_tokens_path="$SCENE_TOKENS_PATH" \
    cache.use_manifest=$USE_MANIFEST \
    dataset.val_ratio=$VAL_RATIO \
    dataset.val_split_seed=$VAL_SPLIT_SEED \
    model.intermidiate_dim=$INTERMEDIATE_DIM \
    model.encoder_depth=$ENCODER_DEPTH \
    model.planning_decoder_depths=$PLANNING_DECODER_DEPTHS \
    model.prediction_decoder_depths=$PREDICTION_DECODER_DEPTHS \
    model.future_sampling.num_poses=$NUM_POSE \
    model.planner_anchor_path="$RESULTS_DIR/planner_anchors/$PLANNER_ANCHOR" \
    model.regression_loss_weight=$REGRESS_LOSS_WEIGHT \
    model.regression_yaw_loss_weight=$REGRESS_YAW_LOSS_WEIGHT \
    model.classification_loss_weight=$CLASSY_LOSS_WEIGHT \
    model.train_anchor_num=$TRAIN_ANCHOR_NUM \
    model.test_anchor_num=$TEST_ANCHOR_NUM \
    model.score_chunk_size=$SCORE_CHUNK_SIZE \
    model.use_prediction=$USE_PREDICTION \
    model.prediction_loss_weight=$PREDICTION_LOSS_WEIGHT \
    model.teacher_anchor_ratio=$TEACHER_ANCHOR_RATIO \
    model.teacher_ce_weight=$TEACHER_CE_WEIGHT \
    model.teacher_ce_label_smoothing=$TEACHER_CE_LABEL_SMOOTHING \
    model.anchor_score_kl_weight=$ANCHOR_SCORE_KL_WEIGHT \
    model.anchor_score_neg_loss_weight=$ANCHOR_SCORE_NEG_LOSS_WEIGHT \
    model.use_anchor_score_kl_loss=$USE_ANCHOR_SCORE_KL \
    model.prediction_use_cv_delta=$PREDICTION_USE_CV_DELTA \
    model.debug=false \
    check_invalid_grad=false \




    
