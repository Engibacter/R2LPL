cwd=$(pwd)
export R2LPL_ROOT=${R2LPL_ROOT:-"$cwd"}
export R2LPL_RESULTS_ROOT=${R2LPL_RESULTS_ROOT:-"$cwd/results"}
RESULTS_DIR="$R2LPL_RESULTS_ROOT"
CKPT_ROOT=$RESULTS_DIR/checkpoints
LOG_ROOT=$RESULTS_DIR/simulation_logs
PORT_NUMBER=5006
PLANNER=muvo_abstract_planner_v4
BUILDER=nuplan_test # nuplan_trainval | nuplan_test
FILTER=test14-hard-test # 'val14', 'val14-reduced', 'test14-random', 'test14-hard', 'test14-hard-test'

# FILTER=test14-hard

JOB_NAME=pm_muvo_v4_t4_4096_full_ap_h4s_30_lw_anchor_score_softce02_cyaw_32_fusedeval_M_p1e2c0_lqrdiv_nolatrefine/$FILTER

CHALLENGE=closed_loop_nonreactive_agents
# CHALLENGE=closed_loop_reactive_agents
# CHALLENGE=open_loop_boxes

python ./run/simulation/run_nuboard.py \
    scenario_builder=$BUILDER \
    simulation_path="$LOG_ROOT/$CHALLENGE/$JOB_NAME" \
    port_number=$PORT_NUMBER
    
