from typing import Optional, List, Type, Dict
import torch
from torch.serialization import safe_globals
import numpy as np
import time
from pathlib import Path
import gc
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
    PlannerReport,
)
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.training.modeling.types import FeaturesType

from lpl_planner.model.muvo_planner import MUVOPlanner
from lpl_planner.planning.scene.scene_manager import SceneManager
from lpl_planner.planning.scene.scene_feature.features import (
    SceneFeature,
    AgentPrediction,
)
# from hybrid_planner.planning.planner.frenet_planner import RuleBasePlanner
from lpl_planner.planning.planner.frenet_idm_planner import FrenetIDMPlanner
from lpl_planner.planning.scene.evaluate.scene_scorer import BatchEvaluator
from lpl_planner.planning.scene.evaluate.simulator import (
    BatchSimulator,
    DEFAULT_SIMULATION_DT,
)
from lpl_planner.planning.planner.emergency_brake import EmergencyBrake
from lpl_planner.planning.planner.planner_utils import (
    trajectory_to_interpolated_trajectory,
    interp_valid_yaw,
    hausdorff_xy,
)
from lpl_planner.planning.planner.debug_utils import dump_planner_debug_artifacts
from lpl_planner.planning.planner.oracle_rollout import OracleRolloutBuilder
from lpl_planner.planning.planner.replay_visualization import draw_muvo_replay_frame

PLANNER_NUMPY_DTYPE = np.float32
GLOBAL_COORD_DTYPE = np.float64


class MUVOAbstractPlanner(AbstractPlanner):
    """Abstract planner class for MUMO planner."""
    requires_scenario: bool = True
    def __init__(self, 
                 model: MUVOPlanner,
                 scenario: AbstractScenario = None,
                 ckpt_path: Optional[str] = None,
                 history_sampling = TrajectorySampling(num_poses=30, interval_length=0.1),
                 future_sampling = TrajectorySampling(num_poses=80, interval_length=0.1),
                 save_replay: bool = False,
                 video_dir: Optional[str] = '/results/videos',
                 use_eval: bool = False,
                 num_samples: int = 64,
                 top_k: float = 0.0,
                 top_p: float = 0.0,
                 device: str = "auto",
                 use_hausdorff: bool = False,
                 hausdorff_weight: float = 0.1,
                 debug: bool = False,
                 use_anchor_velocity: bool = False,
                 use_model: bool = True,
                 use_frenet_brake: bool = False,
                 prediction_mode: str = 'prediction', # 'CV', 'CA', 'CYAW', 'prediction'
                 eval_mode: str = 'logits', # 'prob'
                 pred_logprob_weight: float = 1.0,
                 eval_logprob_weight: float = 2.0,
                 eval_score_temperature: float = 1.0,
                 consensus_weight: float = 0.0,
                 consensus_bandwidth: float = 2.0,
                 extra_candidate_logprob_margin: float = 0.0,
                 save_rollout_data: bool = False,
                 debug_output_dir: Optional[str] = None,
                 rollout_cache_dir: Optional[str] = None,
                 rollout_ref_path_num_points: int = 200,
                 oracle_teacher_topk: int = 0,
                 oracle_max_eval_candidates: int = 0,
                 eval_sampling_dt:float = 0.1,
                 replay_image_size_px: int = 2048,
                 ) -> None:
        """
        Initializes the abstract planner.
        :param name: Name of the planner
        """
        
        self._scenario = scenario
        self._planner: MUVOPlanner = model
        self.ckpt_path = ckpt_path
        self._history_sampling = history_sampling
        self._future_sampling = model.future_sampling
        self._use_hausdorff = use_hausdorff
        self._hausdorff_weight = hausdorff_weight
        self._debug = debug
        self._use_anchor_velocity = use_anchor_velocity
        self._use_model = use_model
        self._prediction_mode = prediction_mode
        self._eval_mode = eval_mode
        self._pred_logprob_weight = pred_logprob_weight
        self._eval_logprob_weight = eval_logprob_weight
        self._eval_score_temperature = eval_score_temperature
        self._consensus_weight = consensus_weight
        self._consensus_bandwidth = consensus_bandwidth
        self._extra_candidate_logprob_margin = extra_candidate_logprob_margin
        self.debug_output_dir = Path(debug_output_dir) if debug_output_dir is not None else Path("results/planner_debug")
        self.save_rollout_data = bool(save_rollout_data)
        self.rollout_cache_dir = Path(rollout_cache_dir) if rollout_cache_dir is not None else None
        self.rollout_ref_path_num_points = max(int(rollout_ref_path_num_points), 16)
        self.oracle_teacher_topk = max(int(oracle_teacher_topk), 0)
        self.oracle_max_eval_candidates = max(int(oracle_max_eval_candidates), 0)
        self._eval_sampling_dt = eval_sampling_dt
        self.replay_image_size_px = max(int(replay_image_size_px), 256)
        if self.save_rollout_data:
            if self.rollout_cache_dir is None:
                raise ValueError("rollout_cache_dir must be provided when save_rollout_data=True")
            self.rollout_cache_dir.mkdir(parents=True, exist_ok=True)
        if self._debug:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)

        if device == "auto":
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                self.device = torch.device(f"cuda:{idx}")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index or 0)

        self._scene_manager: Optional[SceneManager] = SceneManager(planning_step=future_sampling.num_poses,
                                                                    time_step=self._future_sampling.interval_length,)
        self._rule_based_planner = FrenetIDMPlanner(trajectory_sampling=self._future_sampling,
                                                    debug=debug,)

        self.save_replay = save_replay
        self.video_dir = Path(video_dir)
        self._video_writer = None
        self._video_output_path: Optional[Path] = None
        if self.save_replay:
            assert self.video_dir is not None, "video_dir must be provided if save_replay is True"
            self.video_dir.mkdir(parents=True, exist_ok=True)
        self.use_eval = use_eval
        self.num_samples = num_samples
        self.top_k = top_k
        self.top_p = top_p

        # utils for measuring runtime
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []
        self._evaluation_runtimes: List[float] = []
        self.scenario_all_iteration: Optional[np.ndarray] = None
        self.scenario_all_timepoints: Optional[np.ndarray] = None

        # utils for fallback trajectory
        self.fallback_trajectory: Optional[AbstractTrajectory] = None
        self.emergency_brake_planner = EmergencyBrake(trajectory_sampling=self._future_sampling,
                                                      infraction='collision',
                                                      time_to_infraction_threshold=2.0,
                                                      use_frenet=use_frenet_brake)

    def _close_video_writer(self) -> None:
        writer = self._video_writer
        self._video_writer = None
        if writer is not None:
            writer.close()

    def _build_video_output_path(self) -> Path:
        list_of_modifications = {
            "a": "amount_of_agents",
            "d": "density",
            "g": "goal",
            "o": "observation",
            "s": "special_scenario",
        }

        modification = getattr(self._scenario, "modification", None)
        suffix = ""
        if modification is not None:
            for letter, name in list_of_modifications.items():
                if name in modification:
                    if name == "special_scenario":
                        suffix = f"s{modification['special_scenario']}"
                    else:
                        suffix += f"{letter}{modification[name]}"

        return self.video_dir / f"{self._scenario.log_name}_{self._scenario.token}{suffix}.mp4"

    def _append_video_frame(self, img: np.ndarray) -> None:
        if not self.save_replay:
            return

        if self._video_writer is None:
            import imageio

            self._video_output_path = self._build_video_output_path()
            self._video_writer = imageio.get_writer(self._video_output_path, fps=10)

        self._video_writer.append_data(img)

    def __getstate__(self):
        """Drop non-picklable runtime handles before simulation_log serialization."""
        self._close_video_writer()
        state = self.__dict__.copy()
        state["_video_writer"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_video_writer" not in self.__dict__:
            self._video_writer = None

    def _softmax_normalize(self, values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Normalize scene-local scores with softmax; all-zero inputs become uniform."""
        values = np.asarray(values, dtype=PLANNER_NUMPY_DTYPE)
        if values.size == 0:
            return values

        finite_mask = np.isfinite(values)
        if not finite_mask.any():
            return np.full(values.shape, 1.0 / values.size, dtype=PLANNER_NUMPY_DTYPE)

        safe_values = values.copy()
        fill_value = np.min(safe_values[finite_mask])
        safe_values[~finite_mask] = fill_value
        safe_values /= max(temperature, 1e-6)
        safe_values -= np.max(safe_values)
        probs = np.exp(safe_values)
        probs = np.where(np.isfinite(probs), probs, 0.0)
        probs_sum = probs.sum()
        if probs_sum <= 0:
            return np.full(values.shape, 1.0 / values.size, dtype=PLANNER_NUMPY_DTYPE)
        return probs / probs_sum

    def _expand_model_log_scores(
        self,
        model_log_scores: Optional[np.ndarray],
        target_num_candidates: int,
    ) -> np.ndarray:
        """Extend model log-probabilities to non-model candidates such as fallback proposals."""
        if target_num_candidates <= 0:
            return np.zeros((0,), dtype=PLANNER_NUMPY_DTYPE)

        if model_log_scores is None or len(model_log_scores) == 0:
            return np.zeros((target_num_candidates,), dtype=PLANNER_NUMPY_DTYPE)

        model_log_scores = np.asarray(model_log_scores, dtype=PLANNER_NUMPY_DTYPE).reshape(-1)
        if model_log_scores.shape[0] >= target_num_candidates:
            return model_log_scores[:target_num_candidates]

        pad_value = np.min(model_log_scores) - self._extra_candidate_logprob_margin
        return np.pad(
            model_log_scores,
            (0, target_num_candidates - model_log_scores.shape[0]),
            constant_values=pad_value,
        )

    def _compute_consensus_scores(
        self,
        trajectories: np.ndarray,
        model_log_scores: np.ndarray,
    ) -> np.ndarray:
        """Reward proposals supported by nearby high-probability samples from other anchors."""
        num_candidates = trajectories.shape[0]
        if num_candidates <= 1 or self._consensus_weight <= 0:
            return np.zeros((num_candidates,), dtype=PLANNER_NUMPY_DTYPE)

        centered_log_scores = model_log_scores - np.max(model_log_scores)
        model_probs = np.exp(centered_log_scores)
        model_probs /= np.sum(model_probs).clip(min=1e-12)

        pairwise_distance = np.zeros((num_candidates, num_candidates), dtype=PLANNER_NUMPY_DTYPE)
        for src_idx in range(num_candidates):
            pairwise_distance[src_idx, src_idx] = 0.0
            for dst_idx in range(src_idx + 1, num_candidates):
                distance_ij = float(hausdorff_xy(trajectories[src_idx], trajectories[dst_idx]))
                pairwise_distance[src_idx, dst_idx] = distance_ij
                pairwise_distance[dst_idx, src_idx] = distance_ij

        bandwidth = max(self._consensus_bandwidth, 1e-3)
        kernel = np.exp(-(pairwise_distance ** 2) / (bandwidth ** 2))
        consensus_mass = kernel @ model_probs
        consensus_scores = np.log(consensus_mass.clip(min=1e-12))
        consensus_scores -= np.max(consensus_scores)
        return consensus_scores

    def _compute_selection_scores(
        self,
        trajectories: np.ndarray,
        trajectory_score: Optional[np.ndarray],
        model_log_scores: Optional[np.ndarray],
    ) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        """Combine model prior, evaluator ranking, and optional consensus into final selection scores."""
        num_candidates = trajectories.shape[0]
        model_log_scores_full = self._expand_model_log_scores(model_log_scores, num_candidates)

        if trajectory_score is not None and len(trajectory_score) == num_candidates:
            eval_probs = self._softmax_normalize(
                np.asarray(trajectory_score, dtype=np.float64),
                
                temperature=self._eval_score_temperature,
            )
            eval_log_scores = np.log(eval_probs.clip(min=1e-12))
        else:
            eval_probs = None
            eval_log_scores = np.zeros((num_candidates,), dtype=PLANNER_NUMPY_DTYPE)

        consensus_scores = self._compute_consensus_scores(trajectories, model_log_scores_full)
        selection_scores = (
            self._pred_logprob_weight * model_log_scores_full
            + self._eval_logprob_weight * eval_log_scores
            + self._consensus_weight * consensus_scores
        )
        return selection_scores, eval_probs, model_log_scores_full, consensus_scores

    def reset(self) -> None:
        """Reset internal states of the planner."""

        if self.save_replay:
            self._close_video_writer()
            self._video_output_path = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        
         # load model weights from checkpoint
        assert self.ckpt_path is not None, "Checkpoint path must be provided to load model weights."
        if self._use_model:
            with safe_globals([TrajectorySampling]):
                try:
                    ckpt = torch.load(self.ckpt_path, map_location=torch.device("cpu"), weights_only=True)
                except Exception:
                    ckpt = torch.load(self.ckpt_path, map_location=torch.device("cpu"), weights_only=False)
            
            state = ckpt.get("state_dict", ckpt)

            def strip_prefix(state_dict, prefix):
                plen = len(prefix)
                return {k[plen:]: v for k, v in state_dict.items() if k.startswith(prefix)}

            sub_state = {}
            for p in ['model.', 'policy.']:
                stripped_state = strip_prefix(state, p)
                if len(stripped_state) > 0:
                    prefix = p
                    sub_state = stripped_state
                    break
            if not sub_state:
                raise KeyError(f"No matching keys for prefixes {['model.', 'policy.']}. Available top-level keys (sample): "
                            f"{list(state.keys())[:5]}"
                            )
            
            state_dict = {k.replace(prefix, ""): v for k, v in sub_state.items()}
            missing, unexpected = self._planner.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"Strict load ckpt failed. Missing: {missing}, Unexpected: {unexpected}")
            self._planner.eval()
            self._planner = self._planner.to(self.device)
            
            torch.set_grad_enabled(False)
            del state, sub_state, state_dict

        self._initialization = initialization


        self._scene_manager.init_from_planner_init(initialization)

        if self.use_eval:
            self._simulator = BatchSimulator(
                self._future_sampling,
                default_dt=DEFAULT_SIMULATION_DT,
            )
            self._evaluator = BatchEvaluator(
                self._future_sampling,
                default_dt=DEFAULT_SIMULATION_DT,
                use_following_penalty=True,
                trajectory_sample_dt=self._eval_sampling_dt,
            )

        # reset fallback trajectory
        self.fallback_trajectory: Optional[AbstractTrajectory] = None

        self.skip_eval: bool = False

        self._close_video_writer()
        self._video_output_path = None
        self._frame_idx: int = 0
        self._iteration = 0
        self._ego_history_global: List[np.ndarray] = []
        self._simulation_start_time_us: Optional[int] = None

        # store expert_trajectory for replay and rollout
        self.expert_trajectory: Optional[AbstractTrajectory] = self._extract_expert_trajectory()

        if self.save_rollout_data:
            rollout_simulator = self._simulator if self.use_eval else BatchSimulator(
                self._future_sampling,
                default_dt=DEFAULT_SIMULATION_DT,
            )
            rollout_evaluator = self._evaluator if self.use_eval else BatchEvaluator(
                self._future_sampling,
                default_dt=DEFAULT_SIMULATION_DT,
            )
            self._rollout_worker = OracleRolloutBuilder(
                scenario=self._scenario,
                future_sampling=self._future_sampling,
                scene_manager=self._scene_manager,
                simulator=rollout_simulator,
                evaluator=rollout_evaluator,
                rollout_cache_dir=self.rollout_cache_dir,
                ref_path_num_points=self.rollout_ref_path_num_points,
                topk=self.oracle_teacher_topk,
                max_eval_candidates=self.oracle_max_eval_candidates,
                num_samples=self.num_samples,
                planner_model=self._planner,
                planner_device=self.device,
                pred_logprob_weight=self._pred_logprob_weight,
                eval_logprob_weight=self._eval_logprob_weight,
                eval_score_temperature=self._eval_score_temperature,
            )


        # utils for measuring runtime
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []
        self._evaluation_runtimes: List[float] = []

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__
    
    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore
    

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        """
        Infer relative trajectory poses from model and convert to absolute agent states wrapped in a trajectory.
        Inherited, see superclass.
        """
        ego_state: EgoState = current_input.history.ego_states[-1]
        self._record_ego_history_for_replay(ego_state)

        if self._iteration == 0:
            self._scene_manager.lane_map.route_correction(ego_state, self._scenario)
            
        start_time = time.perf_counter()
        # step scene manager
        self._scene_manager.step_with_planner_input(ego_state)
        
        # find current iteration in nuplan scenario
        current_timepoint_us = ego_state.time_point.time_us
        neareast_iteration_idx = np.argmin(np.abs(self.scenario_all_timepoints - current_timepoint_us))
        self._scenario_iteration = self.scenario_all_iteration[neareast_iteration_idx]

        # extract features
        if self.save_rollout_data:
            scene_feature, agent_prediction_gt = self._scene_manager.extract_feature_target_from_scenario(
                self._scenario,
                iteration=self._scenario_iteration,
                ego_state=ego_state,
                ego_history=current_input.history.ego_states,
            )
            scene_feature = SceneFeature.deserialize(scene_feature)
            agent_prediction_gt = AgentPrediction.deserialize(agent_prediction_gt)
        else:
            scene_feature = SceneFeature.deserialize(self._scene_manager.extract_feature_from_simulation(current_input))
        scene_feature_raw = scene_feature
        feature_building_time = time.perf_counter() - start_time
        
        
        # prepare model input
        scene_feature = scene_feature.collate([scene_feature.to_feature_tensor()]).to_device(self.device)
        model_input: FeaturesType = {"scene_feature": scene_feature}
        self._feature_building_runtimes.append(feature_building_time)
        
        # model inference
        if self._use_model:
            with torch.no_grad():
                inference_start_time = time.perf_counter()
                # model_output = self._planner(model_input)
                model_output = self._planner.sample_trajectories(
                                                                features=model_input,
                                                                num_samples=self.num_samples,
                                                                top_k = self.top_k,
                                                                top_p = self.top_p,
                                                                )
                inference_time = time.perf_counter() - inference_start_time
                self._inference_runtimes.append(inference_time)

            # process model output
            trajectories = model_output["trajectories"].cpu().numpy()[0]  # (S, T, 3)
            indices = model_output["indices"].cpu().numpy()[0].reshape(-1)  # (S,)
            scores_pred = model_output["scores"].cpu().numpy()[0].reshape(-1)  # (S,)
            all_model_logp = (
                model_output["all_logp"].detach().cpu().numpy()[0].reshape(-1)
                if "all_logp" in model_output
                else None
            )
            agent_prediction = model_output.get("agent_prediction", None)
            if agent_prediction is not None:
                agent_prediction = agent_prediction.unpack()[0].to_device(torch.device("cpu"))

        else:
            trajectories = np.zeros((0, self._future_sampling.num_poses, 6), dtype=np.float32)
            agent_prediction = None
            indices = np.zeros((0,), dtype=np.int32)
            scores_pred = None
            all_model_logp = None
        rollout_policy_trajectories = np.asarray(trajectories, dtype=np.float32).copy()
        rollout_policy_indices = np.asarray(indices, dtype=np.int32).reshape(-1)
        rollout_policy_model_scores = np.asarray(scores_pred, dtype=np.float32).reshape(-1) if scores_pred is not None else np.zeros((0,), dtype=np.float32)
        scene_feature = scene_feature.unpack()[0].to_device(torch.device("cpu"))

        # evaluate trajectories
        
        trajectory_score = None
        eval_score_probs = None
        fallback_trajectory_local = None
        collision_times = None
        ttc_times = None
        consensus_scores = None
        model_log_scores = None
        evaluation_start_time = time.perf_counter()
        if self.use_eval:
            # compute boundary trajectory
            # boundary_trajectory = self.compute_boundary_trajectory(trajectories[0], ego_state)
            if self._debug:
                print("Computing rule-based trajectories as boundary trajectories.")
                t0 = time.perf_counter()
            # rule_based_trajectories = self.compute_rule_based_trajectory(ego_state)
            # rule_based_trajectories = self.compute_boundary_trajectory(trajectories[0], ego_state)
            # rule_based_trajectories = self.compute_boundary_idm_trajectory(
            #     trajectories=trajectories, ego_state=ego_state, scene_feature=scene_feature
            #     )
            
            if self._debug:
                t1 = time.perf_counter()
                print(f"Rule-based trajectory computation time: {t1 - t0:.4f} s")

            # # append boundary trajectory to candidate trajectories
            # trajectories = np.concatenate((trajectories, rule_based_trajectories), axis=0)

            # # smooth model trajectories
            # trajectories = smooth_trajectories(
            #     trajectories=trajectories,
            #     dt=self._future_sampling.interval_length,
            #     init_speed=ego_state.dynamic_car_state.speed,
            #     init_accel=ego_state.dynamic_car_state.acceleration,
            # )

            # stop trajectory for safety evaluation
            stop_trajectory = np.zeros((1, self._future_sampling.num_poses, 6), dtype=np.float32)
            if stop_trajectory not in trajectories:
                trajectories = np.concatenate((trajectories, stop_trajectory), axis=0)
                
            # eval fallback trajectory and prepend to candidate trajectories
            if (self.fallback_trajectory is not None) and \
            (self.fallback_trajectory.end_time.time_s > ego_state.time_point.time_s + 1.0):
                if self._debug:
                    print("Adding fallback trajectory as candidate.")
                fallback_trajectory_local = self.compute_fallback_trajectory(ego_state, self.fallback_trajectory)
                trajectories = np.concatenate((trajectories, fallback_trajectory_local[np.newaxis, ...] ), axis=0)
            
            # simulate and evaluate all trajectories
            current_state = np.zeros((trajectories.shape[0], 1, trajectories.shape[-1]), dtype=np.float32)
            current_state[:, 0, 3] = ego_state.dynamic_car_state.speed  # set current speed
            current_state[:, 0, 5] = ego_state.dynamic_car_state.angular_velocity  # set current yaw rate
            current_state[:, 0, 4] = ego_state.dynamic_car_state.acceleration  # set current acceleration
            extended_trajectories = np.concatenate((current_state, trajectories), axis=1)
            if self._debug:
                print(f'Number of trajectories to evaluate: {extended_trajectories.shape[0]}')
                print('starting simulation and evaluation...')
            simulated_trajectories = self._simulator.simulate(
                extended_trajectories,
                ego_state=ego_state,
            )
            scores_eval = self._evaluator.batch_evaluate(
                simulated_trajectories,
                scene_feature=scene_feature,
                agent_prediction=agent_prediction,
                discount_factor=1.0,
                ref_path=self._scene_manager.lane_map.ref_path,
                prediction_mode=self._prediction_mode,
                # scene_manager=self._scene_manager,
            )  # (S)
            trajectory_score = np.array(scores_eval['aggregate_scores'])
            min_time_to_collision = scores_eval['min_time_to_collision']
            collision_times = scores_eval.get("collision_times")
            ttc_times = scores_eval.get("ttc_times")

            

        else:
            # trajectories = smooth_trajectories(
            #     trajectories=trajectories,
            #     dt=self._future_sampling.interval_length,
            #     init_speed=ego_state.dynamic_car_state.speed,
            #     init_accel=ego_state.dynamic_car_state.acceleration,
            # )
            pass
            
        
        selection_scores, eval_score_probs, model_log_scores, consensus_scores = self._compute_selection_scores(
            trajectories=trajectories,
            trajectory_score=trajectory_score if self.use_eval else None,
            model_log_scores=scores_pred,
        )
        scores = selection_scores
        best_candidate_idx = int(np.argmax(scores)) if scores.size > 0 else -1
        rollout_policy_eval_scores = np.asarray(trajectory_score[: rollout_policy_indices.shape[0]], dtype=np.float32) if trajectory_score is not None else np.zeros((rollout_policy_indices.shape[0],), dtype=np.float32)
        rollout_policy_selection_scores = np.asarray(scores[: rollout_policy_indices.shape[0]], dtype=np.float32)

        if self.use_eval and trajectory_score is not None and np.allclose(trajectory_score, 0.0):
            if self._debug:
                print("Evaluation scores are all zero. Selection falls back to model probability prior.")
        elif not self.use_eval and self._debug:
            print("No evaluator available. Using model prediction scores only.")

        evaluation_time = time.perf_counter() - evaluation_start_time
        self._evaluation_runtimes.append(evaluation_time)
     
        # convert relative trajectories to absolute trajectories
        ego_xy = np.asarray(ego_state.rear_axle.array, dtype=GLOBAL_COORD_DTYPE)
        ego_yaw = ego_state.rear_axle.heading
        rot_mat = np.array([[np.cos(ego_yaw), -np.sin(ego_yaw)],
                    [np.sin(ego_yaw),  np.cos(ego_yaw)]], dtype=GLOBAL_COORD_DTYPE)
        trajectory_global = np.asarray(trajectories, dtype=GLOBAL_COORD_DTYPE).copy()
        trajectory_global[..., :2] = trajectory_global[..., :2] @ rot_mat.T + ego_xy
        trajectory_global[..., 2] = trajectory_global[..., 2] + ego_yaw
        trajectory_global[..., 2] = np.arctan2(np.sin(trajectory_global[..., 2]), np.cos(trajectory_global[..., 2]))

        if self.use_eval:
            emergency_break_trajectory = self.emergency_brake_planner.brake_if_emergency(
                ego_state=ego_state,
                scores=scores,
                scene_manager=self._scene_manager,
                collision_times=collision_times,
                ttc_times=ttc_times,
            )
        else:
            emergency_break_trajectory = None
            
        if emergency_break_trajectory is not None:
            if self._debug:
                print("Emergency brake trajectory selected!")
            trajectory = emergency_break_trajectory
            chosen_local_index = -1
            chosen_anchor_index = -1
            executed_candidate_idx = -1
            if rollout_policy_selection_scores.size > 0:
                chosen_score = float(np.min(rollout_policy_selection_scores) - 1.0)
            else:
                chosen_score = 0.0
            emergency_brake_flag = True
        else:
            best_idx = best_candidate_idx
            chosen_local_index = best_idx if best_idx < rollout_policy_indices.shape[0] else -1
            chosen_anchor_index = int(rollout_policy_indices[chosen_local_index]) if chosen_local_index >= 0 else -1
            chosen_score = float(rollout_policy_selection_scores[chosen_local_index]) if chosen_local_index >= 0 else 0.0
            executed_candidate_idx = best_idx
            emergency_brake_flag = False
            if self.use_eval and trajectory_score is not None:
                if self._debug:
                    print(f'current frame: {self._frame_idx}')
                    print(f'Best trajectory index: {best_idx}, Score: {scores[best_idx]:.2f}')
                    print(f'trajectory scores of best trajectory: {trajectory_score[best_idx]}')
                    if eval_score_probs is not None:
                        print(f'eval softmax prob of best trajectory: {eval_score_probs[best_idx]:.4f}')
                    if model_log_scores is not None:
                        print(f'model log prob of best trajectory: {model_log_scores[best_idx]:.4f}')
                    if consensus_scores is not None and self._consensus_weight > 0:
                        print(f'consensus score of best trajectory: {consensus_scores[best_idx]:.4f}')
                    weighted_metrics = np.array(scores_eval["weighted_metrics"])
                    multi_metrics = np.array(scores_eval["multi_metrics"])
                    print(f'weighted scores of best trajectory: {np.array2string(weighted_metrics[:, best_idx], formatter={"float_kind":lambda x: f"{x:.2f}"})}')
                    print(f'multiplicative scores of best trajectory: {np.array2string(multi_metrics[2, best_idx], formatter={"float_kind":lambda x: f"{x:.2f}"})}')
                    print(f"all trajectory scores: {np.array2string(trajectory_score, formatter={'float_kind':lambda x: f'{x:.2f}'})}")
                    if eval_score_probs is not None:
                        print(f"all eval softmax probs: {np.array2string(eval_score_probs, formatter={'float_kind':lambda x: f'{x:.4f}'})}")
                    if model_log_scores is not None:
                        print(f"all model log probs: {np.array2string(model_log_scores, formatter={'float_kind':lambda x: f'{x:.2f}'})}")
                    if consensus_scores is not None and self._consensus_weight > 0:
                        print(f"all consensus scores: {np.array2string(consensus_scores, formatter={'float_kind':lambda x: f'{x:.2f}'})}")
                    print(f"all multiplicative scores: {np.array2string(multi_metrics, formatter={'float_kind':lambda x: f'{x:.2f}'})}")
            else:
                if self._debug:
                    print("Using fused selection scores without evaluator contribution.")

            best_trajectory = trajectory_global[best_idx]
            
            trajectory = trajectory_to_interpolated_trajectory(
                trajectory=best_trajectory,
                ego_history=current_input.history.ego_states,
                future_horizon=self._future_sampling.time_horizon,
                step_interval=self._future_sampling.interval_length,
                use_anchor_velocity=False,
                debug=self._debug,
            )
            self.fallback_trajectory = trajectory

        if self._debug:
            candidate_anchor_indices = np.full((trajectories.shape[0],), -1, dtype=np.int32)
            candidate_anchor_indices[: rollout_policy_indices.shape[0]] = rollout_policy_indices
            chosen_trajectory_local = self.compute_fallback_trajectory(ego_state, trajectory)
            expert_local_trajectory = self.compute_fallback_trajectory(ego_state, self.expert_trajectory)
            dump_planner_debug_artifacts(
                output_root=self.debug_output_dir,
                scenario_token=self._scenario.token,
                iteration=self._iteration,
                scene_manager=self._scene_manager,
                scene_feature=scene_feature,
                trajectories=trajectories,
                anchor_indices=candidate_anchor_indices,
                selection_scores=scores,
                prediction_mode=self._prediction_mode,
                agent_prediction=agent_prediction,
                chosen_trajectory=chosen_trajectory_local,
                expert_trajectory=expert_local_trajectory,
                model_scores=model_log_scores,
                eval_scores=trajectory_score,
                executed_candidate_idx=executed_candidate_idx,
                best_candidate_idx=best_candidate_idx,
                eval_details=scores_eval if self.use_eval else None,
                extra_metadata={
                    "scenario_token": self._scenario.token,
                    "chosen_anchor_index": int(chosen_anchor_index),
                    "chosen_score": float(chosen_score),
                    "emergency_brake": bool(emergency_brake_flag),
                    "eval_score_probs": None if eval_score_probs is None else np.asarray(eval_score_probs, dtype=np.float32),
                    "consensus_scores": None if consensus_scores is None else np.asarray(consensus_scores, dtype=np.float32),
                },
            )

        

        ## DEBUGGING CODE TO CHECK TRANSFORMATIONS
        # 放在变换后做一次反变换检查
        # print(f'current frame: {self._frame_idx}')
        # print(f'Best trajectory index: {best_idx}, Score: {scores[best_idx]:.4f}')
        if self._debug:
            R = rot_mat; Rt = R.T
            timestamp = current_input.history.ego_states[-1].time_point.time_s + self._future_sampling.interval_length
            next_step_time = TimePoint(int(timestamp * 1e6))
            p_next: EgoState = trajectory.get_state_at_time(next_step_time)
            p_rel = p_next.rear_axle.array - ego_xy
            p_rel_back = Rt @ (p_rel.reshape(2,1))
            yaw_rel = p_next.rear_axle.heading - ego_yaw
            # print(f'Check relative pos x: {p_rel_back[0,0]:.4f} m, y: {p_rel_back[1,0]:.4f} m, yaw: {yaw_rel:.4f} rad')
            best_idx = np.argmax(scores)
            print(f'Choosed trajectory index: {best_idx}, Score: {scores[best_idx]:.4f}')
            print(f'current_speed: {ego_state.dynamic_car_state.speed:.4f} m/s')
            print(f'current_accel: {ego_state.dynamic_car_state.acceleration:.4f} m/s²')
            traj_str = np.array2string(
                trajectories[best_idx],
                formatter={"float_kind": lambda x: f"{x:.2f}"}
            )
            print(f"chosen trajectory:\n{traj_str}")
            # print(f'Chosen trajectory: {trajectories[best_idx]}')
            

        oracle_rollout = None
        if self.save_rollout_data:
            oracle_rollout = self._build_rollout_oracle(
                scene_feature_tensor=model_input["scene_feature"],
                scene_feature=scene_feature_raw,
                ego_state=ego_state,
                agent_prediction=agent_prediction_gt,
                rollout_anchor_index=chosen_anchor_index,
                rollout_score=chosen_score,
                rollout_model_logp=all_model_logp,
                rollout_policy_indices=rollout_policy_indices,
                rollout_policy_fused_scores=rollout_policy_selection_scores,
            )
            self._save_rollout_sample(
                scene_feature=scene_feature_raw,
                agent_prediction=agent_prediction_gt,
                ego_state=ego_state,
                sampled_trajectories=oracle_rollout["candidate_trajectories"],
                sampled_anchor_indices=oracle_rollout["candidate_anchor_indices"],
                sampled_teacher_sources=oracle_rollout["candidate_teacher_sources"],
                sampled_model_log_scores=oracle_rollout["candidate_model_log_scores"],
                sampled_eval_scores=oracle_rollout["candidate_eval_scores"],
                sampled_selection_scores=oracle_rollout["candidate_selection_scores"],
                chosen_local_index=chosen_local_index,
                chosen_anchor_index=chosen_anchor_index,
                chosen_score=chosen_score,
                emergency_brake=emergency_brake_flag,
                expert_path_local=oracle_rollout["expert_path_local"],
                expert_route_ref_path=oracle_rollout["expert_route_ref_path"],
            )

        # draw scene and trajectories
        if self.save_replay:
            chosen_trajectory = np.asarray([
                ego_state.rear_axle.array for ego_state in trajectory.get_sampled_trajectory()
            ], dtype=GLOBAL_COORD_DTYPE)
            chosen_trajectory = self._project_xy_to_ego_local(chosen_trajectory, ego_state)
            expert_future_trajectory_local = self.compute_fallback_trajectory(ego_state, self.expert_trajectory)
            ego_history_trajectory_local = self._project_replay_history_to_ego_local(ego_state)
            expert_history_trajectory_local = self._sample_expert_history_local(ego_state)
            expert_current_pose_local = self._sample_expert_current_pose_local(ego_state)

            img = draw_muvo_replay_frame(
                scene_feature=scene_feature,
                chosen_trajectory=chosen_trajectory,
                all_trajectories=trajectories,
                all_trajectory_scores=scores,
                agent_prediction=agent_prediction,
                prediction_mode=self._prediction_mode,
                expert_future_trajectory=expert_future_trajectory_local,
                ego_history_trajectory=ego_history_trajectory_local,
                expert_history_trajectory=expert_history_trajectory_local,
                expert_current_pose=expert_current_pose_local,
                rollout_subset_trajectories=None if oracle_rollout is None else oracle_rollout["candidate_trajectories"],
                rollout_subset_scores=None if oracle_rollout is None else oracle_rollout["candidate_selection_scores"],
                rollout_anchor_indices=None if oracle_rollout is None else oracle_rollout["candidate_anchor_indices"],
                rollout_teacher_sources=None if oracle_rollout is None else oracle_rollout["candidate_teacher_sources"],
                expert_path_local=None if oracle_rollout is None else oracle_rollout["expert_path_local"],
                expert_route_ref_path=None if oracle_rollout is None else oracle_rollout["expert_route_ref_path"],
                image_size_px=self.replay_image_size_px,
            )

            self._append_video_frame(img)
        
        self._frame_idx += 1
        self._iteration += 1
        return trajectory
    
    def compute_rule_based_trajectory(
        self, 
        ego_state: EgoState,
    ) -> np.ndarray:
        """ compute rule-based trajectory for muvo planner"""
        # use frenet rule-based planner to compute greedy forward trajectory
        rule_trajectories = self._rule_based_planner.calculate_frenet_paths(
            ego_state=ego_state,
            scene_manager=self._scene_manager,
        )  # (N_paths, T+1, 3)
        trajectories = np.zeros((rule_trajectories.shape[0], self._future_sampling.num_poses, 6), dtype=np.float32)
        trajectories[:, :, :3] = rule_trajectories[:, 1:, :3]  # skip first point
        return trajectories
    
    def compute_boundary_idm_trajectory(
            self,
            trajectories: np.ndarray,
            ego_state: EgoState,
            scene_feature: SceneFeature,
            ) -> np.ndarray:
        """ compute boundary trajectory using IDM model for muvo planner"""
        
        return self._rule_based_planner.plan(
            ego_state=ego_state,
            scene_feature=scene_feature,
            scenario_manager=self._scene_manager,
            trajectories=trajectories,
            )
        
    def _extract_expert_trajectory(self):

        expert_traj = self._scenario.get_expert_ego_trajectory()
        ego_traj = [ego_taj_state for ego_taj_state in expert_traj]
        scenario_iteration = [iteration for iteration in range(self._scenario.get_number_of_iterations())]
        scenario_timepoints = [ego_state.time_point.time_us for ego_state in ego_traj]
        
        expert_past = self._scenario.get_ego_past_trajectory(
            iteration=0,
            time_horizon=1.0,
            num_samples=10
            )
        ego_past_traj = [ego_past_state for ego_past_state in expert_past]
        scenario_past_iteration = [-1 for _ in range(len(ego_past_traj))]
        scenario_past_timepoints = [ego_state.time_point.time_us for ego_state in ego_past_traj]
        
        expert_future_traj = self._scenario.get_ego_future_trajectory(
            iteration=self._scenario.get_number_of_iterations() - 1,
            time_horizon=4.0,
            num_samples=40
        )
        ego_fut_traj = [ego_future_state for ego_future_state in expert_future_traj]
        scenario_future_iteration = [-1 for _ in range(len(ego_fut_traj))]
        scenario_future_timepoints = [ego_state.time_point.time_us for ego_state in ego_fut_traj]

        ego_traj = ego_past_traj + ego_traj + ego_fut_traj
        ego_traj = InterpolatedTrajectory(ego_traj)

        scenario_all_iteration = scenario_past_iteration + scenario_iteration + scenario_future_iteration
        scenario_all_timepoints = scenario_past_timepoints + scenario_timepoints + scenario_future_timepoints

        self.scenario_all_iteration = np.array(scenario_all_iteration)
        self.scenario_all_timepoints = np.array(scenario_all_timepoints)
        
            
        return ego_traj

    def _record_ego_history_for_replay(self, ego_state: EgoState) -> None:
        if self._simulation_start_time_us is None:
            self._simulation_start_time_us = int(ego_state.time_point.time_us)

        point = np.asarray(ego_state.rear_axle.array, dtype=GLOBAL_COORD_DTYPE)
        if self._ego_history_global:
            previous = self._ego_history_global[-1]
            if np.linalg.norm(point - previous) < 1e-6:
                return
        self._ego_history_global.append(point)

    def _project_xy_to_ego_local(self, xy_global: np.ndarray, ego_state: EgoState) -> np.ndarray:
        xy_global = np.asarray(xy_global, dtype=GLOBAL_COORD_DTYPE)
        if xy_global.shape[0] == 0:
            return np.zeros((0, 2), dtype=GLOBAL_COORD_DTYPE)

        ego_xy = np.asarray(ego_state.rear_axle.array, dtype=GLOBAL_COORD_DTYPE)
        ego_yaw = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(ego_yaw), -np.sin(ego_yaw)], [np.sin(ego_yaw), np.cos(ego_yaw)]],
            dtype=GLOBAL_COORD_DTYPE,
        )
        return (rot_mat.T @ (xy_global[:, :2] - ego_xy).T).T

    def _project_replay_history_to_ego_local(self, ego_state: EgoState) -> np.ndarray:
        if not self._ego_history_global:
            return np.zeros((0, 2), dtype=GLOBAL_COORD_DTYPE)
        ego_history_global = np.vstack(self._ego_history_global)
        return self._project_xy_to_ego_local(ego_history_global, ego_state)

    def _sample_expert_history_local(self, ego_state: EgoState) -> np.ndarray:
        if self.expert_trajectory is None:
            return np.zeros((0, 2), dtype=GLOBAL_COORD_DTYPE)

        start_time_us = self._simulation_start_time_us
        if start_time_us is None:
            start_time_us = int(ego_state.time_point.time_us)
        start_time_us = max(start_time_us, int(self.expert_trajectory.start_time.time_us))
        end_time_us = min(int(ego_state.time_point.time_us), int(self.expert_trajectory.end_time.time_us))
        if end_time_us < start_time_us:
            return np.zeros((0, 2), dtype=GLOBAL_COORD_DTYPE)

        dt_us = int(max(self._future_sampling.interval_length, 1e-3) * 1e6)
        sample_times_us = np.arange(start_time_us, end_time_us + 1, dt_us, dtype=np.int64)
        if sample_times_us.size == 0 or sample_times_us[-1] != end_time_us:
            sample_times_us = np.concatenate([sample_times_us, np.asarray([end_time_us], dtype=np.int64)])
        states = self.expert_trajectory.get_state_at_times([TimePoint(int(t)) for t in sample_times_us])
        expert_history_global = np.asarray(
            [state.rear_axle.array for state in states],
            dtype=GLOBAL_COORD_DTYPE,
        )
        return self._project_xy_to_ego_local(expert_history_global, ego_state)

    def _sample_expert_current_pose_local(self, ego_state: EgoState) -> Optional[np.ndarray]:
        if self.expert_trajectory is None:
            return None

        query_time_us = int(np.clip(
            ego_state.time_point.time_us,
            self.expert_trajectory.start_time.time_us,
            self.expert_trajectory.end_time.time_us,
        ))
        expert_state = self.expert_trajectory.get_state_at_time(TimePoint(query_time_us))
        expert_pose_global = np.asarray(
            [
                expert_state.rear_axle.x,
                expert_state.rear_axle.y,
                expert_state.rear_axle.heading,
            ],
            dtype=GLOBAL_COORD_DTYPE,
        )
        expert_pose_local = np.zeros((3,), dtype=GLOBAL_COORD_DTYPE)
        expert_pose_local[:2] = self._project_xy_to_ego_local(expert_pose_global[None, :2], ego_state)[0]
        expert_pose_local[2] = np.arctan2(
            np.sin(expert_pose_global[2] - ego_state.rear_axle.heading),
            np.cos(expert_pose_global[2] - ego_state.rear_axle.heading),
        )
        return expert_pose_local

    def _build_rollout_oracle(
        self,
        scene_feature_tensor: SceneFeature,
        scene_feature: SceneFeature,
        ego_state: EgoState,
        agent_prediction: Optional[AgentPrediction],
        rollout_anchor_index: int,
        rollout_score: float,
        rollout_model_logp: Optional[np.ndarray],
        rollout_policy_indices: np.ndarray,
        rollout_policy_fused_scores: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        expert_path = self._extract_rollout_expert_path()
        expert_path_local = self._project_path_to_ego_local(expert_path, ego_state)
        route_ref_path = self._scene_manager.lane_map.get_ref_path_feature(
            ref_path_num_points=self.rollout_ref_path_num_points,
        )
        route_ref_path = np.asarray(route_ref_path[:, :3], dtype=GLOBAL_COORD_DTYPE)
        return self._rollout_worker.build_rollout_oracle(
            scene_feature_tensor=scene_feature_tensor,
            scene_feature=scene_feature,
            ego_state=ego_state,
            agent_prediction=agent_prediction,
            rollout_anchor_index=rollout_anchor_index,
            rollout_score=rollout_score,
            rollout_model_logp=rollout_model_logp,
            rollout_policy_indices=rollout_policy_indices,
            rollout_policy_fused_scores=rollout_policy_fused_scores,
            expert_path=expert_path,
            expert_path_local=expert_path_local,
            route_ref_path=route_ref_path,
        )

    def _extract_rollout_expert_path(self) -> np.ndarray:
        dt = self._future_sampling.interval_length
        current_time_us = self.expert_trajectory.start_time.time_us
        traj_end_time_us = self.expert_trajectory.end_time.time_us
        if traj_end_time_us <= current_time_us:
            states = [self.expert_trajectory.get_state_at_time(TimePoint(int(traj_end_time_us)))]
        else:
            horizon_steps = max(int(np.ceil((traj_end_time_us - current_time_us) / (dt * 1e6))), self._future_sampling.num_poses)
            sample_times_us = current_time_us + np.arange(0, horizon_steps + 1, dtype=np.int64) * int(dt * 1e6)
            sample_times_us = np.clip(sample_times_us, current_time_us, traj_end_time_us)
            sample_times = [TimePoint(int(t)) for t in sample_times_us]
            states = self.expert_trajectory.get_state_at_times(sample_times)

        expert_path_global = np.zeros((len(states), 6), dtype=GLOBAL_COORD_DTYPE)
        for idx, state in enumerate(states):
            expert_path_global[idx, 0] = state.rear_axle.point.x
            expert_path_global[idx, 1] = state.rear_axle.point.y
            expert_path_global[idx, 2] = state.rear_axle.heading
            expert_path_global[idx, 3] = state.dynamic_car_state.speed
            expert_path_global[idx, 5] = state.dynamic_car_state.acceleration

        return expert_path_global

    def _project_path_to_ego_local(self, path_global: np.ndarray, ego_state: EgoState) -> np.ndarray:
        if path_global.shape[0] == 0:
            return np.zeros((0, 6), dtype=GLOBAL_COORD_DTYPE)

        ego_xy = np.asarray(ego_state.rear_axle.array, dtype=GLOBAL_COORD_DTYPE)
        ego_yaw = ego_state.rear_axle.heading
        rot_mat = np.array(
            [[np.cos(ego_yaw), -np.sin(ego_yaw)], [np.sin(ego_yaw), np.cos(ego_yaw)]],
            dtype=GLOBAL_COORD_DTYPE,
        )
        expert_path_local = np.copy(path_global)
        expert_path_local[:, :2] = (rot_mat.T @ (path_global[:, :2] - ego_xy).T).T
        expert_path_local[:, 2] = np.arctan2(
            np.sin(path_global[:, 2] - ego_yaw),
            np.cos(path_global[:, 2] - ego_yaw),
        )
        return expert_path_local

    def _save_rollout_sample(
        self,
        scene_feature: SceneFeature,
        agent_prediction: Optional[AgentPrediction],
        ego_state: EgoState,
        sampled_trajectories: np.ndarray,
        sampled_anchor_indices: np.ndarray,
        sampled_teacher_sources: np.ndarray,
        sampled_model_log_scores: np.ndarray,
        sampled_eval_scores: np.ndarray,
        sampled_selection_scores: np.ndarray,
        chosen_local_index: int,
        chosen_anchor_index: int,
        chosen_score: float,
        emergency_brake: bool,
        expert_path_local: np.ndarray,
        expert_route_ref_path: np.ndarray,
    ) -> None:
        self._rollout_worker.save_rollout_sample(
            scene_feature=scene_feature,
            agent_prediction=agent_prediction,
            ego_state=ego_state,
            sampled_trajectories=sampled_trajectories,
            sampled_anchor_indices=sampled_anchor_indices,
            sampled_teacher_sources=sampled_teacher_sources,
            sampled_model_log_scores=sampled_model_log_scores,
            sampled_eval_scores=sampled_eval_scores,
            sampled_selection_scores=sampled_selection_scores,
            chosen_local_index=chosen_local_index,
            chosen_anchor_index=chosen_anchor_index,
            chosen_score=chosen_score,
            emergency_brake=emergency_brake,
            expert_path_local=expert_path_local,
            expert_route_ref_path=expert_route_ref_path,
            iteration=self._iteration,
        )


    def compute_boundary_trajectory(
        self, 
        greedy_trajectory: np.ndarray, 
        ego_state: EgoState,
        max_deceleration: float = -4.0,
        max_lateral_velocity: float = 0.6,
        max_lateral_acceleration: float = 2.50,       
        max_longitudinal_acceleration: float = 2.0,  
        yaw_rate_max: float = 0.6,     
        max_long_jerk: float = 4.0, 
    ) -> np.ndarray:
        """ compute boundary trajectory for muvo planner"""
        # compute emergency stop trajectory or lane keeping trajectory if needed
        dt = self._future_sampling.interval_length
        T = self._future_sampling.num_poses
        # 1. compute lateral boundary (default to lane keeping)
        v_ego = ego_state.dynamic_car_state.speed
        current_ego = np.array([0.0, 0.0, 0.0])
        current_sd = self._scene_manager.lane_map.cartesian_to_frenet(
            points=current_ego.reshape(1, 3)
        )[0]  # (2,)
        current_s = current_sd[0]
        current_d = current_sd[1]
        expected_yaw = self._scene_manager.lane_map.frenet_to_cartesian(
            frenet_points=np.array([[current_s + v_ego*self._future_sampling.interval_length, 0]]),
            with_yaw=True,
        )[0, 2]
        expected_yaw_rate = expected_yaw / self._future_sampling.interval_length
        ego_yaw_rate = ego_state.dynamic_car_state.angular_velocity
        yaw_rate_diff = expected_yaw_rate - ego_yaw_rate
        sign_turn = np.sign(expected_yaw_rate) if abs(expected_yaw_rate) > 1e-6 else 0.0
        k_ff = 0.6          # 前馈增益（可调：0.4~1.0）
        tau_yaw = 0.3       # 估计的航向响应时间常数(s)
        yaw_rate_ref = 0.5  # 参考尺度(rad/s)
        d_bias_max = 0.20   # 最大内侧偏置(m)，避免越界
        scale = np.clip(abs(yaw_rate_diff) / yaw_rate_ref, 0.0, 1.5)
        d_feedforward = np.clip(sign_turn * k_ff * v_ego * tau_yaw * scale,
                                -d_bias_max, d_bias_max)
        min_lat_v = 0.1  # m/s
        max_lateral_velocity = min(max_lateral_velocity, np.sqrt(v_ego*np.tan(0.2) + 1e-6)) # limit by max steer angle at low speed
        if sign_turn != 0.0 and np.sign(yaw_rate_diff) == sign_turn:
            beta = 1.2
            lateral_velocity = np.clip(max_lateral_velocity / (1.0 + beta * scale),
                                       min_lat_v, max_lateral_velocity)
        else:
            lateral_velocity = max_lateral_velocity
        # print(f'lateral velocity for boundary computation: {lateral_velocity:.2f} m/s')

        # 目标 d = 前馈偏置（朝弯内侧）；从 current_d 以受限速度线性收敛到 d_target
        d_target = float(d_feedforward)
        d_plan = np.full(T, d_target, dtype=PLANNER_NUMPY_DTYPE)
        if lateral_velocity > 1e-6:
            steps = int(np.ceil(abs(d_target - current_d) / (lateral_velocity * dt)))
        else:
            steps = 0
        steps = max(0, min(steps, T))
        if steps > 0:
            ramp = current_d + (np.arange(1, steps + 1) / steps) * (d_target - current_d)
            d_plan[:steps] = ramp
            d_plan[steps:] = d_target
        else:
            d_plan[0] = current_d  # 很小的误差，保持当前

        # 2. compute longitudinal boundary (emergency stop + greedy forward)
        # 2.1 emergency stop
        s_plan_break = np.ones(self._future_sampling.num_poses) * current_s
        if v_ego > 0.1:
            
            t_stop = max(-v_ego / max_deceleration, self._future_sampling.interval_length)
            t_stop = round(np.ceil((t_stop - 1e-12) / self._future_sampling.interval_length) * self._future_sampling.interval_length, 2)
            s_stop = current_s + v_ego * t_stop + 0.5 * max_deceleration * t_stop ** 2
            deccel_range = max(int(t_stop / self._future_sampling.interval_length) + 1, 2)
            s_plan_decel = np.linspace(current_s, s_stop, deccel_range)[1:]
            v_plan_break = v_ego + max_deceleration * dt * np.arange(1, T + 1)
            v_plan_break = np.clip(v_plan_break, 0.0, None)
            a_plan_break = np.full_like(v_plan_break, max_deceleration)
            a_plan_break[v_plan_break <= 0.1] = 0.0
            if deccel_range < self._future_sampling.num_poses:
                s_plan_break[:deccel_range - 1] = s_plan_decel
                s_plan_break[deccel_range - 1:] = s_stop
            else:
                s_plan_break = s_plan_decel[:self._future_sampling.num_poses]
                
        else:
            v_plan_break = np.zeros_like(s_plan_break)
            a_plan_break = np.zeros_like(s_plan_break)
        
        
        # 2.2 greedy forward
        s_plan_greedy = self._scene_manager.lane_map.cartesian_to_frenet(
            points=greedy_trajectory[:, :3]
        )[:, 0]  # (T,)
        v_plan_greedy = greedy_trajectory[:, 3]
        a_plan_greedy = greedy_trajectory[:, 5]
        # limit velocity if curvature too high
        frenet_util = self._scene_manager.lane_map.frenet_path_util
        start_idx, end_idx = np.searchsorted(frenet_util.cumulative_s, [current_s, s_plan_greedy[-1]])
        ref_path_s = frenet_util.cumulative_s[start_idx:end_idx]
        ref_path_yaw = frenet_util.ref_theta[start_idx:end_idx]

        if self._debug:
            print(f'ego yaw_rate: {ego_state.dynamic_car_state.angular_velocity:.2f} rad/s, ego steer: {ego_state.dynamic_car_state.tire_steering_rate:.2f} rad')
            print(f'ego yaw_acc: {ego_state.dynamic_car_state.angular_acceleration:.2f} rad/s^2, ego steer angle: {ego_state.tire_steering_angle:.2f} rad')
            print(f'ego_lon_acc: {ego_state.dynamic_car_state.acceleration:.2f} m/s^2, ego speed: {v_ego:.2f} m/s')
            print(f'ego lat acc: {ego_state.dynamic_car_state.center_acceleration_2d.y:.2f} m/s^2')
        
        ds, dyaw = np.diff(ref_path_s), np.diff(ref_path_yaw)

        # 3. combine and transform longitudinal and lateral plan
        # 3.1 break trajectory
        break_trajectory_sd = np.zeros((self._future_sampling.num_poses, 2))
        break_trajectory_sd[:, 0] = s_plan_break
        break_trajectory_sd[:, 1] = np.ones(self._future_sampling.num_poses) * current_d
        break_trajectory_xy = self._scene_manager.lane_map.frenet_to_cartesian(
            frenet_points=break_trajectory_sd
        )  # (T, 2)
        break_trajectory_xy_extend = np.concatenate((
            np.array([[0, 0]]),
            break_trajectory_xy
        ), axis=0)  # (T+1, 2)
        break_trajectory_diff = np.diff(break_trajectory_xy_extend,axis=0) # (T, 2)
        break_trajectory_yaw = np.arctan2(break_trajectory_diff[:,1], break_trajectory_diff[:,0]) # (T,)
        break_trajectory_yaw = interp_valid_yaw(break_trajectory_yaw)
        break_trajectory_yaw = np.unwrap(break_trajectory_yaw)
        zero_yaws = np.zeros_like(break_trajectory_yaw)
        break_trajectory = np.zeros((self._future_sampling.num_poses, 6))
        break_trajectory[:, :2] = break_trajectory_xy
        break_trajectory[:, 2] = zero_yaws
        break_trajectory[:, 3] = v_plan_break
        break_trajectory[:, 5] = a_plan_break
        # if ego if out of frenet scope, force stopping
        if np.abs(current_s - self._scene_manager.lane_map.frenet_path_util.cumulative_s[-1]) < 0.1 or \
            np.abs(s_plan_break[-1] - self._scene_manager.lane_map.frenet_path_util.cumulative_s[-1]) < 0.1:
            break_time = ego_state.dynamic_car_state.speed / -max_deceleration
            break_time = round(np.ceil((break_time - 1e-12) / self._future_sampling.interval_length) * self._future_sampling.interval_length, 2)
            break_trajectory[:, :2] = np.zeros_like(break_trajectory[:, :2]) * ego_state.rear_axle.point.array
            x_stop = ego_state.dynamic_car_state.speed * break_time + 0.5 * max_deceleration * break_time ** 2
            deccel_range = int(break_time / self._future_sampling.interval_length) + 1
            x_plan_decel = np.linspace(0, x_stop, deccel_range)[1:]
            if deccel_range < self._future_sampling.num_poses:
                break_trajectory[:deccel_range - 1, 0] = x_plan_decel
                break_trajectory[deccel_range - 1:, 0] = x_stop
            else:
                break_trajectory[:, 0] = x_plan_decel[:self._future_sampling.num_poses]

        # 3.2 greedy trajectory
        greedy_trajectory_sd = np.zeros((self._future_sampling.num_poses, 2))
        greedy_trajectory_sd[:, 0] = s_plan_greedy
        greedy_trajectory_sd[:, 1] = d_plan
        
        if self._debug:
            print(f'[MUVO Planner] Boundary Trajectory: current_d = {current_d:.2f} m')
            print(f'   d_plan = {d_plan}')
        
        greedy_trajectory_xy = self._scene_manager.lane_map.frenet_to_cartesian(
            frenet_points=greedy_trajectory_sd
        )  # (T, 2)
        greedy_trajectory_xy_extend = np.concatenate((
            np.array([[0, 0]]),
            greedy_trajectory_xy
        ), axis=0)  # (T+1, 2)
        greedy_trajectory_diff = np.diff(greedy_trajectory_xy_extend,axis=0) # (T, 2)
        greedy_trajectory_yaw = np.arctan2(greedy_trajectory_diff[:,1], greedy_trajectory_diff[:,0]) # (T,)
        # handle NaN or inf in yaw
        greedy_trajectory_yaw = interp_valid_yaw(greedy_trajectory_yaw)
        greedy_trajectory_yaw = np.unwrap(greedy_trajectory_yaw)
        greedy_trajectory_final = np.zeros((self._future_sampling.num_poses, 6))
        greedy_trajectory_final[:, :2] = greedy_trajectory_xy
        greedy_trajectory_final[:, 2] = greedy_trajectory_yaw
        greedy_trajectory_final[:, 3] = v_plan_greedy
        greedy_trajectory_final[:, 5] = a_plan_greedy
        # print(f' greedy_trajectory_final se2: {greedy_trajectory_final[:10,:3]}')
        
        # 3.3 offset greedy trajectory
        lateral_offset = 0.0  # meters
        d_left_plan = np.ones(self._future_sampling.num_poses) * lateral_offset
        d_right_plan = np.ones(self._future_sampling.num_poses) * -lateral_offset
        left_shift_time = np.abs(current_d - lateral_offset) / lateral_velocity
        right_shift_time = np.abs(current_d + lateral_offset) / lateral_velocity
        left_shift_range = min(int(left_shift_time / self._future_sampling.interval_length) + 1, self._future_sampling.num_poses)
        right_shift_range = min(int(right_shift_time / self._future_sampling.interval_length) + 1, self._future_sampling.num_poses)
        d_left_plan[:left_shift_range] = np.linspace(current_d, lateral_offset, left_shift_range)
        d_right_plan[:right_shift_range] = np.linspace(current_d, -lateral_offset, right_shift_range)

        #3.3.1 right offset greedy trajectory
        greedy_trajectory_sd_right = np.zeros((self._future_sampling.num_poses, 2))
        greedy_trajectory_sd_right[:, 0] = s_plan_greedy
        greedy_trajectory_sd_right[:, 1] = d_right_plan
        greedy_trajectory_xy_right = self._scene_manager.lane_map.frenet_to_cartesian(
            frenet_points=greedy_trajectory_sd_right
        )
        greedy_trajectory_xy_right_extend = np.concatenate((
            np.array([[0, 0]]),
            greedy_trajectory_xy_right
        ), axis=0)  # (T+1, 2)
        greedy_trajectory_diff_right = np.diff(greedy_trajectory_xy_right_extend,axis=0) # (T, 2)
        greedy_trajectory_yaw = np.arctan2(greedy_trajectory_diff_right[:,1], greedy_trajectory_diff_right[:,0]) # (T,)
        # handle NaN or inf in yaw
        greedy_trajectory_yaw = interp_valid_yaw(greedy_trajectory_yaw)
        greedy_trajectory_yaw = np.unwrap(greedy_trajectory_yaw)
        ds_greedy_right = np.r_[0.0, np.linalg.norm(np.diff(greedy_trajectory_xy_right, axis=0), axis=1)]
        # greedy_trajectory_yaw = smooth_and_limit_yaw(greedy_trajectory_yaw, ds_greedy_right, yaw_rate_max, self._future_sampling.interval_length)
        greedy_trajectory_final_right = np.zeros((self._future_sampling.num_poses, 6))
        greedy_trajectory_final_right[:, :2] = greedy_trajectory_xy_right
        greedy_trajectory_final_right[:, 2] = greedy_trajectory_yaw
        greedy_trajectory_final_right[:, 3] = v_plan_greedy
        greedy_trajectory_final_right[:, 5] = a_plan_greedy

        #3.3.2 left offset greedy trajectory
        greedy_trajectory_sd_left = np.zeros((self._future_sampling.num_poses, 2))
        greedy_trajectory_sd_left[:, 0] = s_plan_greedy
        greedy_trajectory_sd_left[:, 1] = d_left_plan
        greedy_trajectory_xy_left = self._scene_manager.lane_map.frenet_to_cartesian(
            frenet_points=greedy_trajectory_sd_left
        )  # (T, 2)
        greedy_trajectory_xy_left_extend = np.concatenate((
            np.array([[0, 0]]),
            greedy_trajectory_xy_left
        ), axis=0)  # (T+1, 2)
        greedy_trajectory_diff_left = np.diff(greedy_trajectory_xy_left_extend,axis=0) # (T, 2)
        greedy_trajectory_yaw = np.arctan2(greedy_trajectory_diff_left[:,1], greedy_trajectory_diff_left[:,0]) # (T,)
        greedy_trajectory_yaw = interp_valid_yaw(greedy_trajectory_yaw) # handle NaN or inf in yaw
        greedy_trajectory_yaw = np.unwrap(greedy_trajectory_yaw)
        ds_greedy_left = np.r_[0.0, np.linalg.norm(np.diff(greedy_trajectory_xy_left, axis=0), axis=1)]
        # greedy_trajectory_yaw = smooth_and_limit_yaw(greedy_trajectory_yaw, ds_greedy_left, yaw_rate_max, self._future_sampling.interval_length)
        greedy_trajectory_final_left = np.zeros((self._future_sampling.num_poses, 6))
        greedy_trajectory_final_left[:, :2] = greedy_trajectory_xy_left
        greedy_trajectory_final_left[:, 2] = greedy_trajectory_yaw
        greedy_trajectory_final_left[:, 3] = v_plan_greedy
        greedy_trajectory_final_left[:, 5] = a_plan_greedy

        # 4. combine trajectories
        remaining_distances = np.abs(current_s - self._scene_manager.lane_map.frenet_path_util.cumulative_s[-1])
        if remaining_distances < 5.0 or \
            s_plan_greedy[-1] < s_plan_greedy[0] + 0.1 or\
            np.abs(s_plan_greedy[-1] - self._scene_manager.lane_map.frenet_path_util.cumulative_s[-1]) < 0.1:
            # use emergency stop trajectory as boundary
            # print(f'[MUVO Planner] Using emergency stop trajectory as boundary. Remaining distance to end of frenet path: {remaining_distances:.2f} m, \
            #     Greedy end s: {s_plan_greedy[-1]:.2f} m')
            # print(f'    Current s: {current_s:.2f} m, Greedy start s: {s_plan_greedy[0]:.2f} m, Greedy delta s: {s_plan_greedy[-1]-s_plan_greedy[0]:.2f} m')
            # print(f'    Frenet path s: {frenet_util.cumulative_s} m')
            boundary_trajectory = break_trajectory[np.newaxis, ...]  # (1, T, 6)
            return boundary_trajectory
        else:
            # if d_target == d_bias_max or d_target == -d_bias_max:
            #     print(f'[MUVO Planner] Warning: Lateral target d at boundary limit: {d_target:.2f} m')
            #     y_local_add = np.sign(d_target) * 0.5
            #     greedy_trajectory_final[:,1] += y_local_add
            #     greedy_trajectory_final_left[:,1] += y_local_add
            #     greedy_trajectory_final_right[:,1] += y_local_add
            #     yaw_local_add = np.sign(d_target) * 0.3
            #     greedy_trajectory_final[:,2] += yaw_local_add
            #     greedy_trajectory_final_left[:,2] += yaw_local_add
            #     greedy_trajectory_final_right[:,2] += yaw_local_add

            boundary_trajectory = np.concatenate((
                greedy_trajectory_final[np.newaxis, ...],
                greedy_trajectory_final_left[np.newaxis, ...],
                greedy_trajectory_final_right[np.newaxis, ...],
                break_trajectory[np.newaxis, ...],
            ), axis=0) # (N, T, 6)
        
        # print(f"Boundary trajectory : {boundary_trajectory}.")
        if np.any(np.isnan(boundary_trajectory)) or np.any(np.isinf(boundary_trajectory)):
            print(f"Boundary trajectory : {boundary_trajectory}.")
            print(f'ref_path: {self._scene_manager.lane_map.frenet_path_util.ref_path}')
            print(f'ref_path has nan: {np.any(np.isnan(self._scene_manager.lane_map.frenet_path_util.ref_path))}')
            print(f'ref_theta: {self._scene_manager.lane_map.frenet_path_util.ref_theta}')
            print(f'ref_path theta has nan: {np.any(np.isnan(self._scene_manager.lane_map.frenet_path_util.ref_theta))}')
            raise ValueError("Boundary trajectory contains NaN or inf values.")
        return boundary_trajectory

    def compute_fallback_trajectory(self, 
                                    ego_state: EgoState,
                                    trajectory: AbstractTrajectory) -> np.ndarray:
        """ compute fallback trajectory for muvo planner"""
        dt = self._future_sampling.interval_length
        current_time_point = ego_state.time_point
        T = self._future_sampling.num_poses
        fallback_trajectory = np.zeros((T, 6), dtype=GLOBAL_COORD_DTYPE)
        dt_time_point = TimePoint(int(dt * 1e6))
        trajectory_times = np.arange(1, T + 1) * dt_time_point.time_us + current_time_point.time_us
        traj_start_time = trajectory.start_time.time_us
        traj_end_time = trajectory.end_time.time_us
        trajectory_times_clipped = np.clip(trajectory_times, traj_start_time, traj_end_time)
        trajectory_times_clipped_point = [TimePoint(int(t)) for t in trajectory_times_clipped]
        states = trajectory.get_state_at_times(trajectory_times_clipped_point)
        
        for i, state in enumerate(states):
            fallback_trajectory[i, 0] = state.rear_axle.point.x
            fallback_trajectory[i, 1] = state.rear_axle.point.y
            fallback_trajectory[i, 2] = state.rear_axle.heading
            fallback_trajectory[i, 3] = state.dynamic_car_state.speed
            fallback_trajectory[i, 4] = 0
            fallback_trajectory[i, 5] = state.dynamic_car_state.acceleration
        
        ego_xy = np.asarray(ego_state.rear_axle.array, dtype=GLOBAL_COORD_DTYPE)
        ego_yaw = ego_state.rear_axle.heading
        rot_mat = np.array([[np.cos(ego_yaw), -np.sin(ego_yaw)],
                    [np.sin(ego_yaw),  np.cos(ego_yaw)]], dtype=GLOBAL_COORD_DTYPE)
        R = rot_mat; Rt = R.T
        
        p_rel = fallback_trajectory[:, :2] - ego_xy
        p_rel_back = Rt @ (p_rel.T)
        yaw_rel = fallback_trajectory[:, 2] - ego_yaw
        fallback_trajectory_local = np.copy(fallback_trajectory)
        fallback_trajectory_local[:, 0] = p_rel_back[0, :]
        fallback_trajectory_local[:, 1] = p_rel_back[1, :]
        fallback_trajectory_local[:, 2] = yaw_rel

        return fallback_trajectory_local
        
    def emergency_break_trajectory(self, ego_state: EgoState, min_time: float) -> np.ndarray:
        """
        Computes the emergency break trajectory for the given ego state.
        """
        current_ego = np.array([0.0, 0.0, 0.0])
        current_sd = self._scene_manager.lane_map.cartesian_to_frenet(
            points=current_ego.reshape(1, 3)
        )[0]  # (2,)
        current_d = current_sd[1]
        d_plan = np.ones(self._future_sampling.num_poses) * current_d
        v_ego = ego_state.dynamic_car_state.speed
        current_s = current_sd[0]
        s_plan_break = np.ones(self._future_sampling.num_poses) * current_s
        t_stop = min_time-0.1
        t_stop = max(self._future_sampling.interval_length, t_stop)
        desired_deceleration = 0.0 if v_ego <= 1e-3 else -v_ego / t_stop
        desired_deceleration = float(np.clip(desired_deceleration, -8.0, 0.0))
        s_stop = current_s + v_ego * t_stop + 0.5 * desired_deceleration * t_stop ** 2
        deccel_range = max(2, int(t_stop / self._future_sampling.interval_length) + 1)
        s_plan_decel = np.linspace(current_s, s_stop, deccel_range)[1:]
        if deccel_range < self._future_sampling.num_poses:
            s_plan_break[:deccel_range - 1] = s_plan_decel
            s_plan_break[deccel_range - 1:] = s_stop
        else:
            s_plan_break = s_plan_decel[:self._future_sampling.num_poses]

        break_trajectory_sd = np.zeros((self._future_sampling.num_poses, 2))
        break_trajectory_sd[:, 0] = s_plan_break
        break_trajectory_sd[:, 1] = d_plan
        break_trajectory_xy = self._scene_manager.lane_map.frenet_to_cartesian(
            frenet_points=break_trajectory_sd
        )  # (T, 2)
        current_yaw = 0.0
        break_trajectory_diff = np.diff(break_trajectory_xy,axis=0) # (T-1, 2)
        break_trajectory_yaw = np.arctan2(break_trajectory_diff[:,1], break_trajectory_diff[:,0]) # (T-1,)
        break_trajectory_yaw = np.concatenate((
            np.array([current_yaw]),
            break_trajectory_yaw
        ))  # (T,)
        break_trajectory_yaw = interp_valid_yaw(break_trajectory_yaw)
        break_trajectory_yaw = np.unwrap(break_trajectory_yaw)
        zero_yaws = np.zeros_like(break_trajectory_yaw)
        break_trajectory = np.zeros((self._future_sampling.num_poses, 6))
        break_trajectory[:, :2] = break_trajectory_xy
        break_trajectory[:, 2] = zero_yaws
        break_trajectory[0, 3] = ego_state.dynamic_car_state.speed
        break_trajectory[0, 5] = ego_state.dynamic_car_state.acceleration
        if np.abs(current_s - self._scene_manager.lane_map.frenet_path_util.cumulative_s[-1]) < 0.1 or \
            np.abs(s_plan_break[-1] - self._scene_manager.lane_map.frenet_path_util.cumulative_s[-1]) < 0.1:
            break_time = (v_ego / -desired_deceleration) if desired_deceleration < -1e-6 else self._future_sampling.interval_length
            break_trajectory[:, :2] = np.zeros_like(break_trajectory[:, :2]) * ego_state.rear_axle.point.array
            x_stop = ego_state.dynamic_car_state.speed * break_time + 0.5 * desired_deceleration * break_time ** 2
            deccel_range = max(2, int(break_time / self._future_sampling.interval_length) + 1)
            x_plan_decel = np.linspace(0, x_stop, deccel_range)[1:]
            if deccel_range < self._future_sampling.num_poses:
                break_trajectory[:deccel_range - 1, 0] = x_plan_decel
                break_trajectory[deccel_range - 1:, 0] = x_stop
            else:
                break_trajectory[:, 0] = x_plan_decel[:self._future_sampling.num_poses]

        return break_trajectory

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """Inherited, see superclass."""
        report = MLPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []
            self._evaluation_runtimes = []

        if self.save_replay:
            saved_path = self._video_output_path
            self._close_video_writer()
            if saved_path is not None:
                print("\n video saved to ", saved_path, "\n")

        return report
       
