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
from lpl_planner.planning.scene.evaluate.scene_scorer import BatchEvaluator
from lpl_planner.planning.scene.evaluate.simulator import (
    BatchSimulator,
    DEFAULT_SIMULATION_DT,
)
from lpl_planner.planning.planner.utils.emergency_brake import EmergencyBrake
from lpl_planner.planning.planner.utils.planner_utils import (
    trajectory_to_interpolated_trajectory,
    hausdorff_xy,
)
from lpl_planner.planning.planner.utils.replay_visualization import draw_muvo_replay_frame

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
                 eval_sampling_dt:float = 0.1,
                 replay_image_size_px: int = 1024,
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
        self._eval_sampling_dt = eval_sampling_dt
        self.replay_image_size_px = max(int(replay_image_size_px), 256)

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

        # store expert_trajectory for replay
        self.expert_trajectory: Optional[AbstractTrajectory] = self._extract_expert_trajectory()


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
        scene_feature = SceneFeature.deserialize(self._scene_manager.extract_feature_from_simulation(current_input))
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
            # stop trajectory for safety evaluation
            stop_trajectory = np.zeros((1, self._future_sampling.num_poses, 6), dtype=np.float32)
            if stop_trajectory not in trajectories:
                trajectories = np.concatenate((trajectories, stop_trajectory), axis=0)
                
            # eval fallback trajectory and prepend to candidate trajectories
            if (self.fallback_trajectory is not None) and \
            (self.fallback_trajectory.end_time.time_s > ego_state.time_point.time_s + 1.0):
                fallback_trajectory_local = self.compute_fallback_trajectory(ego_state, self.fallback_trajectory)
                trajectories = np.concatenate((trajectories, fallback_trajectory_local[np.newaxis, ...] ), axis=0)
            
            # simulate and evaluate all trajectories
            current_state = np.zeros((trajectories.shape[0], 1, trajectories.shape[-1]), dtype=np.float32)
            current_state[:, 0, 3] = ego_state.dynamic_car_state.speed  # set current speed
            current_state[:, 0, 5] = ego_state.dynamic_car_state.angular_velocity  # set current yaw rate
            current_state[:, 0, 4] = ego_state.dynamic_car_state.acceleration  # set current acceleration
            extended_trajectories = np.concatenate((current_state, trajectories), axis=1)

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
            
            pass
            
        
        selection_scores, eval_score_probs, model_log_scores, consensus_scores = self._compute_selection_scores(
            trajectories=trajectories,
            trajectory_score=trajectory_score if self.use_eval else None,
            model_log_scores=scores_pred,
        )
        scores = selection_scores
        best_candidate_idx = int(np.argmax(scores)) if scores.size > 0 else -1

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
            trajectory = emergency_break_trajectory
            chosen_local_index = -1
            chosen_anchor_index = -1
            executed_candidate_idx = -1
            chosen_score = 0.0
            emergency_brake_flag = True
        else:
            best_idx = best_candidate_idx
            executed_candidate_idx = best_idx
            emergency_brake_flag = False

            best_trajectory = trajectory_global[best_idx]
            
            trajectory = trajectory_to_interpolated_trajectory(
                trajectory=best_trajectory,
                ego_history=current_input.history.ego_states,
                future_horizon=self._future_sampling.time_horizon,
                step_interval=self._future_sampling.interval_length,
                use_anchor_velocity=False,
            )
            self.fallback_trajectory = trajectory

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
                image_size_px=self.replay_image_size_px,
            )

            self._append_video_frame(img)
        
        self._frame_idx += 1
        self._iteration += 1
        return trajectory
    
    
        
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
       
