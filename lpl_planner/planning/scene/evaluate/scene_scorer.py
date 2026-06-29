import copy
from typing import List, Tuple, Set, Optional, Dict, Union, Any
import numpy as np
import numpy.typing as npt
import uuid
import os
import gc
import time
from scipy.interpolate import interp1d

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import chunk_list, worker_map
from nuplan.planning.metrics.utils.collision_utils import CollisionType
from nuplan.planning.simulation.observation.idm.utils import  is_agent_behind, is_agent_ahead
from lpl_planner.planning.planner.utils.int_enum import RoadType

from shapely import LineString, Polygon, creation, distance

# from hybrid_planner.planning.scene.trajectory_library import TrajectoryState
from lpl_planner.planning.scene.map.occupancy_map import OccupancyMap
from lpl_planner.planning.scene.scene_manager import SceneManager
from lpl_planner.planning.scene.evaluate.utils.evaluate_utils import (BBCoordsIndex,
                             WeightedMetricIndex,
                             EgoAreaIndex,
                             MultiMetricIndex,
                             state_array_to_coords_array,
                             coords_array_to_polygon_array,
                             get_collision_type,
                             )
from lpl_planner.planning.scene.evaluate.utils.comfort_utils import mean_ratio_of_trajectories
from lpl_planner.planning.scene.evaluate.utils.control_utils import StateIndex
from lpl_planner.planning.scene.evaluate.simulator import (
    BatchSimulator,
    DEFAULT_SIMULATION_DT,
)
from lpl_planner.planning.scene.scene_feature.features import (SceneFeature,
                                                                  AgentPrediction,
                                                                  Trajectories
                                                                  )


import logging
logger = logging.getLogger(__name__)

STOPPED_SPEED_THRESHOLD : float = 5e-1 # [m/s] determine stop state of object/ego to avoid noise in data
MAX_PROGRESS_VEL = 20 # [m/s] max expected progress alone route, used for normalize progress score
MAX_DEVIATION = 7.5 # [m] max expected deviation, used for normalize deviation score

TIME_DISCOUNT = 0.99 # discount factor for future time steps in score calculation

# some weight for evaluation
EFFICIENCY_WEIGHT = 5
COMFORT_WEIGHT = 5
RISK_WEIGHT = 5
ALIGN_WEIGHT = 1

# constants
# TODO: Add to config
WEIGHTED_METRICS_WEIGHTS = np.zeros(len(WeightedMetricIndex), dtype=np.float32)
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.PROGRESS] = 5.0
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.TTC] = 5.0
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.SPEED_LIMIT] = 4.0
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.COMFORTABLE] = 2.0
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.LANE_CENTER_DISTANCE] = ALIGN_WEIGHT
WEIGHTED_METRICS_WEIGHTS[WeightedMetricIndex.HEADING_COMPLIANCE] = 0.0

# TODO: Add to config
DRIVING_DIRECTION_COMPLIANCE_THRESHOLD = 2.0  # [m] (driving direction)
DRIVING_DIRECTION_VIOLATION_THRESHOLD = 6.0  # [m] (driving direction)
MULTI_LANE_COMPLIANCE_THRESHOLD = 6.0  # [m] (within lane)
MULTI_LANE_VIOLATION_THRESHOLD = 20.0  # [m] (within lane)
STOPPED_SPEED_THRESHOLD = 5e-03  # [m/s] (ttc)
PROGRESS_DISTANCE_THRESHOLD = 0.5  # [m] (progress)
PROGRESS_SATURATION_MIN = 1.0  # [m] avoid over-rewarding tiny progress deltas
PROGRESS_SATURATION_ACCEL = 1.5  # [m/s^2] comfortable forward acceleration budget
PROGRESS_TTC_GATE_THRESHOLD = 0.5  # normalized TTC below this suppresses progress reward
FOLLOWING_BASE_GAP = 3.0  # [m] minimum comfortable standstill gap to a lead vehicle
FOLLOWING_TIME_HEADWAY = 1.0  # [s] desired time headway when following a slower lead vehicle
FOLLOWING_PENALTY_MIN_SCORE = 0.2  # [1] minimum following penalty when violating safe following distance and time headway
FOLLOWING_PROGRESS_TOLERANCE = 0.1  # [m] tolerate tiny progress while inside the following gap
FOLLOWING_PENALTY_ALPHA = 1.2  # [1/m] exponential penalty slope for unsafe following progress
FOLLOWING_RECOVERY_TIME_S = 1.0  # [s] time constant for forgiving past unsafe following after returning safe
DRIVABLE_AREA_SHORT_HORIZON_S = 1.5  # [s] fallback horizon when all long-horizon proposals leave road
DRIVABLE_AREA_EPS = 1e-3  # [m] numerical tolerance for treating a trajectory as on-road
AGENT_FUTURE_DEFAULT_DT = 0.2  # [s] agent_prediction / prediction_gt are currently stored at 5Hz
RED_LIGHT_PROGRESS_TOLERANCE = 0.2  # [m] tolerate tiny motion inside the red-light polygon before penalizing
RED_LIGHT_PENALTY_ALPHA = 1.2
RED_LIGHT_MIN_SCORE = 0.05
CENTERLINE_ON_ROUTE_MIN_FRACTION = 0.85


LANE_CENTER_DISTANCE_CHUNK = 4096
LANE_CENTER_DISTANCE_TAIL_WINDOW_S = 1.0
MAX_MEAN_LANE_CENTER_DISTANCE = 1.5  # [m] max distance to lane center for scoring, used for normalize lane center distance score
MIN_MEAN_LANE_CENTER_DISTANCE = 0.2  # [m] min distance to lane center for scoring, used for normalize lane center distance score
HEADING_ALIGNMENT_MIN_DIFF = 0.1 # [rad] smaller deviation than this is treated as perfectly aligned
HEADING_ALIGNMENT_MAX_DIFF = np.pi / 6.0  # [rad] larger deviation than this is treated as misaligned
HEADING_ALIGNMENT_SPEED_THRESHOLD = 0.2  # [m/s] avoid over-penalizing heading while nearly stopped
VALID_PROPOSAL_THRESHOLD = 1e-3

DEBUG = False
SCORER_DTYPE = np.float32
COLLISION_KIND_AGENT = 0
COLLISION_KIND_STATIC = 1

class BatchEvaluator:
    """
    Borrowed from tuplan_garage, revised for evaluation with higher resolution
    Class to score proposals in PDM pipeline. Re-implements nuPlan's closed-loop metrics.
    """

    def __init__(self, 
                 proposal_sampling: TrajectorySampling,
                 ttc_period: float = 1.0,
                 minimun_ttc: float = 0.95,
                 default_dt: float = DEFAULT_SIMULATION_DT,
                 trajectory_sample_dt: Optional[float] = None,
                 enable_valid_proposal_mask: bool = True,
                 use_following_penalty: bool = True,
                 following_check_interval_s: float = 0.5,):
        """
        Constructor of PDMScorer
        :param proposal_sampling: Sampling parameters for proposals
        """
        self.original_proposal_sampling = proposal_sampling
        self._simulation_sampling = self._build_effective_sampling(
            proposal_sampling,
            default_dt,
        )
        self.trajectory_sample_dt = self._validate_trajectory_sample_dt(
            trajectory_sample_dt
        )
        effective_dt = (
            self.trajectory_sample_dt
            if self.trajectory_sample_dt is not None
            else default_dt
        )
        self._proposal_sampling = self._build_effective_sampling(
            proposal_sampling,
            effective_dt,
        )

        # lazy loaded
        self._initial_ego_state: Optional[EgoState] = None
        self._drivable_area_map: Optional[OccupancyMap] = None
        self._lane_map: Optional[OccupancyMap] = None
        self._lane_centers: Optional[npt.NDArray[np.object_]] = None
        self._scene_lane_map: Optional[OccupancyMap] = None
        self._scene_lane_connection_hash: Optional[Dict[str, frozenset[str]]] = None
        self._future_collision_map: Optional[List[OccupancyMap]] = None
        self._future_collision_meta: Optional[List[Dict[str, np.ndarray]]] = None
        self._intersections: Optional[List[Polygon]] = None
        self._red_light_map: Optional[OccupancyMap] = None
        self._speed_limits: Optional[Dict[str, float]] = None
        self._lane_center_cache: Optional[List[Dict[str, npt.NDArray[np.float32]]]] = None

        self._num_proposals: Optional[int] = None
        self._states: Optional[npt.NDArray[np.float64]] = None
        self._comfort_states: Optional[npt.NDArray[np.float64]] = None
        self._ego_coords: Optional[npt.NDArray[np.float64]] = None
        self._ego_polygons: Optional[npt.NDArray[np.object_]] = None

        self._ego_areas: Optional[npt.NDArray[np.bool_]] = None
        self._ego_lane_token_mask: Optional[npt.NDArray[np.bool_]] = None
        self._drivable_lane_indices: Optional[npt.NDArray[np.int64]] = None
        self._route_lane_indices: Optional[npt.NDArray[np.int64]] = None

        self._multi_metrics: Optional[npt.NDArray[np.float32]] = None
        self._weighted_metrics: Optional[npt.NDArray[np.float32]] = None
        self._progress_raw: Optional[npt.NDArray[np.float32]] = None
        self._valid_proposal_mask: Optional[npt.NDArray[np.bool_]] = None
        self._following_penalty: Optional[npt.NDArray[np.float32]] = None

        self._collision_time_idcs: Optional[npt.NDArray[np.float32]] = None
        self._ttc_time_idcs: Optional[npt.NDArray[np.float32]] = None

        self._ignored_collision_tokens: Set[str] = None

        self.ttc_period = ttc_period
        self.minimun_ttc = minimun_ttc
        self.enable_valid_proposal_mask = enable_valid_proposal_mask
        self.use_following_penalty = use_following_penalty
        self.following_check_interval_s = following_check_interval_s

    def _release_transient_state(self) -> None:
        """Drop per-scenario state so large shapely/ndarray objects can be reclaimed before the next reset."""
        self._route_map = None
        self._states = None
        self._comfort_states = None
        self._ego_coords = None
        self._ego_polygons = None
        self._future_collision_map = None
        self._future_collision_meta = None
        self._ego_areas = None
        self._ego_lane_token_mask = None
        self._drivable_lane_indices = None
        self._route_lane_indices = None
        self._multi_metrics = None
        self._weighted_metrics = None
        self._progress_raw = None
        self._collision_time_idcs = None
        self._ttc_time_idcs = None
        self._ignored_collision_tokens = None
        self._lane_map = None
        self._lane_centers = None
        self._lane_center_cache = None
        self._scene_lane_map = None
        self._scene_lane_connection_hash = None
        self._intersections = None
        self._red_light_map = None
        self._speed_limits = None
        self._drivable_area_map = None
        self.static_obstacle_position = None
        self.agent_position = None
        self.expert_trajectory = None
        self.ref_path = None
        self._valid_proposal_mask = None
        self._following_penalty = None

    @staticmethod
    def _build_effective_sampling(
        source_sampling: TrajectorySampling,
        target_dt: float,
    ) -> TrajectorySampling:
        if np.isclose(source_sampling.interval_length, target_dt):
            return source_sampling

        target_num_poses = int(round(source_sampling.time_horizon / target_dt))
        return TrajectorySampling(
            num_poses=target_num_poses,
            interval_length=target_dt,
        )

    @staticmethod
    def _validate_trajectory_sample_dt(
        trajectory_sample_dt: Optional[float],
    ) -> Optional[float]:
        if trajectory_sample_dt is None:
            return None

        for allowed_dt in (0.2, 0.5):
            if np.isclose(trajectory_sample_dt, allowed_dt):
                return float(allowed_dt)

        raise ValueError(
            "trajectory_sample_dt must be one of None, 0.2, or 0.5 seconds"
        )

    def _sample_simulated_trajectories(
        self,
        states: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        if self.trajectory_sample_dt is None:
            return states

        target_num_steps = self._proposal_sampling.num_poses + 1
        if states.shape[1] == target_num_steps:
            return states

        if states.shape[1] <= 1:
            raise ValueError("simulated trajectories must contain at least two time steps")

        source_num_steps = states.shape[1]
        source_dt = self.original_proposal_sampling.time_horizon / (source_num_steps - 1)
        target_dt = self._proposal_sampling.interval_length
        if target_dt + 1e-6 < source_dt:
            raise ValueError(
                f"trajectory_sample_dt={target_dt} cannot be finer than source dt={source_dt}"
            )

        target_times = np.arange(target_num_steps, dtype=np.float64) * target_dt
        source_indices = np.rint(target_times / source_dt).astype(np.int64)
        if source_indices[-1] >= source_num_steps:
            raise ValueError(
                "trajectory_sample_dt is incompatible with simulated trajectory length"
            )

        reconstructed_times = source_indices.astype(np.float64) * source_dt
        if not np.allclose(reconstructed_times, target_times, atol=1e-6):
            raise ValueError(
                "trajectory_sample_dt is incompatible with the simulated trajectory time grid"
            )

        return states[:, source_indices, :]

    @staticmethod
    def _resample_agent_future_states(
        agent_current_state: npt.NDArray[np.float64],
        agent_future_state: npt.NDArray[np.float64],
        agent_future_mask: npt.NDArray[np.bool_],
        target_num_steps: int,
        target_dt: float,
        source_dt: float = AGENT_FUTURE_DEFAULT_DT,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """Resample sparse agent futures onto the evaluator time grid and truncate to the requested horizon."""
        if target_num_steps <= 0:
            num_agents = min(agent_current_state.shape[0], agent_future_state.shape[0], agent_future_mask.shape[0])
            state_dim = agent_future_state.shape[-1] if agent_future_state.ndim == 3 else agent_current_state.shape[-1]
            return (
                np.zeros((num_agents, 0, state_dim), dtype=SCORER_DTYPE),
                np.zeros((num_agents, 0), dtype=np.bool_),
            )

        num_agents = min(agent_current_state.shape[0], agent_future_state.shape[0], agent_future_mask.shape[0])
        state_dim = agent_future_state.shape[-1]
        if num_agents == 0:
            return (
                np.zeros((0, target_num_steps, state_dim), dtype=SCORER_DTYPE),
                np.zeros((0, target_num_steps), dtype=np.bool_),
            )

        target_times = np.arange(1, target_num_steps + 1, dtype=SCORER_DTYPE) * target_dt
        future_times = np.arange(1, agent_future_state.shape[1] + 1, dtype=SCORER_DTYPE) * source_dt
        source_times = np.concatenate((np.array([0.0], dtype=SCORER_DTYPE), future_times), axis=0)

        resampled_states = np.zeros((num_agents, target_num_steps, state_dim), dtype=SCORER_DTYPE)
        resampled_mask = np.zeros((num_agents, target_num_steps), dtype=np.bool_)

        for agent_idx in range(num_agents):
            source_states = np.concatenate(
                (
                    agent_current_state[agent_idx : agent_idx + 1, :state_dim],
                    agent_future_state[agent_idx],
                ),
                axis=0,
            )
            source_mask = np.concatenate(
                (np.array([True], dtype=np.bool_), agent_future_mask[agent_idx].astype(np.bool_)),
                axis=0,
            )
            source_valid = source_mask & np.isfinite(source_states).all(axis=-1)
            if not np.any(source_valid):
                continue

            valid_times = source_times[source_valid]
            valid_states = source_states[source_valid]

            if valid_times.size == 1:
                resampled_states[agent_idx] = valid_states[0]
                resampled_mask[agent_idx] = True
                continue

            in_range = (target_times >= valid_times[0] - 1e-6) & (target_times <= valid_times[-1] + 1e-6)
            if not np.any(in_range):
                continue

            interp_times = target_times[in_range]
            interp_state = np.empty((interp_times.shape[0], state_dim), dtype=SCORER_DTYPE)
            for dim_idx in range(state_dim):
                values = valid_states[:, dim_idx]
                if dim_idx == 2:
                    values = np.unwrap(values)
                interp_fn = interp1d(
                    valid_times,
                    values,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(values[0], values[-1]),
                    assume_sorted=True,
                )
                interp_state[:, dim_idx] = interp_fn(interp_times)

            if state_dim > 2:
                interp_state[:, 2] = np.arctan2(np.sin(interp_state[:, 2]), np.cos(interp_state[:, 2]))

            resampled_states[agent_idx, in_range] = interp_state
            resampled_mask[agent_idx, in_range] = True

        return resampled_states, resampled_mask

    @staticmethod
    def _build_lane_center_cache(
        lane_centers: npt.NDArray[np.object_],
    ) -> List[Dict[str, npt.NDArray[np.float32]]]:
        cache: List[Dict[str, npt.NDArray[np.float32]]] = []
        for lane_center in lane_centers:
            coords = np.asarray(lane_center.coords, dtype=np.float32)
            if coords.shape[0] < 2:
                continue

            segment_vectors = np.diff(coords, axis=0)
            segment_lengths = np.linalg.norm(segment_vectors, axis=1)
            valid_segments = segment_lengths > 1e-4
            if not np.any(valid_segments):
                continue

            segment_starts = coords[:-1][valid_segments]
            segment_vectors = segment_vectors[valid_segments]
            segment_lengths = segment_lengths[valid_segments].astype(np.float32, copy=False)
            segment_headings = np.arctan2(segment_vectors[:, 1], segment_vectors[:, 0]).astype(np.float32)
            cache.append(
                {
                    "starts": segment_starts.astype(np.float32, copy=False),
                    "vectors": segment_vectors.astype(np.float32, copy=False),
                    "lengths": segment_lengths,
                    "headings": segment_headings,
                }
            )

        return cache

    @staticmethod
    def _project_points_to_polyline(
        points_xy: npt.NDArray[np.float32],
        lane_center: Dict[str, npt.NDArray[np.float32]],
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        starts = lane_center["starts"]
        vectors = lane_center["vectors"]
        lengths = lane_center["lengths"]
        headings = lane_center["headings"]

        if points_xy.shape[0] == 0:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        rel_points = points_xy[:, None, :] - starts[None, :, :]
        denom = np.maximum(lengths * lengths, 1e-6)[None, :]
        projection_ratio = np.clip(
            np.sum(rel_points * vectors[None, :, :], axis=-1) / denom,
            0.0,
            1.0,
        )
        projected_points = starts[None, :, :] + projection_ratio[..., None] * vectors[None, :, :]
        distance_sq = np.sum((points_xy[:, None, :] - projected_points) ** 2.0, axis=-1)
        closest_segment_idx = np.argmin(distance_sq, axis=1)
        point_indices = np.arange(points_xy.shape[0])
        closest_distance = np.sqrt(distance_sq[point_indices, closest_segment_idx]).astype(np.float32)
        closest_heading = headings[closest_segment_idx].astype(np.float32, copy=False)
        return closest_distance, closest_heading

    @staticmethod
    def _normalize_angle(angle: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.arctan2(np.sin(angle), np.cos(angle)).astype(np.float32)

    def _refresh_valid_proposal_mask(self) -> npt.NDArray[np.bool_]:
        if not self.enable_valid_proposal_mask:
            num_proposals = self._num_proposals
            if num_proposals is None and self._multi_metrics is not None:
                num_proposals = self._multi_metrics.shape[1]
            if num_proposals is None:
                raise ValueError("Cannot refresh valid proposal mask before proposals are initialized")
            self._valid_proposal_mask = np.ones(num_proposals, dtype=np.bool_)
            return self._valid_proposal_mask

        self._valid_proposal_mask = (
            self._multi_metrics.prod(axis=0) > VALID_PROPOSAL_THRESHOLD
        )
        return self._valid_proposal_mask

    def _get_active_multi_metric_mask(self) -> npt.NDArray[np.bool_]:
        if self._valid_proposal_mask is None:
            return np.ones(self._num_proposals, dtype=np.bool_)
        return self._valid_proposal_mask

    def time_to_at_fault_collision(self, proposal_idx: int) -> float:
        """
        Returns time to at-fault collision for given proposal
        :param proposal_idx: index for proposal
        :return: time to infraction
        """
        return (
            self._collision_time_idcs[proposal_idx]
            * self._proposal_sampling.interval_length
        )

    def time_to_ttc_infraction(self, proposal_idx: int) -> float:
        """
        Returns time to ttc infraction for given proposal
        :param proposal_idx: index for proposal
        :return: time to infraction
        """
        return (
            self._ttc_time_idcs[proposal_idx] * self._proposal_sampling.interval_length
        )

    def batch_evaluate(
        self,
        states: npt.NDArray[np.float64],
        scene_feature: SceneFeature = None,
        agent_prediction: AgentPrediction = None,
        agent_prediction_gt: AgentPrediction = None,
        discount_factor: float = 1.0,
        scene_manager: SceneManager = None,
        expert_trajectory: npt.NDArray[np.float64] = None,
        ref_path: npt.NDArray[np.float64] = None,
        prediction_mode: str = "prediction",
        aggregate_only: bool = False,
        debug: bool = False,
        return_timing: bool = False,
    ) -> Dict[str, Any]:
        """
        Scores proposal similar to nuPlan's closed-loop metrics
        :param states: array representation of simulated proposals
        :param scene_feature: scene feature containing map and actor information
        :param agent_prediction: predicted future states of surrounding agents
        :param agent_prediction_gt: ground truth future states of surrounding agents
        :param discount_factor: discount factor for future time steps in score calculation
        :param scene_manager: scene manager containing map and actor information
        :param expert_trajectory: expert trajectory for comparison, should have same dt as proposal sampling
        :param ref_path: reference path for calculating progress
        :param prediction_mode: mode for prediction, "prediction", "CV", "CA", "CYAW", "CAYAW"
        :return: dict containing score of each proposal
        """
    
        timing_log_level = logging.INFO if debug else logging.DEBUG

        timing_profile: Dict[str, float] = {}

        def log_timing(key: str, message: str, started_at: float) -> float:
            elapsed = time.time() - started_at
            timing_profile[key] = float(elapsed)
            logger.log(timing_log_level, "%s: %.2f seconds", message, elapsed)
            return time.time()

        try:
            start_time = time.time()
            step_start_time = start_time
            self._comfort_states = states
            states = self._sample_simulated_trajectories(states)
            # initialize & lazy load class values
            self._reset(
                trajectories=states,
                scene_feature=scene_feature,
                agent_prediction=agent_prediction,
                agent_prediction_gt=agent_prediction_gt,
                scene_manager=scene_manager,
                expert_trajectory=expert_trajectory,
                ref_path=ref_path,
                prediction_mode=prediction_mode,
            )
            step_start_time = log_timing("reset", "Reset time", step_start_time)

            # fill value ego-area array (used across multiple metrics)
            self._calculate_ego_area()
            step_start_time = log_timing("ego_area", "Ego area calculation time", step_start_time)

            # 1. multiplicative metrics
            self._calculate_drivable_area_compliance()
            step_start_time = log_timing("drivable_area_compliance", "Drivable area compliance calculation time", step_start_time)

            self._refresh_valid_proposal_mask()
            self._calculate_no_at_fault_collision()
            step_start_time = log_timing("no_at_fault_collision", "No at fault collision calculation time", step_start_time)

            self._calculate_driving_direction_compliance()
            step_start_time = log_timing("driving_direction_compliance", "Driving direction compliance calculation time", step_start_time)

            self._calculate_within_lane_compliance()
            step_start_time = log_timing("within_lane_compliance", "Within lane compliance calculation time", step_start_time)

            self._calculate_red_light_compliance()
            step_start_time = log_timing("red_light_compliance", "Red light compliance calculation time", step_start_time)

            self._calculate_following_compliance()
            step_start_time = log_timing("following_compliance", "Following compliance calculation time", step_start_time)
            self._refresh_valid_proposal_mask()

            # 2. weighted metrics
            self._calculate_within_speed_limit()
            step_start_time = log_timing("speed_limit", "Within speed limit calculation time", step_start_time)

            self._calculate_progress()
            step_start_time = log_timing("progress", "Progress calculation time", step_start_time)


            self._calculate_ttc()
            step_start_time = log_timing("ttc", "TTC calculation time", step_start_time)

            self._calculate_is_comfortable()
            step_start_time = log_timing("comfort", "Comfort calculation time", step_start_time)

            self._calculate_lane_center_distance()
            step_start_time = log_timing("lane_center_distance", "Lane center distance calculation time", step_start_time)

            scores = self._explainable_scores()
            step_start_time = log_timing("explainable_scores", "Explainable scores calculation time", step_start_time)
            total_elapsed = time.time() - start_time
            timing_profile["total"] = float(total_elapsed)
            logger.log(timing_log_level, "Total batch evaluation time: %.2f seconds", total_elapsed)

            if return_timing:
                scores["timing"] = timing_profile.copy()

            if aggregate_only:
                return {
                    'aggregate_scores': scores['aggregate_scores'],
                    'min_time_to_collision': scores['min_time_to_collision'],
                    **({'timing': timing_profile.copy()} if return_timing else {}),
                }
            return scores
        finally:
            self._release_transient_state()
    
    def _explainable_scores(self) -> Dict[str, npt.NDArray[np.float32]]:
        """
        Returns a dictionary with all scores for explainability.
        """
        # accumulate multiplicative metrics
        multiplicate_metric_scores = self._multi_metrics.prod(axis=0)

        weighted_metrics = self._weighted_metrics.copy()

        # accumulate weighted metrics
        weighted_metric_scores = (
            weighted_metrics * WEIGHTED_METRICS_WEIGHTS[..., None]
        ).sum(axis=0)
        # weighted_metric_scores /= WEIGHTED_METRICS_WEIGHTS.sum()

        following_penalty = self._following_penalty.copy()

        # calculate final scores
        final_scores = multiplicate_metric_scores * weighted_metric_scores

        return {
            'multi_metrics': self._multi_metrics,
            'weighted_metrics': weighted_metrics,
            'following_penalty': following_penalty,
            'aggregate_scores': final_scores,
            'min_time_to_collision': self._collision_time_idcs.min() * self._proposal_sampling.interval_length,
            'collision_times': self._collision_time_idcs.copy() * self._proposal_sampling.interval_length,
            'ttc_times': self._ttc_time_idcs.copy() * self._proposal_sampling.interval_length,
        }

    def _compute_following_penalty(
        self,
    ) -> npt.NDArray[np.float32]:
        penalty = np.ones((self._num_proposals,), dtype=np.float32)

        if (
            not self.use_following_penalty
            or self._future_collision_meta is None
            or self._future_collision_map is None
            or len(self._future_collision_meta) == 0
            or len(self._future_collision_map) == 0
        ):
            return penalty

        num_steps = min(len(self._future_collision_meta), self._proposal_sampling.num_poses + 1)
        if num_steps <= 0:
            return penalty

        valid_mask = self._get_active_multi_metric_mask()
        if not np.any(valid_mask):
            return penalty

        valid_indices = np.flatnonzero(valid_mask)
        ego_points = np.asarray(self._ego_coords[valid_mask, :num_steps, BBCoordsIndex.CENTER], dtype=np.float32)
        ego_heading = np.asarray(self._states[valid_mask, :num_steps, StateIndex.HEADING], dtype=np.float32)
        ego_speed = np.hypot(
            self._states[valid_mask, :num_steps, StateIndex.VELOCITY_X],
            self._states[valid_mask, :num_steps, StateIndex.VELOCITY_Y],
        ).astype(np.float32)

        safe_gap = FOLLOWING_BASE_GAP + FOLLOWING_TIME_HEADWAY * ego_speed
        safe_shift = np.stack(
            [
                safe_gap * np.cos(ego_heading),
                safe_gap * np.sin(ego_heading),
            ],
            axis=-1,
        ).astype(np.float32)

        front_left = np.asarray(
            self._ego_coords[valid_mask, :num_steps, BBCoordsIndex.FRONT_LEFT],
            dtype=np.float32,
        )
        front_right = np.asarray(
            self._ego_coords[valid_mask, :num_steps, BBCoordsIndex.FRONT_RIGHT],
            dtype=np.float32,
        )
        shifted_front_left = front_left + safe_shift
        shifted_front_right = front_right + safe_shift
        corridor_coords = np.stack(
            [
                front_left,
                front_right,
                shifted_front_right,
                shifted_front_left,
                front_left,
            ],
            axis=2,
        ).astype(np.float32, copy=False)

        num_valid_proposals = int(valid_mask.sum())
        violating_following = np.zeros((num_valid_proposals, num_steps), dtype=np.bool_)
        check_stride = max(
            int(round(self.following_check_interval_s / self._proposal_sampling.interval_length)),
            1,
        )
        checked_time_indices = np.arange(0, num_steps, check_stride, dtype=np.int64)
        if checked_time_indices.size == 0 or checked_time_indices[-1] != num_steps - 1:
            checked_time_indices = np.concatenate(
                [checked_time_indices, np.asarray([num_steps - 1], dtype=np.int64)],
                axis=0,
            )
        checked_time_indices = np.unique(checked_time_indices)

        for time_idx in checked_time_indices:
            meta = self._future_collision_meta[time_idx]
            if meta is None or meta["x"].size == 0:
                continue

            occupancy = self._future_collision_map[time_idx]
            if len(occupancy) == 0:
                continue

            projected_polygons = creation.polygons(corridor_coords[:, time_idx])
            intersecting = occupancy.query(projected_polygons, predicate="intersects")
            if len(intersecting) == 0:
                continue

            actor_mask = np.asarray(meta["kind"], dtype=np.int8) == COLLISION_KIND_AGENT
            if not np.any(actor_mask):
                continue

            actor_geometry_mask = actor_mask[intersecting[1]]
            if not np.any(actor_geometry_mask):
                continue

            violating_following[intersecting[0][actor_geometry_mask], time_idx] = True

        step_progress = np.zeros((num_valid_proposals, num_steps), dtype=np.float32)
        if num_steps > 1:
            step_progress[:, 1:] = np.linalg.norm(
                ego_points[:, 1:] - ego_points[:, :-1],
                axis=-1,
            ).astype(np.float32)

        cumulative_progress = np.cumsum(step_progress, axis=-1, dtype=np.float32)
        checked_cumulative_progress = cumulative_progress[:, checked_time_indices]
        checked_incremental_progress = np.zeros_like(checked_cumulative_progress, dtype=np.float32)
        if checked_time_indices.size > 1:
            checked_incremental_progress[:, 1:] = (
                checked_cumulative_progress[:, 1:] - checked_cumulative_progress[:, :-1]
            )

        checked_violating = violating_following[:, checked_time_indices]
        unsafe_progress_debt = np.zeros((num_valid_proposals,), dtype=np.float32)
        for local_idx, time_idx in enumerate(checked_time_indices):
            if local_idx > 0:
                interval_s = (
                    float(time_idx - checked_time_indices[local_idx - 1])
                    * self._proposal_sampling.interval_length
                )
                safe_mask = ~checked_violating[:, local_idx]
                if np.any(safe_mask):
                    recovery = np.exp(-interval_s / FOLLOWING_RECOVERY_TIME_S)
                    unsafe_progress_debt[safe_mask] *= np.float32(recovery)

            unsafe_progress_debt += (
                checked_incremental_progress[:, local_idx]
                * checked_violating[:, local_idx].astype(np.float32)
            )

        excess_unsafe_progress = np.maximum(
            unsafe_progress_debt - FOLLOWING_PROGRESS_TOLERANCE,
            0.0,
        )
        penalty[valid_indices] = np.clip(
            np.exp(-FOLLOWING_PENALTY_ALPHA * excess_unsafe_progress),
            FOLLOWING_PENALTY_MIN_SCORE,
            1.0,
        ).astype(np.float32)

        return penalty

    def _calculate_following_compliance(self) -> None:
        following_penalty = self._compute_following_penalty()
        self._following_penalty = following_penalty
        self._multi_metrics[MultiMetricIndex.FOLLOWING_COMPLIANCE] = following_penalty

    def _aggregate_scores(self) -> npt.NDArray[np.float32]:
        """
        Aggregates metrics with multiplicative and weighted average.
        :return: array containing score of each proposal
        """
        logging.info(f"multi-metrics are: {self._multi_metrics} ")
        
        # accumulate multiplicative metrics
        multiplicate_metric_scores = self._multi_metrics.prod(axis=0)

        # normalize and fill progress values
        raw_progress = self._progress_raw * multiplicate_metric_scores
        max_raw_progress = np.max(raw_progress)
        if max_raw_progress > PROGRESS_DISTANCE_THRESHOLD:
            normalized_progress = raw_progress / max_raw_progress
        else:
            normalized_progress = np.ones(len(raw_progress), dtype=np.float32)
            normalized_progress[multiplicate_metric_scores == 0.0] = 0.0
        self._weighted_metrics[WeightedMetricIndex.PROGRESS] = normalized_progress

        weighted_metrics = self._weighted_metrics.copy()

        logging.info(f"weighted-metrics are: {self._weighted_metrics} ")

        # accumulate weighted metrics
        weighted_metric_scores = (
            weighted_metrics * WEIGHTED_METRICS_WEIGHTS[..., None]
        ).sum(axis=0)
        weighted_metric_scores /= WEIGHTED_METRICS_WEIGHTS.sum()

        # calculate final scores
        final_scores = multiplicate_metric_scores * weighted_metric_scores

        return final_scores

    def _reset(
        self,
        trajectories: npt.NDArray[np.float64],
        scene_feature: SceneFeature = None,
        agent_prediction: AgentPrediction = None,
        agent_prediction_gt: AgentPrediction = None,
        scene_manager: SceneManager = None,
        lateral_buffer: float = 0.1,
        longitudinal_buffer: float = 0.1,
        longitudinal_velocity_buffer: float = 0.5,
        expert_trajectory: npt.NDArray[np.float64] = None,
        ref_path: npt.NDArray[np.float64] = None,
        prediction_mode: str = "prediction",
        lane_inflation: float = 0.1,
        static_inflation: float = 0.2,
    ) -> None:
        """
        Resets metric values and lazy loads input classes.
        :param trajectories: array representation of simulated proposals # [N, T+1, state_size]
        :param initial_ego_state: ego-vehicle state at current iteration
        :param observation: 
        :param centerline: path of the centerline
        :param route_lane_dict: dictionary containing on-route lanes
        :param drivable_area_map: Occupancy map of drivable are polygons
        :param map_api: map object
        :param expert_trajectory: expert trajectory for comparison, should have same dt as proposal sampling
        :param lane_inflation: inflation value for lane boundaries
        :param static_inflation: inflation value for static obstacles
        """
        assert trajectories.ndim == 3
        assert trajectories.shape[1] == self._proposal_sampling.num_poses+1, f"Expected {self._proposal_sampling.num_poses} poses, but got {trajectories.shape[1]}"
        assert trajectories.shape[2] >= 6, f"Expected state size at least 6, but got {trajectories.shape[2]}"

        # save ego state values
        self._states = trajectories

        # calculate coordinates of ego corners and center
        self._ego_coords = state_array_to_coords_array(
            trajectories, 
            half_width=scene_feature.ego_feature.ego_geometry[..., 0],
            half_length=scene_feature.ego_feature.ego_geometry[..., 1],
            rear_axle_to_center=scene_feature.ego_feature.ego_geometry[..., 2],
        ) # [N, T+1, 5, (x,y)]

        # initialize all ego polygons from corners
        coords_exterior = self._ego_coords.copy()
        coords_exterior[:, :, BBCoordsIndex.CENTER, :] = coords_exterior[:, :, BBCoordsIndex.FRONT_LEFT, :]
        if not np.isfinite(coords_exterior).all():
            coords_exterior = np.nan_to_num(coords_exterior, copy=False)
        self._ego_polygons = coords_array_to_polygon_array(coords_exterior)  # [N, T+1] of Polygon
        
        self.expert_trajectory = expert_trajectory

        # 0. construct route lane map
        self.ref_path = None if ref_path is None else np.asarray(ref_path, dtype=SCORER_DTYPE)

        route_feature = scene_feature.route_feature
        route_polygon = np.asarray(route_feature.route_geometry)
        route_tokens = [f"route_roadblock_{i}" for i in range(len(route_polygon))]
        route_polygons = [Polygon(polygon) for polygon in route_polygon]
        self._route_map = OccupancyMap(route_tokens, route_polygons)


        # 1. construct drivable area map
        road_feature = scene_feature.road_feature
        road_polygon = np.asarray(road_feature.road_geometry)
        road_type = np.asarray(road_feature.road_type)
        tl_status = np.asarray(road_feature.road_traffic_light)
        centerlines = np.asarray(road_feature.center_line)
        speed_limits = np.asarray(road_feature.road_speed_limit)

        lane_polygons = []
        lane_tokens = []
        red_light_polygons = []
        red_light_tokens = []
        intersection_tokens = []
        intersection_polygons = []
        car_park_tokens = []
        car_park_polygons = []
        lane_centers = []
        speed_limits_dict = {}
        route_lane_tokens = []
        for polygon, r_type, t_stat, centerline, speed_limit in zip(road_polygon, road_type, tl_status, centerlines, speed_limits):
            centerline = np.asarray(centerline, dtype=np.float32)
            if centerline.ndim == 2:
                centerline_xy = centerline[:, :2]
                valid_centerline_xy = centerline_xy[np.isfinite(centerline_xy).all(axis=-1)]
            else:
                valid_centerline_xy = np.zeros((0, 2), dtype=np.float32)

            poly = Polygon(polygon)
            centerline_on_route = False
            
            
            if r_type in [RoadType.LANE, RoadType.CONNECTOR]:
                lane_token = f"lane_{len(lane_tokens)}"
                lane_polygons.append(poly)
                lane_tokens.append(lane_token)
                speed_limits_dict[lane_token] = speed_limit

                if valid_centerline_xy.shape[0] > 0:
                    centerline_in_route = self._route_map.points_in_polygons(valid_centerline_xy).any(axis=0)
                    centerline_on_route = bool(
                        centerline_in_route.mean() >= CENTERLINE_ON_ROUTE_MIN_FRACTION
                    )

                if centerline_on_route:
                    route_lane_tokens.append(lane_token)

                # add centerline to lane center list if it is fully in route, this is used for later calculating lane center distance metric
                if centerline_on_route and valid_centerline_xy.shape[0] > 1:
                    centerline_ls = LineString(valid_centerline_xy)
                    lane_centers.append(centerline_ls)
                
                if t_stat == 3: # TODO: 3 is red light
                    red_light_poly = poly
                    if centerline_on_route:  # only consider red light on route lane
                        red_light_polygons.append(red_light_poly)
                        red_light_tokens.append(f"red_light_lane_{len(red_light_tokens)}")
            
            if r_type == RoadType.INTERSECTION:
                intersection_tokens.append(f"intersection_{len(intersection_polygons)}")
                intersection_polygons.append(poly)
            
            if r_type in [RoadType.CARPARK]: # consider lane, connector and intersection as drivable area
                car_park_polygons.append(poly)
                car_park_tokens.append(f"car_park_{len(car_park_tokens)}")

        self._intersections = intersection_polygons
        self._lane_map = OccupancyMap(lane_tokens, lane_polygons)
        self._lane_centers = np.asarray(lane_centers, dtype=object)
        self._lane_center_cache = self._build_lane_center_cache(self._lane_centers)
        self._speed_limits = speed_limits_dict
        if scene_manager is not None:
            self._drivable_area_map = scene_manager.lane_map._drivable_area_map_local
            self._lane_map = getattr(scene_manager.lane_map, "_lane_map_local", self._lane_map)
            self._speed_limits = getattr(scene_manager.lane_map, "_lane_speed_limit_hash", self._speed_limits)
            self._scene_lane_map = self._lane_map
            self._scene_lane_connection_hash = getattr(scene_manager.lane_map, "_lane_connection_hash", None)
        else:
            self._drivable_area_map = OccupancyMap(
                car_park_tokens+lane_tokens+intersection_tokens, 
                car_park_polygons+lane_polygons+intersection_polygons
                )
            self._scene_lane_map = None
            self._scene_lane_connection_hash = None

        self._red_light_map = OccupancyMap(red_light_tokens, red_light_polygons)

        self._drivable_lane_indices = None
        self._route_lane_indices = None
        if self._drivable_area_map is not None and self._lane_map is not None:
            drivable_token_to_idx = {
                token: idx for idx, token in enumerate(self._drivable_area_map.tokens)
            }
            lane_indices = [
                drivable_token_to_idx.get(token, -1) for token in self._lane_map.tokens
            ]
            if lane_indices and all(idx >= 0 for idx in lane_indices):
                self._drivable_lane_indices = np.asarray(lane_indices, dtype=np.int64)

            lane_token_to_idx = {
                token: idx for idx, token in enumerate(self._lane_map.tokens)
            }
            route_lane_indices = [
                lane_token_to_idx.get(token, -1) for token in route_lane_tokens
            ]
            if route_lane_indices and all(idx >= 0 for idx in route_lane_indices):
                self._route_lane_indices = np.asarray(route_lane_indices, dtype=np.int64)

        

        self._num_proposals = trajectories.shape[0]

        # 2. extract static obstacles' polygons
        static_obstacle_polygons = []
        static_obstacle_tokens = []
        static_obstacle_feature = scene_feature.static_obstacle_feature
        self.static_obstacle_position = np.asarray(static_obstacle_feature.static_obstacle_position)
        if len(static_obstacle_feature.static_obstacle_position) > 0:
            static_obstacle_geo = np.asarray(static_obstacle_feature.static_object_dimension)
            static_obstacle_position = self.static_obstacle_position
            hx, hy = static_obstacle_geo[:, 0] + static_inflation, static_obstacle_geo[:, 1] + static_inflation
            offs = np.stack([
                np.stack([-hx, -hy], axis=-1),
                np.stack([-hx,  hy], axis=-1),
                np.stack([ hx,  hy], axis=-1),
                np.stack([ hx, -hy], axis=-1),
            ], axis=1) 
            cosyaw = np.cos(static_obstacle_position[..., 2])[:, None]                                              # [N,1]
            sinyaw = np.sin(static_obstacle_position[..., 2])[:, None]
            x = offs[..., 0]; y = offs[..., 1]          # [N,4]
            xr = cosyaw * x - sinyaw * y
            yr = sinyaw * x + cosyaw * y
            offs = np.stack([xr, yr], axis=-1) 
            obs_poly = offs + static_obstacle_position[:, None, :2]  # [N,4,(x,y)]
            obs_polygons = creation.polygons(obs_poly)
            static_obstacle_polygons.extend(obs_polygons)
            static_obstacle_tokens.extend([f"obstacle_{i}" for i in range(len(obs_polygons))])
            

        # 3. construct future collision map
        self.agent_position = np.array(scene_feature.agent_feature.agent_current_state)
        self._future_collision_map = []
        self._future_collision_meta = []
        future_collision_map_len = max(self._proposal_sampling.num_poses + 1,
                                        self._proposal_sampling.num_poses + 1 + int(self.ttc_period/self._proposal_sampling.interval_length))


        agent_feature = scene_feature.agent_feature
        agent_current_state = np.asarray(agent_feature.agent_current_state, dtype=SCORER_DTYPE)
        agent_geometry = np.asarray(agent_feature.agent_geometry, dtype=SCORER_DTYPE)
        agent_type_all = np.asarray(agent_feature.agent_type)
        num_agents = len(agent_current_state)

        resampled_prediction_state = None
        resampled_prediction_mask = None
        resampled_prediction_gt_state = None
        resampled_prediction_gt_mask = None
        target_future_steps = max(future_collision_map_len - 1, 0)

        if agent_prediction_gt is not None and prediction_mode == "prediction":
            gt_future_state = np.asarray(agent_prediction_gt.agent_future_state, dtype=SCORER_DTYPE)
            gt_future_mask = np.asarray(agent_prediction_gt.agent_future_mask, dtype=bool)
            resampled_prediction_gt_state, resampled_prediction_gt_mask = self._resample_agent_future_states(
                agent_current_state=agent_current_state,
                agent_future_state=gt_future_state,
                agent_future_mask=gt_future_mask,
                target_num_steps=target_future_steps,
                target_dt=self._proposal_sampling.interval_length,
            )

        if agent_prediction is not None and prediction_mode == "prediction":
            pred_future_state = np.asarray(agent_prediction.agent_future_state, dtype=SCORER_DTYPE)
            pred_future_mask = np.asarray(agent_prediction.agent_future_mask, dtype=bool)
            num_pred_agents = min(pred_future_state.shape[0], agent_current_state.shape[0])
            stopped_agent_idx = np.where(agent_current_state[:num_pred_agents, 3] < STOPPED_SPEED_THRESHOLD)[0]
            if stopped_agent_idx.size > 0:
                pred_future_state = pred_future_state.copy()
                pred_future_mask = pred_future_mask.copy()
                pred_future_state[stopped_agent_idx, :, :3] = agent_current_state[stopped_agent_idx, None, :3]
                pred_future_mask[stopped_agent_idx, :] = True

            resampled_prediction_state, resampled_prediction_mask = self._resample_agent_future_states(
                agent_current_state=agent_current_state,
                agent_future_state=pred_future_state,
                agent_future_mask=pred_future_mask,
                target_num_steps=target_future_steps,
                target_dt=self._proposal_sampling.interval_length,
            )

        static_obstacle_count = len(static_obstacle_tokens)
        static_meta = {
            "kind": np.full(static_obstacle_count, COLLISION_KIND_STATIC, dtype=np.int8),
            "x": self.static_obstacle_position[:, 0].astype(SCORER_DTYPE, copy=False) if static_obstacle_count > 0 else np.zeros((0,), dtype=SCORER_DTYPE),
            "y": self.static_obstacle_position[:, 1].astype(SCORER_DTYPE, copy=False) if static_obstacle_count > 0 else np.zeros((0,), dtype=SCORER_DTYPE),
            "yaw": self.static_obstacle_position[:, 2].astype(SCORER_DTYPE, copy=False) if static_obstacle_count > 0 else np.zeros((0,), dtype=SCORER_DTYPE),
            "v": np.zeros((static_obstacle_count,), dtype=SCORER_DTYPE),
            "idx": np.arange(static_obstacle_count, dtype=np.int32),
        }
        for time_idx in range(future_collision_map_len):  
            polygons, tokens = [], []
            meta_parts = []

            # draw surrounding agents' polygons
            
            constant_velocity_buffer_time = 0  # number of time steps to use constant velocity 
            valid_agent_idx = np.arange(num_agents)  # initialize valid agent indices
            if num_agents > 0:
                # 获取当前时刻的 agent 位姿，确保为 float64 ndarray
                if time_idx == 0:
                    agent_current = agent_current_state[:num_agents, :3]
                    agent_previous = agent_current_state[:num_agents, :]
                elif resampled_prediction_gt_state is not None:
                    future_idx = time_idx - 1
                    if future_idx < resampled_prediction_gt_state.shape[1]:
                        agent_current_mask = resampled_prediction_gt_mask[:num_agents, future_idx]
                        valid_agent_idx = np.flatnonzero(agent_current_mask)
                        agent_current = resampled_prediction_gt_state[valid_agent_idx, future_idx, :3]
                    else:
                        valid_agent_idx = np.zeros((0,), dtype=np.int64)
                        agent_current = np.zeros((0, 3), dtype=SCORER_DTYPE)
                elif resampled_prediction_state is not None:
                    future_idx = time_idx - 1
                    if future_idx < resampled_prediction_state.shape[1]:
                        agent_current_mask = resampled_prediction_mask[:num_agents, future_idx]
                        valid_agent_idx = np.flatnonzero(agent_current_mask)
                        agent_current = resampled_prediction_state[valid_agent_idx, future_idx, :3]
                    else:
                        valid_agent_idx = np.zeros((0,), dtype=np.int64)
                        agent_current = np.zeros((0, 3), dtype=SCORER_DTYPE)
                else:
                    # propagate using constant yawrate + acceleration model
                    agent_previous_pose = agent_previous[:, :3]
                    agent_current = agent_previous.copy()
                    agent_speed = agent_previous[:, 3]
                    effective_agent_speed = agent_speed.copy()
                    agent_type = agent_type_all[: len(agent_previous)]
                    effective_agent_speed[agent_type == 2] = effective_agent_speed[agent_type == 2].clip(0.0, 1.4)  # for pedestrians, clip speed to 1.2 m/s to avoid large prediction error, this is based on the observation that most pedestrians in nuPlan walk below 1.2 m/s
                    agent_acceleration = agent_previous[:, 4]
                    agent_yawrate = agent_previous[:, 5]    
                    effective_yawrate = agent_yawrate.copy()
                    # effective_yawrate[agent_type == 2] = 0.0
                    agent_current[:, :2] = agent_previous_pose[:, :2] + np.stack([
                        effective_agent_speed * np.cos(agent_previous_pose[:, 2]),
                        effective_agent_speed * np.sin(agent_previous_pose[:, 2])
                    ], axis=-1) * self._proposal_sampling.interval_length

                    if prediction_mode == "CA" or prediction_mode == "CAYAW":
                        agent_current[:, 3] = np.clip(agent_speed + agent_acceleration * self._proposal_sampling.interval_length, 0.0, 15.0)  # no reverse
                    if prediction_mode == "CYAW" or prediction_mode == "CAYAW":
                        agent_current[:, 2] = agent_previous_pose[:, 2] + effective_yawrate * self._proposal_sampling.interval_length
                    
                    agent_previous = agent_current.copy()
                
                ## DEBUG ###
                # if time_idx == 1:
                #     if agent_prediction_gt is not None:
                #         prediction_to_init_error = np.linalg.norm(agent_current[:,:2] - np.asarray(agent_feature.agent_current_state, dtype=np.float64)[valid_agent_idx, :2], axis=1)
                #         if np.any(prediction_to_init_error > 3.0):
                #             idx = np.where(prediction_to_init_error > 3.0)[0]
                #             print(f"Large prediction to init error found at time_idx {time_idx}, errors: {prediction_to_init_error[idx]}")
                #             print(f"agent_current: {agent_current[idx,:]}")
                #             print(f"agent_init: {np.asarray(agent_feature.agent_current_state, dtype=np.float64)[idx, :3]}")
                ## DEBUG ###

                # construct agent polygons
                agent_geo = agent_geometry[valid_agent_idx]  # [N,2] -> (hx, hy)
                agent_type = agent_type_all[valid_agent_idx]
                pedestrain_buffer = 0.1 * (agent_type == 2)  # [m]
                # agent_longtitudial_buffer = longitudinal_buffer + time_idx * self._proposal_sampling.interval_length * longitudinal_velocity_buffer * np.linalg.norm(agent_current[:, :2], axis=1)  # [N,]
                agent_longitudial_buffer = longitudinal_buffer
                hx, hy = agent_geo[:, 0] + agent_longitudial_buffer + pedestrain_buffer, agent_geo[:, 1] + lateral_buffer + pedestrain_buffer
                offs = np.stack([
                     np.stack([-hx, -hy], axis=-1),
                     np.stack([-hx,  hy], axis=-1),
                     np.stack([ hx,  hy], axis=-1),
                     np.stack([ hx, -hy], axis=-1),
                 ], axis=1) 
                cosyaw = np.cos(agent_current[..., 2])[:, None]                          # [N,1]
                sinyaw = np.sin(agent_current[..., 2])[:, None]
                x = offs[..., 0]; y = offs[..., 1]                                    # [N,4]
                xr = cosyaw * x - sinyaw * y
                yr = sinyaw * x + cosyaw * y
                offs = np.stack([xr, yr], axis=-1) 
                agent_poly = offs + agent_current[:, None, :2]                          # [N,4,(x,y)]
                agent_polygons = creation.polygons(agent_poly)                           # Vectorized shapely Polygons
                polygons.extend(list(agent_polygons))
                tokens.extend([f"agent_{agent_idx}" for agent_idx in valid_agent_idx])
                meta_parts.append(
                    {
                        "kind": np.full(len(valid_agent_idx), COLLISION_KIND_AGENT, dtype=np.int8),
                        "x": agent_current[:, 0].astype(SCORER_DTYPE, copy=False),
                        "y": agent_current[:, 1].astype(SCORER_DTYPE, copy=False),
                        "yaw": agent_current[:, 2].astype(SCORER_DTYPE, copy=False),
                        "v": agent_current[:, 3].astype(SCORER_DTYPE, copy=False) if agent_current.shape[1] > 3 else np.zeros((len(valid_agent_idx),), dtype=SCORER_DTYPE),
                        "idx": valid_agent_idx.astype(np.int32, copy=False),
                    }
                )

                
                    
            polygons.extend(static_obstacle_polygons)
            tokens.extend(static_obstacle_tokens)
            if static_obstacle_count > 0:
                meta_parts.append(static_meta)

            self._future_collision_map.append(OccupancyMap(tokens, polygons))
            if meta_parts:
                self._future_collision_meta.append(
                    {
                        key: np.concatenate([part[key] for part in meta_parts], axis=0)
                        for key in ["kind", "x", "y", "yaw", "v", "idx"]
                    }
                )
            else:
                self._future_collision_meta.append(
                    {
                        "kind": np.zeros((0,), dtype=np.int8),
                        "x": np.zeros((0,), dtype=SCORER_DTYPE),
                        "y": np.zeros((0,), dtype=SCORER_DTYPE),
                        "yaw": np.zeros((0,), dtype=SCORER_DTYPE),
                        "v": np.zeros((0,), dtype=SCORER_DTYPE),
                        "idx": np.zeros((0,), dtype=np.int32),
                    }
                )

        # zero initialize all remaining arrays.
        self._ego_areas = np.zeros(
            (
                self._num_proposals,
                self._proposal_sampling.num_poses + 1,
                len(EgoAreaIndex),
            ),
            dtype=np.bool_,
        )
        self._ego_lane_token_mask = None
        self._multi_metrics = np.ones(
            (len(MultiMetricIndex), self._num_proposals), dtype=np.float32
        )
        self._weighted_metrics = np.zeros(
            (len(WeightedMetricIndex), self._num_proposals), dtype=np.float32
        )
        self._progress_raw = np.zeros(self._num_proposals, dtype=np.float32)
        self._valid_proposal_mask = np.ones(self._num_proposals, dtype=np.bool_)
        self._following_penalty = np.ones(self._num_proposals, dtype=np.float32)

        # initialize infraction arrays with infinity (meaning no infraction occurs)
        self._collision_time_idcs = np.zeros(self._num_proposals, dtype=np.float32)
        self._ttc_time_idcs = np.zeros(self._num_proposals, dtype=np.float32)
        self._collision_time_idcs.fill(np.inf)
        self._ttc_time_idcs.fill(np.inf)

        # ignore token if collision in first frame
        self._ignored_collision_tokens: Set[str] = set()

    def _calculate_ego_area(self) -> None:
        """
        Determines the area of proposals over time.
        Areas are (1) in multiple lanes, (2) non-drivable area, or (3) oncoming traffic
        """

        n_proposals, n_horizon, n_points, _ = self._ego_coords.shape

        coordinates = self._ego_coords.reshape(n_proposals * n_horizon * n_points, 2)
        in_polygons = self._drivable_area_map.points_in_polygons(coordinates,prefilter_with_point_obb=True)
        in_polygons = in_polygons.reshape(
            len(self._drivable_area_map), n_proposals, n_horizon, n_points
        ).transpose(
            1, 2, 0, 3
        )  # shape: n_proposals, n_horizon, n_polygons, n_points

        corners_in_polygon = in_polygons[..., :-1]  # ignore center coordinate
        batch_nondrivable_area_mask = (corners_in_polygon.sum(axis=-2) > 0).sum(axis=-1) < 4
        self._ego_areas[
            batch_nondrivable_area_mask, EgoAreaIndex.NON_DRIVABLE_AREA
        ] = True

        # in_multiple_lanes: if
        # - more than one lane polygon contains at least one corner
        # - no lane polygon contains all corners
        if self._drivable_lane_indices is not None:
            in_lanes = in_polygons[:, :, self._drivable_lane_indices, :]
        else:
            in_lanes = self._lane_map.points_in_polygons(coordinates).reshape(
                len(self._lane_map), n_proposals, n_horizon, n_points
            ).transpose(1, 2, 0, 3)

        corners_in_lanes = in_lanes[..., :-1]
        self._ego_lane_token_mask = in_lanes[..., -1] > 0

        batch_multiple_lanes_mask = (corners_in_lanes.sum(axis=-1) > 0).sum(axis=-1) > 1
        batch_not_single_lanes_mask = np.all(corners_in_lanes.sum(axis=-1) != 4, axis=-1)
        multiple_lanes_mask = np.logical_and(batch_multiple_lanes_mask, batch_not_single_lanes_mask)
        self._ego_areas[multiple_lanes_mask, EgoAreaIndex.MULTIPLE_LANES] = True

        # in_oncoming_traffic: if center is not in any route lane. Fall back to
        # route polygons when route-lane indices are unavailable.
        center_coordinates = self._ego_coords[:, :, BBCoordsIndex.CENTER, :].reshape(
            n_proposals * n_horizon,
            2,
        )
        in_routes = self._route_map.points_in_polygons(center_coordinates)
        center_in_routes = in_routes.reshape(len(self._route_map), n_proposals, n_horizon).transpose(1, 2, 0)
        batch_oncoming_traffic_mask = center_in_routes.sum(axis=-1) == 0
        self._ego_areas[
            batch_oncoming_traffic_mask, EgoAreaIndex.ONCOMING_TRAFFIC
        ] = True

    def _compute_multiple_lanes_mask(
        self,
        corners_in_lanes: npt.NDArray[np.bool_],
        scene_lane_map: Optional[OccupancyMap],
        ego_coords: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.bool_]:
        batch_multiple_lanes_mask = (corners_in_lanes.sum(axis=-1) > 0).sum(axis=-1) > 1
        batch_not_single_lanes_mask = np.all(corners_in_lanes.sum(axis=-1) != 4, axis=-1)
        multiple_lanes_mask = np.logical_and(batch_multiple_lanes_mask, batch_not_single_lanes_mask)

        if (
            scene_lane_map is None
            or len(scene_lane_map) == 0
            or not self._scene_lane_connection_hash
            or not np.any(multiple_lanes_mask)
        ):
            return multiple_lanes_mask

        for proposal_idx, time_idx in np.argwhere(multiple_lanes_mask):
            corner_points = np.asarray(
                ego_coords[proposal_idx, time_idx, :-1, :],
                dtype=np.float64,
            )
            intersected_lanes = scene_lane_map.points_in_polygons(corner_points)
            intersected_lane_indices = np.flatnonzero(intersected_lanes.any(axis=-1))
            if intersected_lane_indices.size <= 1:
                continue

            intersected_lane_ids = [scene_lane_map.tokens[idx] for idx in intersected_lane_indices]
            if self._lanes_belong_to_same_connected_component(intersected_lane_ids):
                multiple_lanes_mask[proposal_idx, time_idx] = False

        return multiple_lanes_mask

    def _lanes_belong_to_same_connected_component(self, lane_ids: List[str]) -> bool:
        if len(lane_ids) <= 1:
            return True

        if not self._scene_lane_connection_hash:
            return False

        connected_component = self._scene_lane_connection_hash.get(lane_ids[0])
        if connected_component is None:
            return False

        return all(lane_id in connected_component for lane_id in lane_ids[1:])

    def _calculate_within_speed_limit(self) -> None:
        """
        Re-implementation of nuPlan's within speed limit metric.
        """

        speed_limit_scores = np.zeros(self._num_proposals, dtype=np.float32)
        timed_speed_limit_scores = np.zeros(
            (self._num_proposals, self._proposal_sampling.num_poses + 1),
            dtype=np.float32,
        )
        valid_mask = self._valid_proposal_mask
        if valid_mask is None:
            valid_mask = np.ones(self._num_proposals, dtype=np.bool_)

        if not np.any(valid_mask):
            self._weighted_metrics[WeightedMetricIndex.SPEED_LIMIT] = speed_limit_scores
            return

        if (
            self._ego_lane_token_mask is None
            or self._speed_limits is None
            or len(self._lane_map) == 0
        ):
            speed_limit_scores[valid_mask] = 1.0
            self._weighted_metrics[WeightedMetricIndex.SPEED_LIMIT] = speed_limit_scores
            return

        lane_speed_limits = np.full(len(self._lane_map.tokens), np.inf, dtype=np.float32)
        for lane_idx, lane_token in enumerate(self._lane_map.tokens):
            speed_limit = self._speed_limits.get(lane_token, np.inf)
            if speed_limit is None:
                continue
            speed_limit = float(speed_limit)
            if np.isfinite(speed_limit) and speed_limit > 0.0:
                lane_speed_limits[lane_idx] = speed_limit

        if not np.isfinite(lane_speed_limits).any():
            speed_limit_scores[valid_mask] = 1.0
            self._weighted_metrics[WeightedMetricIndex.SPEED_LIMIT] = speed_limit_scores
            return

        lane_membership = self._ego_lane_token_mask[valid_mask].astype(bool)
        point_speed_limits = np.min(
            np.where(lane_membership, lane_speed_limits[None, None, :], np.inf),
            axis=-1,
        ).astype(np.float32)

        speeds = np.hypot(
            self._states[valid_mask, ..., StateIndex.VELOCITY_X],
            self._states[valid_mask, ..., StateIndex.VELOCITY_Y],
        ).astype(np.float32)

        overspeed = np.maximum(speeds - point_speed_limits, 0.0)
        overspeed[~np.isfinite(point_speed_limits)] = 0.0

        cumulative_penalty = np.cumsum(
            overspeed * self._proposal_sampling.interval_length,
            axis=-1,
        )
        timed_speed_limit_scores = np.clip(
            1.0 - cumulative_penalty / 6.69,
            0.0,
            1.0,
        ).astype(np.float32)
        speed_limit_scores[valid_mask] = timed_speed_limit_scores[:, -1]

        self._weighted_metrics[WeightedMetricIndex.SPEED_LIMIT] = speed_limit_scores


    def _calculate_no_at_fault_collision(self) -> None:
        """
        Re-implementation of nuPlan's at-fault collision metric.
        """
        no_collision_scores = np.ones(self._num_proposals, dtype=np.float32)
        valid_mask = self._get_active_multi_metric_mask()
        valid_indices = np.flatnonzero(valid_mask)
        if valid_indices.size == 0:
            self._multi_metrics[MultiMetricIndex.NO_COLLISION] = no_collision_scores
            return


        for time_idx in range(self._proposal_sampling.num_poses + 1):
            ego_polygons = self._ego_polygons[valid_indices, time_idx]
            collision_map = self._future_collision_map[time_idx]
            collision_meta = self._future_collision_meta[time_idx]
            intersecting = collision_map.query(
                ego_polygons, predicate="intersects"
            )

            if len(intersecting) == 0:
                continue

            for local_proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                proposal_idx = valid_indices[local_proposal_idx]
                token = collision_map.tokens[geometry_idx]
                if time_idx == 0:
                    self._ignored_collision_tokens.add(token)
                    continue
                
                if token in self._ignored_collision_tokens:
                    continue

                ego_in_multiple_lanes_or_nondrivable_area = (
                    self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                    or self._ego_areas[
                        proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA
                    ]
                )
                geo_polygon = collision_map[token]
                agent_x = float(collision_meta["x"][geometry_idx])
                agent_y = float(collision_meta["y"][geometry_idx])
                agent_yaw = float(collision_meta["yaw"][geometry_idx])
                agent_v = float(collision_meta["v"][geometry_idx])
                
                # classify collision
                collision_type: CollisionType = get_collision_type(
                    state=self._states[proposal_idx, time_idx],
                    ego_polygon=self._ego_polygons[proposal_idx, time_idx],
                    tracked_object_polygon=geo_polygon,
                    object_token=token,
                    object_start_x=agent_x,
                    object_start_y=agent_y,
                    object_start_yaw=agent_yaw,
                    object_velocity=agent_v,
                )
                collisions_at_stopped_track_or_active_front: bool = collision_type in [
                    CollisionType.ACTIVE_FRONT_COLLISION,
                    CollisionType.STOPPED_TRACK_COLLISION,
                ]
                collision_at_lateral: bool = (
                    collision_type == CollisionType.ACTIVE_LATERAL_COLLISION
                )

                # 1. at fault collision
                if collisions_at_stopped_track_or_active_front or (
                    ego_in_multiple_lanes_or_nondrivable_area and collision_at_lateral
                ):
                    no_at_fault_collision_score = (
                        0.0
                        if 'agent' in token
                        else 0.0
                    )
                    no_collision_scores[proposal_idx] = np.minimum(
                        no_collision_scores[proposal_idx], no_at_fault_collision_score
                    )
                    self._collision_time_idcs[proposal_idx] = min(
                        time_idx, self._collision_time_idcs[proposal_idx]
                    )

                    

        self._multi_metrics[MultiMetricIndex.NO_COLLISION] = no_collision_scores

    def _calculate_ttc(self):
        """
        Re-implementation of nuPlan's time-to-collision metric.
        Memory-optimized: stream polygons per (time_idx, future_step) instead of prebuilding all.
        """
        ttc_scores = np.zeros(self._num_proposals, dtype=np.float32)
        timed_ttc_scores = np.zeros(
            (self._num_proposals, self._proposal_sampling.num_poses + 1), dtype=np.float32
        )
        valid_mask = self._valid_proposal_mask
        if valid_mask is None:
            valid_mask = np.ones(self._num_proposals, dtype=np.bool_)

        valid_indices = np.flatnonzero(valid_mask)
        if valid_indices.size == 0:
            self._weighted_metrics[WeightedMetricIndex.TTC] = ttc_scores
            return

        valid_ttc_scores = np.ones(valid_indices.size, dtype=np.float32) * self.ttc_period
        valid_timed_ttc_scores = np.ones(
            (valid_indices.size, self._proposal_sampling.num_poses + 1), dtype=np.float32
        ) * self.ttc_period

        future_time_idcs = np.arange(0, self.ttc_period + 0.1, 0.2, dtype=np.float32)  # K steps

        # Precompute per-step speeds and directions
        speeds = np.hypot(
            self._states[..., StateIndex.VELOCITY_X],
            self._states[..., StateIndex.VELOCITY_Y],
        )  # [N, T+1]
        dxy_per_s = np.stack(
            [
                np.cos(self._states[..., StateIndex.HEADING]) * speeds,
                np.sin(self._states[..., StateIndex.HEADING]) * speeds,
            ],
            axis=-1,
        )  # [N, T+1, 2]

        # Build base rectangles (close polygon by copying front-left to center slot)
        coords_exterior = self._ego_coords.copy()  # [N, T+1, 5, 2]
        coords_exterior[:, :, BBCoordsIndex.CENTER, :] = coords_exterior[:, :, BBCoordsIndex.FRONT_LEFT, :]

        max_time_idx = self._proposal_sampling.num_poses + 1 - len(future_time_idcs)
        num_horizon = self._proposal_sampling.num_poses + 1

        # Stream over time and future steps to keep peak memory low
        for time_idx in range(max_time_idx):
            moving_local_indices = np.flatnonzero(
                speeds[valid_indices, time_idx] >= STOPPED_SPEED_THRESHOLD
            )
            if moving_local_indices.size == 0:
                continue

            # Base corners at current time step: [N, 5, 2]
            active_valid_indices = valid_indices[moving_local_indices]
            base_corners = coords_exterior[active_valid_indices, time_idx]
            base_dirs = dxy_per_s[active_valid_indices, time_idx]
            projected_corners = base_corners[:, None, :, :] + (
                base_dirs[:, None, None, :] * future_time_idcs[None, :, None, None]
            )
            polygons_by_offset = creation.polygons(projected_corners.reshape(-1, projected_corners.shape[-2], 2)).reshape(
                active_valid_indices.size, len(future_time_idcs)
            )

            for future_offset_idx, future_time_idx in enumerate(future_time_idcs):
                polygons_at_time_step = polygons_by_offset[:, future_offset_idx]

                current_time_idx = time_idx + int(future_time_idx / self._proposal_sampling.interval_length)
                if current_time_idx >= len(self._future_collision_map):
                    current_time_idx = len(self._future_collision_map) - 1

                intersecting = self._future_collision_map[current_time_idx].query(
                    polygons_at_time_step, predicate="intersects"
                )
                if len(intersecting) == 0:
                    continue

                collision_map = self._future_collision_map[current_time_idx]
                collision_meta = self._future_collision_meta[current_time_idx]

                for local_proposal_idx, geometry_idx in zip(intersecting[0], intersecting[1]):
                    proposal_idx = active_valid_indices[local_proposal_idx]
                    token = collision_map.tokens[geometry_idx]
                    token_kind = int(collision_meta["kind"][geometry_idx])
                    # Skip red-light or stopped-ego cases
                    if (token_kind == COLLISION_KIND_STATIC) or \
                    (speeds[proposal_idx, time_idx] < STOPPED_SPEED_THRESHOLD):
                        continue
                    
                    if token in self._ignored_collision_tokens:
                        continue

                    ego_in_multiple_lanes_or_nondrivable_area = (
                        self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.MULTIPLE_LANES]
                        or self._ego_areas[proposal_idx, time_idx, EgoAreaIndex.NON_DRIVABLE_AREA]
                    )

                    ego_polygon = self._ego_polygons[proposal_idx, time_idx]
                    track_polygon = collision_map[token]
                    agent_x = float(collision_meta["x"][geometry_idx])
                    agent_y = float(collision_meta["y"][geometry_idx])
                    agent_yaw = float(collision_meta["yaw"][geometry_idx])

                    ego_rear_axle: StateSE2 = StateSE2(*self._states[proposal_idx, time_idx, StateIndex.STATE_SE2])
                    track_state = StateSE2(agent_x, agent_y, agent_yaw)

                    if is_agent_ahead(ego_rear_axle, track_state) or (
                        (
                            ego_in_multiple_lanes_or_nondrivable_area
                            or (False if not self._intersections
                                else [ego_polygon.intersects(intersection) for intersection in self._intersections].count(True) > 0)
                        )
                        and not is_agent_behind(ego_rear_axle, track_state)
                    ):
                        # Update TTC accumulators
                        valid_ttc_scores[local_proposal_idx] = np.minimum(valid_ttc_scores[local_proposal_idx], future_time_idx)
                        valid_timed_ttc_scores[local_proposal_idx, time_idx] = np.minimum(
                            valid_timed_ttc_scores[local_proposal_idx, time_idx], future_time_idx
                        )
                        self._ttc_time_idcs[proposal_idx] = min(time_idx, self._ttc_time_idcs[proposal_idx])

        # Normalize
        valid_timed_ttc_scores = valid_timed_ttc_scores / self.ttc_period
        min_ttc_ratio = self.minimun_ttc / self.ttc_period
        valid_ttc_scores = valid_ttc_scores / self.ttc_period
        valid_ttc_scores[valid_ttc_scores < min_ttc_ratio] = 0.0
        ttc_scores[valid_mask] = valid_ttc_scores
        self._weighted_metrics[WeightedMetricIndex.TTC] = ttc_scores

    def _calculate_progress(self) -> None:
        """
        Re-implementation of nuPlan's progress metric (non-normalized).
        Calculates progress along the centerline.
        """
        multiplicate_metric_scores = self._multi_metrics.prod(axis=0)
        valid_mask = self._valid_proposal_mask
        if valid_mask is None:
            valid_mask = np.ones(self._num_proposals, dtype=np.bool_)
        if not np.any(valid_mask):
            self._weighted_metrics[WeightedMetricIndex.PROGRESS] = np.zeros(self._num_proposals, dtype=np.float32)
            self._progress_raw = np.zeros(self._num_proposals, dtype=np.float32)
            return

        if self.expert_trajectory is not None:
            expert_ls = LineString(self.expert_trajectory[:, :2])
            expert_progress_all = expert_ls.length
            if expert_progress_all < 1.0:
                proposal_progress = np.zeros(self._num_proposals, dtype=np.float32)
                proposal_progress[valid_mask] = 1.0
                self._weighted_metrics[WeightedMetricIndex.PROGRESS] = proposal_progress
                return
            expert_clipped_traj = self.expert_trajectory[:self.original_proposal_sampling.num_poses+1, :2] # match length with ego trajectory
            expert_progress = expert_ls.project(creation.points(expert_clipped_traj)) # [T,]
            expert_total_progress = expert_progress[-1] - expert_progress[0]
            if expert_total_progress < 1.0:
                proposal_progress = np.zeros(self._num_proposals, dtype=np.float32)
                proposal_progress[valid_mask] = 1.0
                self._weighted_metrics[WeightedMetricIndex.PROGRESS] = proposal_progress
                return
            # calculate normalized progress in meter
            ego_points = self._ego_coords[valid_mask, :, BBCoordsIndex.CENTER] # B, T, 2
            ego_progress = expert_ls.project(creation.points(ego_points)) # [B, T]
            ego_total_progress = ego_progress[:, -1] - ego_progress[:, 0]  # [B,]
            ego_total_progress[multiplicate_metric_scores[valid_mask] < VALID_PROPOSAL_THRESHOLD] = 0.0
            max_progress = np.max(ego_total_progress)
            max_progress = min(max_progress, expert_total_progress * 1.2) # cap max progress to 120% of expert progress
            if max_progress < 1.0:
                proposal_progress = np.zeros(self._num_proposals, dtype=np.float32)
                proposal_progress[valid_mask] = 1.0
                self._weighted_metrics[WeightedMetricIndex.PROGRESS] = proposal_progress
                return
            normalized_progress = np.clip(ego_total_progress / max_progress, 0.0, 1.0).astype(np.float32)
            proposal_progress = np.zeros(self._num_proposals, dtype=np.float32)
            proposal_progress[valid_mask] = normalized_progress
            self._weighted_metrics[WeightedMetricIndex.PROGRESS] = proposal_progress
            return

        # calculate raw progress in meter
        ref_xy = np.asarray(self.ref_path[:, :2], dtype=SCORER_DTYPE)
        valid_ref = np.isfinite(ref_xy).all(axis=1)
        ref_xy = ref_xy[valid_ref]
        if ref_xy.shape[0] >= 2:
            diffs = np.diff(ref_xy, axis=0)
            keep = np.ones(ref_xy.shape[0], dtype=bool)
            keep[1:] = np.any(diffs != 0.0, axis=1)
            ref_xy = ref_xy[keep]
        else:
            self._progress_raw = np.zeros(self._num_proposals, dtype=np.float32)
            return
        
        ref_path_ls = LineString(ref_xy)  # convert to shapely LineString
        points = self._ego_coords[valid_mask, :, BBCoordsIndex.CENTER] # B, T, 2
        ego_progress = ref_path_ls.project(creation.points(points))
        ego_progress_from_start = np.maximum(ego_progress - ego_progress[:, [0]], 0.0)
        ego_total_progress = ego_progress_from_start[:, -1]  # [B,]
        ego_total_progress[multiplicate_metric_scores[valid_mask] < VALID_PROPOSAL_THRESHOLD] = 0.0
        max_progress = np.max(ego_total_progress)
        
        initial_speeds = np.hypot(
            self._states[valid_mask, 0, StateIndex.VELOCITY_X],
            self._states[valid_mask, 0, StateIndex.VELOCITY_Y],
        )
        
        horizon_s = self._proposal_sampling.time_horizon
        comfortable_progress_budget = (
            initial_speeds * horizon_s + 0.5 * PROGRESS_SATURATION_ACCEL * horizon_s ** 2
        )
        remaining_progress = np.maximum(ref_path_ls.length - ego_progress[:, 0], 0.0)
        target_progress = np.minimum(remaining_progress, comfortable_progress_budget)
        target_progress = np.minimum(target_progress, max_progress)
        target_progress = np.maximum(target_progress, PROGRESS_SATURATION_MIN)

        normalized_progress = ego_total_progress / target_progress
        normalized_progress = np.clip(normalized_progress, 0.0, 1.0).astype(np.float32)

        proposal_progress = np.zeros(self._num_proposals, dtype=np.float32)
        valid_progress = normalized_progress.copy()
        valid_progress[multiplicate_metric_scores[valid_mask] < VALID_PROPOSAL_THRESHOLD] = 0.0

        proposal_progress[valid_mask] = valid_progress

        self._weighted_metrics[WeightedMetricIndex.PROGRESS] = proposal_progress
        self._progress_raw = np.zeros(self._num_proposals, dtype=np.float32)
        self._progress_raw[valid_mask] = ego_total_progress
        return

    def _calculate_is_comfortable(self) -> None:
        """
        Re-implementation of nuPlan's comfortability metric.
        Revised to quantify the comfortability of planned trajectory.
        """
        comfort_states = self._comfort_states if self._comfort_states is not None else self._states
        if comfort_states is None or comfort_states.shape[0] == 0:
            self._weighted_metrics[WeightedMetricIndex.COMFORTABLE] = np.zeros(self._num_proposals, dtype=np.float32)
            return

        comfort_dt = (
            self.original_proposal_sampling.interval_length
            if self._comfort_states is not None
            else self._proposal_sampling.interval_length
        )
        time_point_s: npt.NDArray[np.float32] = (
            np.arange(comfort_states.shape[1], dtype=np.float32) * comfort_dt
        )

        comfortable_value = mean_ratio_of_trajectories(comfort_states, time_point_s)
        # logging.info(f"comfort metrics is {comfortable_value}")
        # normalize comfort eval:  good [0 1] bad -> bad [0 1] good
        is_comfort = np.all(comfortable_value<1 , axis=-1)
        comfort_metrics = np.mean(comfortable_value, axis=-1)
        # comfort_metrics_bounded = np.where(is_comfort, comfort_metrics, 1)
        # comfort_metrics_normalized = np.abs(comfort_metrics_bounded - 1)
        comfort_metrics_bounded = np.where(is_comfort, 1, 0)
        comfort_metrics_normalized = comfort_metrics_bounded
        self._weighted_metrics[WeightedMetricIndex.COMFORTABLE] = np.asarray(
            comfort_metrics_normalized,
            dtype=np.float32,
        )

    def _calculate_drivable_area_compliance(self) -> None:
        """
        Re-implementation of nuPlan's drivable area compliance metric
        """
        # Base score: fully on drivable area
        drivable_area_compliance_scores = np.ones(self._num_proposals, dtype=np.float32)

        # Per-step traveled distance along the trajectory center
        center_coordinates = self._ego_coords[:, :, BBCoordsIndex.CENTER]
        step_distances = np.zeros(
            (self._num_proposals, self._proposal_sampling.num_poses + 1),
            dtype=np.float32,
        )
        step_distances[:, 1:] = (
            (center_coordinates[:, 1:] - center_coordinates[:, :-1]) ** 2.0
        ).sum(axis=-1) ** 0.5

        # Off-road mask over time
        off_road_mask = self._ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA]

        # Cumulative off-road distance per proposal over the full horizon.
        # Keep this untouched unless every proposal is off-road, so feasible on-road
        # trajectories preserve the original scoring behavior.
        full_horizon_off_road_distance = (
            step_distances * off_road_mask.astype(np.float32)
        ).sum(axis=-1)
        off_road_distance = full_horizon_off_road_distance.copy()

        if np.all(full_horizon_off_road_distance > DRIVABLE_AREA_EPS):
            short_horizon_steps = int(
                np.ceil(DRIVABLE_AREA_SHORT_HORIZON_S / self._proposal_sampling.interval_length)
            )
            short_horizon_steps = min(short_horizon_steps, self._proposal_sampling.num_poses)
            short_horizon_slice = slice(0, short_horizon_steps + 1)
            off_road_distance = (
                step_distances[:, short_horizon_slice]
                * off_road_mask[:, short_horizon_slice].astype(np.float32)
            ).sum(axis=-1)

        # For trajectories that leave drivable area:
        #  - if total off-road distance >= 2m: score = 0
        #  - if 0 < distance < 2m: score in (0, 0.1], decreasing linearly with distance
        # Fully on-road trajectories keep score 1.0
        small_violation_mask = (off_road_distance > 0.0) & (off_road_distance < 2.0)
        large_violation_mask = off_road_distance >= 2.0

        # Map (0, 2m) -> (0.1, 0]
        clipped_dist = np.clip(off_road_distance, 0.0, 2.0)
        partial_scores = 0.1 * (1.0 - clipped_dist / 2.0)

        drivable_area_compliance_scores[small_violation_mask] = partial_scores[small_violation_mask]
        drivable_area_compliance_scores[large_violation_mask] = 0.0

        self._multi_metrics[MultiMetricIndex.DRIVABLE_AREA] = drivable_area_compliance_scores

    def _calculate_driving_direction_compliance(self) -> None:
        """
        Re-implementation of nuPlan's driving direction compliance metric
        """
        driving_direction_compliance_scores = np.ones(self._num_proposals, dtype=np.float32)
        valid_mask = self._get_active_multi_metric_mask()
        if not np.any(valid_mask):
            self._multi_metrics[
                MultiMetricIndex.DRIVING_DIRECTION
            ] = driving_direction_compliance_scores
            return

        center_coordinates = self._ego_coords[valid_mask, :, BBCoordsIndex.CENTER]
        step_progress = np.zeros(
            (int(valid_mask.sum()), self._proposal_sampling.num_poses + 1),
            dtype=np.float32,
        )
        step_progress[:, 1:] = (
            (center_coordinates[:, 1:] - center_coordinates[:, :-1]) ** 2.0
        ).sum(axis=-1) ** 0.5

        # mask out progress along the driving direction
        oncoming_traffic_masks = self._ego_areas[valid_mask, :, EgoAreaIndex.ONCOMING_TRAFFIC]
        masked_progress = step_progress * oncoming_traffic_masks.astype(np.float32)
        cumulative_progress = np.cumsum(masked_progress, axis=1, dtype=np.float32)
        previous_cumulative = np.concatenate(
            [np.zeros((int(valid_mask.sum()), 1), dtype=np.float32), cumulative_progress[:, :-1]],
            axis=1,
        )
        segment_starts = oncoming_traffic_masks.copy()
        segment_starts[:, 1:] &= ~oncoming_traffic_masks[:, :-1]
        time_indices = np.arange(segment_starts.shape[1], dtype=np.int32)[None, :]
        start_indices = np.where(segment_starts, time_indices, -1)
        last_start_indices = np.maximum.accumulate(start_indices, axis=1)
        run_bases = np.take_along_axis(
            previous_cumulative,
            np.maximum(last_start_indices, 0),
            axis=1,
        )
        contiguous_oncoming_progress = np.where(
            oncoming_traffic_masks,
            cumulative_progress - run_bases,
            0.0,
        )
        max_oncoming_traffic_progress = contiguous_oncoming_progress.max(axis=1)

        valid_driving_direction_scores = np.where(
            max_oncoming_traffic_progress < DRIVING_DIRECTION_COMPLIANCE_THRESHOLD,
            1.0,
            np.where(
                max_oncoming_traffic_progress < DRIVING_DIRECTION_VIOLATION_THRESHOLD,
                0.6,
                0.1,
            ),
        ).astype(np.float32)

        # calculate driving direction consistency with expert trajectory
        if self.expert_trajectory is not None:
            max_yaw_diff = np.pi / 3  # 60 degrees
            trajectories_end_points = self._states[valid_mask, -1, :3]  # [N, 3]
            endpoints_xy = trajectories_end_points[:, :2]  # [N,2]
            endpoints_yaw = trajectories_end_points[:, 2]  # [N,]

            expert_xy = self.expert_trajectory[:, :2]
            expert_yaw = self.expert_trajectory[:, 2]

            distance = np.linalg.norm(endpoints_xy[:, np.newaxis, :] - expert_xy[np.newaxis, :, :], axis=-1)  # [N, T_expert]
            nearest_expert_idx = np.argmin(distance, axis=1)  # [N,]
            neareat_expert_dist = np.min(distance, axis=1)  # [N,]
            expect_yaw = expert_yaw[nearest_expert_idx]  # [N,]
            yaw_diff = np.abs(np.arctan2(np.sin(endpoints_yaw - expect_yaw), np.cos(endpoints_yaw - expect_yaw)))  # [N,]
            yaw_diff[yaw_diff < max_yaw_diff] = 0.0  # within threshold
            consistent = np.clip(1.0 - yaw_diff / (np.pi/2), 0.0, 1.0)  # [N,] normalize to [0,1]
            consistent[neareat_expert_dist > 10.0] = 0.0  # 超过距离阈值认为不一致
            valid_driving_direction_scores = valid_driving_direction_scores * consistent

        driving_direction_compliance_scores[valid_mask] = valid_driving_direction_scores
        
        # store results
        self._multi_metrics[
            MultiMetricIndex.DRIVING_DIRECTION
        ] = driving_direction_compliance_scores

    def _calculate_within_lane_compliance(self) -> None:
        """
        
        """
        valid_mask = self._get_active_multi_metric_mask()
        center_coordinates = self._ego_coords[:, :, BBCoordsIndex.CENTER]
        cum_progress = np.zeros(
            (self._num_proposals, self._proposal_sampling.num_poses + 1),
            dtype=np.float32,
        )
        cum_progress[:, 1:] = (
            (center_coordinates[:, 1:] - center_coordinates[:, :-1]) ** 2.0
        ).sum(axis=-1) ** 0.5

        # mask out progress along the driving direction
        multi_lane_masks = self._ego_areas[:, :, EgoAreaIndex.MULTIPLE_LANES]
        cum_progress[~multi_lane_masks] = 0.0

        within_lane_compliance_scores = np.ones(
            self._num_proposals, dtype=np.float32
        )
        
        # TODO: enable multi-lane compliance metric
        # for proposal_idx in range(self._num_proposals):
        #     multi_lane_progress, multi_lane_mask = (
        #         cum_progress[proposal_idx],
        #         multi_lane_masks[proposal_idx],
        #     )

        #     # split progress whenever ego changes traffic direction
        #     multi_lane_progress_splits = np.split(
        #         multi_lane_progress,
        #         np.where(np.diff(multi_lane_mask))[0] + 1,
        #     )

        #     # sum up progress of splitted intervals
        #     # Note: splits along the driving direction will have a sum of zero.
        #     max_multi_lane_progress = max(
        #         multi_lane_progress.sum()
        #         for multi_lane_progress in multi_lane_progress_splits
        #     )

        #     if max_multi_lane_progress < MULTI_LANE_COMPLIANCE_THRESHOLD:
        #         within_lane_compliance_scores[proposal_idx] = 1.0
        #     elif max_multi_lane_progress < MULTI_LANE_VIOLATION_THRESHOLD:
        #         within_lane_compliance_scores[proposal_idx] = 0.90
        #     else:
        #         within_lane_compliance_scores[proposal_idx] = 0.80

        within_lane_compliance_scores[~valid_mask] = 1.0
        self._multi_metrics[
            MultiMetricIndex.WITHIN_LANE
        ] = within_lane_compliance_scores

    def _calculate_red_light_compliance(self) -> None:
        """Penalize proposals continuously by how much progress they make while inside a route red-light region."""
        red_light_scores = np.ones(self._num_proposals, dtype=np.float32)
        valid_mask = self._get_active_multi_metric_mask()
        if not np.any(valid_mask):
            self._multi_metrics[MultiMetricIndex.RED_LIGHT_COMPLIANCE] = red_light_scores
            return

        if self._red_light_map is None or len(self._red_light_map) == 0:
            self._multi_metrics[MultiMetricIndex.RED_LIGHT_COMPLIANCE] = red_light_scores
            return

        valid_indices = np.flatnonzero(valid_mask)
        num_steps = min(
            self._proposal_sampling.num_poses + 1,
            self._ego_coords.shape[1],
            self._ego_polygons.shape[1],
        )
        center_coordinates = self._ego_coords[valid_mask, :num_steps, BBCoordsIndex.CENTER]
        step_progress = np.zeros(
            (int(valid_mask.sum()), num_steps),
            dtype=np.float32,
        )
        step_progress[:, 1:] = (
            (center_coordinates[:, 1:] - center_coordinates[:, :-1]) ** 2.0
        ).sum(axis=-1) ** 0.5

        red_light_mask = np.zeros(
            (int(valid_mask.sum()), num_steps),
            dtype=np.bool_,
        )
        for time_idx in range(num_steps):
            ego_polygons = self._ego_polygons[valid_indices, time_idx]
            intersecting = self._red_light_map.query(
                ego_polygons,
                predicate="intersects",
            )
            if len(intersecting) == 0:
                continue
            red_light_mask[intersecting[0], time_idx] = True

        red_light_progress = (step_progress * red_light_mask.astype(np.float32)).sum(axis=-1)
        violating = red_light_progress > RED_LIGHT_PROGRESS_TOLERANCE
        if np.any(violating):
            excess_progress = red_light_progress[violating] - RED_LIGHT_PROGRESS_TOLERANCE
            red_light_scores[valid_mask] = 1.0
            red_light_scores[np.flatnonzero(valid_mask)[violating]] = np.maximum(
                np.exp(-RED_LIGHT_PENALTY_ALPHA * excess_progress).astype(np.float32),
                RED_LIGHT_MIN_SCORE,
            )

        self._multi_metrics[MultiMetricIndex.RED_LIGHT_COMPLIANCE] = red_light_scores

    def _calculate_lane_center_distance(self) -> None:
        """
        Evaluate how well each proposal settles near a lane center near the trajectory end.

        Score definition:
        1. For every ego center point, compute the distance to the nearest lane center.
        2. Average only over a tail window near the trajectory end for each proposal.
        3. Apply inverse min-max normalization over the batch:
        smaller tail-window mean distance -> higher score.
        """
        proposal_scores = np.zeros(self._num_proposals, dtype=np.float32)
        heading_scores = np.zeros(self._num_proposals, dtype=np.float32)
        valid_mask = self._valid_proposal_mask
        if valid_mask is None:
            valid_mask = np.ones(self._num_proposals, dtype=np.bool_)

        if self._lane_centers is None or self._lane_centers.size == 0 or not self._lane_center_cache:
            proposal_scores[valid_mask] = 1.0
            heading_scores[valid_mask] = 1.0
            self._weighted_metrics[WeightedMetricIndex.LANE_CENTER_DISTANCE] = proposal_scores
            self._weighted_metrics[WeightedMetricIndex.HEADING_COMPLIANCE] = heading_scores
            return

        if not np.any(valid_mask):
            self._weighted_metrics[WeightedMetricIndex.LANE_CENTER_DISTANCE] = proposal_scores
            self._weighted_metrics[WeightedMetricIndex.HEADING_COMPLIANCE] = heading_scores
            return

        tail_window_steps = max(
            int(np.ceil(LANE_CENTER_DISTANCE_TAIL_WINDOW_S / self._proposal_sampling.interval_length)),
            1,
        )
        tail_window_steps = min(tail_window_steps, self._proposal_sampling.num_poses + 1)
        tail_slice = slice(
            self._proposal_sampling.num_poses + 1 - tail_window_steps,
            self._proposal_sampling.num_poses + 1,
        )

        center_coordinates = np.asarray(
            self._ego_coords[valid_mask, tail_slice, BBCoordsIndex.CENTER, :],
            dtype=SCORER_DTYPE,
        )  # [N_valid, T_tail, 2]

        if center_coordinates.size == 0:
            self._weighted_metrics[WeightedMetricIndex.LANE_CENTER_DISTANCE] = proposal_scores
            self._weighted_metrics[WeightedMetricIndex.HEADING_COMPLIANCE] = heading_scores
            return

        flat_points_xy = center_coordinates.reshape(-1, 2)
        valid_mask = np.isfinite(flat_points_xy).all(axis=-1)
        nearest_distances = np.full(flat_points_xy.shape[0], np.inf, dtype=np.float32)
        nearest_headings = np.zeros(flat_points_xy.shape[0], dtype=np.float32)

        if np.any(valid_mask):
            flat_valid_indices = np.flatnonzero(valid_mask)
            valid_points_xy = flat_points_xy[flat_valid_indices]
            nearest_lane_idx = np.zeros(flat_valid_indices.size, dtype=np.int32)
            for start_idx in range(0, flat_valid_indices.size, LANE_CENTER_DISTANCE_CHUNK):
                end_idx = min(start_idx + LANE_CENTER_DISTANCE_CHUNK, flat_valid_indices.size)
                valid_points = creation.points(valid_points_xy[start_idx:end_idx])
                lane_distance_matrix = np.asarray(
                    distance(self._lane_centers[:, None], valid_points[None, :]),
                    dtype=np.float32,
                )
                nearest_lane_idx[start_idx:end_idx] = lane_distance_matrix.argmin(axis=0)

            for lane_idx in np.unique(nearest_lane_idx):
                lane_point_mask = nearest_lane_idx == lane_idx
                lane_point_indices = flat_valid_indices[lane_point_mask]
                lane_distances, lane_headings = self._project_points_to_polyline(
                    flat_points_xy[lane_point_indices],
                    self._lane_center_cache[lane_idx],
                )
                nearest_distances[lane_point_indices] = lane_distances
                nearest_headings[lane_point_indices] = lane_headings

        nearest_distances = nearest_distances.reshape(
            center_coordinates.shape[0],
            center_coordinates.shape[1],
        )
        nearest_headings = nearest_headings.reshape(
            center_coordinates.shape[0],
            center_coordinates.shape[1],
        )

        finite_mask = np.isfinite(nearest_distances)
        if not np.any(finite_mask):
            self._weighted_metrics[WeightedMetricIndex.LANE_CENTER_DISTANCE] = proposal_scores
            self._weighted_metrics[WeightedMetricIndex.HEADING_COMPLIANCE] = heading_scores
            return

        # Replace invalid entries with the worst observed finite distance, so they
        # do not become artificially good during normalization.
        worst_distance = float(nearest_distances[finite_mask].max())
        nearest_distances[~finite_mask] = worst_distance
        mean_distances = nearest_distances.mean(axis=-1)

        # Proposal-level inverse min-max normalization.
        min_mean_distance = float(mean_distances.min())
        max_mean_distance = float(mean_distances.max())
        if max_mean_distance - min_mean_distance > 1e-6 and max_mean_distance > 0.2:
            proposal_scores_valid = 1.0 - (
                (mean_distances - min_mean_distance)
                / (max_mean_distance - min_mean_distance)
            )
        else:
            proposal_scores_valid = np.ones(mean_distances.shape[0], dtype=np.float32)

        proposal_scores_valid = np.clip(proposal_scores_valid, 0.0, 1.0).astype(np.float32)

        heading_scores_valid = np.ones(mean_distances.shape[0], dtype=np.float32)

        # Keep consistency with the rest of the scorer: proposals that already fail
        # multiplicative metrics should not keep a positive alignment reward.
        multiplicative_scores = self._multi_metrics.prod(axis=0)
        invalid_proposals = multiplicative_scores < 1e-3
        proposal_scores[self._valid_proposal_mask] = proposal_scores_valid
        heading_scores[self._valid_proposal_mask] = heading_scores_valid

        proposal_scores[invalid_proposals] = 0.0
        heading_scores[invalid_proposals] = 0.0

        self._weighted_metrics[WeightedMetricIndex.LANE_CENTER_DISTANCE] = proposal_scores
        self._weighted_metrics[WeightedMetricIndex.HEADING_COMPLIANCE] = heading_scores

def evaluate_scene(args: List[Dict[str, Union[npt.NDArray, TrajectorySampling, SceneFeature, AgentPrediction]]]) -> List[Optional[Any]]:
    

    def evaluate_scene_internal(args: List[Dict[str, Union[npt.NDArray, TrajectorySampling, SceneFeature, AgentPrediction]]]) -> List[Optional[Any]]:
        
        # process = psutil.Process(os.getpid())
        node_id = int(os.environ.get("NODE_RANK", 0))
        thread_id = str(uuid.uuid4())

        scene_features: List[SceneFeature] = [a["scene_feature"] for a in args]
        agent_predictions: List[AgentPrediction] = [a["agent_prediction"] for a in args]
        agent_predictions_gts: List[AgentPrediction] = [a["agent_prediction_gt"] for a in args]
        proposal_sampling: TrajectorySampling = args[0]["proposal_sampling"]
        trajectories: npt.NDArray[np.float64] = np.array([a["trajectories"] for a in args])
        expert_trajectories: Optional[List[Optional[npt.NDArray[np.float64]]]] = [a["expert_trajectory"] for a in args] if "expert_trajectory" in args[0] else None
        ref_paths: Optional[List[Optional[npt.NDArray[np.float64]]]] = [a["ref_path"] for a in args] if "ref_path" in args[0] else None
        batch_idxs: List[int] = [a["batch_id"] for a in args]
        discount_factor = args[0]["discount_factor"]
        
        simulator = BatchSimulator(
            proposal_sampling=proposal_sampling,
            default_dt=DEFAULT_SIMULATION_DT,
        )
        evaluator = BatchEvaluator(
            proposal_sampling,
            default_dt=DEFAULT_SIMULATION_DT,
        )
        all_scores = []
        if expert_trajectories is None:
            expert_trajectories = [None] * len(args)
        if ref_paths is None:
            ref_paths = [None] * len(args)

        for scene_feature, agent_prediction, agent_prediction_gt, trajs, expert_traj, ref_path, batch_idx in zip(
            scene_features,
            agent_predictions,
            agent_predictions_gts,
            trajectories,
            expert_trajectories,
            ref_paths,
            batch_idxs,
        ):
            # add current state to trajectories
            current_state = np.zeros((trajs.shape[0],1,trajs.shape[2]), dtype=trajs.dtype)
            current_velocity = scene_feature.ego_feature.ego_current_state[3]
            current_acceleration = scene_feature.ego_feature.ego_current_state[4]
            current_yaw_rate = scene_feature.ego_feature.ego_current_state[5]
            current_state[:,0,3] = current_velocity
            current_state[:,0,4] = current_acceleration
            current_state[:,0,5] = current_yaw_rate
            trajs = np.concatenate([current_state, trajs], axis=1)  # [N, T+1, state_size]

            # simulate to get necessary intermediate results
            simulated_trajs = simulator.simulate(trajs)

            scores = evaluator.batch_evaluate(
                simulated_trajs,
                scene_feature,
                agent_prediction,
                agent_prediction_gt,
                expert_trajectory=expert_traj,
                ref_path=ref_path,
                discount_factor=discount_factor,
            )
            all_scores.append((batch_idx, scores, simulated_trajs))

        return all_scores
    
    results = evaluate_scene_internal(args)

    gc.collect()

    return results

def parallel_evaluate(
    worker: WorkerPool,
    trajectories: npt.NDArray[np.float64],
    proposal_sampling: TrajectorySampling,
    scene_feature: Optional[List[SceneFeature]] = None,
    agent_prediction_gt: Optional[List[AgentPrediction]] = None,
    agent_prediction: Optional[List[AgentPrediction]] = None,
    expert_trajectory: Optional[List[npt.NDArray[np.float64]]] = None,
    ref_path: Optional[List[npt.NDArray[np.float64]]] = None,
    discount_factor: float = 1.00,
) -> Dict[str, npt.NDArray[np.float64]]:
    """
    Scores proposal similar to nuPlan's closed-loop metrics
    :param trajectories: array representation of trajectories # [B, N, T, state_size]
    :param scene_feature: list of SceneFeature object containing scene information [B]
    :param agent_prediction: list of AgentPrediction object containing predicted agent states [B]
    :param proposal_sampling: Sampling parameters for proposals
    :param expert_trajectory: list of expert trajectories [B]
    :param ref_path: list of reference paths [B]
    :param discount_factor: discount factor for rewards aggregation, should be in (0, 1]
    :return: dict containing score of each proposal

    """
    batch_size = len(trajectories)
    if scene_feature is None:
        scene_feature = [None] * batch_size
    if agent_prediction is None:
        agent_prediction = [None] * batch_size
    if agent_prediction_gt is None:
        agent_prediction_gt = [None] * batch_size
    if ref_path is None:
        ref_path = [None] * batch_size

    data_points = [ {"scene_feature": scene_feature, 
                     "agent_prediction": agent_prediction,  
                     "agent_prediction_gt": agent_prediction_gt,
                     "proposal_sampling": proposal_sampling,
                     "trajectories": trajs,
                     "expert_trajectory": expert_trajectory[idx] if expert_trajectory is not None else None,
                     "ref_path": ref_path[idx],
                     "batch_id": idx,
                     "discount_factor": discount_factor,
                     } 
        for scene_feature, agent_prediction, agent_prediction_gt, trajs, idx in zip(
            scene_feature, agent_prediction, agent_prediction_gt, trajectories, range(len(trajectories))
            )
        ]
    
    results = worker_map(worker, evaluate_scene, data_points)

    trajectory_scores = [None] * batch_size
    multi_metrics = [None] * batch_size
    simulated_trajs = [None] * batch_size
    # timed_comfort_score = [None] * batch_size # for debugging purpose only
    for result in results:
        batch_idx, scores_dict, simulated_traj = result
        trajectory_scores[batch_idx] = scores_dict['aggregate_scores']
        multi_metrics[batch_idx] = scores_dict.get('multi_metrics', None)
        simulated_trajs[batch_idx] = simulated_traj
        # timed_comfort_score[batch_idx] = scores_dict.get('timed_scores', None) # for debugging purpose only 
    return {"aggregate_scores": trajectory_scores,
            "multi_metrics": multi_metrics,
            "simulated_trajs": np.array(simulated_trajs),
            # 'timed_comfort_score': timed_comfort_score   # for debugging purpose only TODO: DELETE
            }
