from __future__ import annotations

from collections import deque
import logging
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

from lpl_planner.planning.scene.scene_feature.features import (
    AgentPrediction,
    AnchorIndice,
    AnchorScores,
    SceneFeature,
    Trajectory,
)
from lpl_planner.planning.scene.scene_manager import SceneManager
from lpl_planner.planning.scene.trajectory_library import get_trajectory_from_scenario
from lpl_planner.training.dataset.dataset_utils import dump_feature_target_to_pickle


logger = logging.getLogger(__name__)


def extract_planner_state_dict(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], Mapping):
        state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Unsupported checkpoint type: {type(state_dict)!r}")

    def strip_prefix(state_dict, prefix):
        plen = len(prefix)
        return {k[plen:]: v for k, v in state_dict.items() if k.startswith(prefix)}
    
    sub_state = {}
    for p in ['model.', 'policy.']:
        stripped_state = strip_prefix(state_dict, p)
        if len(stripped_state) > 0:
            prefix = p
            sub_state = stripped_state
            break
    if not sub_state:
        raise KeyError(f"No matching keys for prefixes {['model.', 'policy.']}. Available top-level keys (sample): "
                    f"{list(state_dict.keys())[:5]}"
                    )
    
    state_dict = {k.replace(prefix, ""): v for k, v in sub_state.items()}

    return state_dict


def build_expert_trajectory_feature(
    scenario: AbstractScenario,
    iteration: int,
    time_horizon: float,
    time_interval: float,
) -> Trajectory:
    expert_trajectory = get_trajectory_from_scenario(
        scenario,
        run_step=iteration,
        time_horizon=time_horizon,
        time_interval=time_interval,
    )
    return Trajectory(data=np.asarray(expert_trajectory, dtype=np.float32))


def save_rollout_sample(
    output_dir: Path,
    scene_feature: SceneFeature,
    agent_prediction: AgentPrediction,
    expert_trajectory: Trajectory,
    anchor_index: Optional[np.ndarray] = None,
    anchor_score: Optional[np.ndarray] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    dump_feature_target_to_pickle(output_dir / "scene_feature.gz", scene_feature.serialize())
    dump_feature_target_to_pickle(output_dir / "agent_prediction.gz", agent_prediction.serialize())
    dump_feature_target_to_pickle(output_dir / "expert_trajectory.gz", expert_trajectory.serialize())

    if anchor_index is not None and anchor_score is not None:
        anchor_indice = AnchorIndice(indice=np.asarray(anchor_index, dtype=np.int32).reshape(-1))
        anchor_scores = AnchorScores(aggregated_scores=np.asarray(anchor_score, dtype=np.float32).reshape(-1))
        dump_feature_target_to_pickle(output_dir / "anchor_indice.gz", anchor_indice.serialize())
        dump_feature_target_to_pickle(output_dir / "anchor_scores.gz", anchor_scores.serialize())


class LightweightLocalDrivingEnv:
    def __init__(self, cfg: DictConfig, scenario: Optional[AbstractScenario] = None):
        self.cfg = cfg
        self.scenario = scenario
        self.proposal_sampling = cfg.model.future_sampling
        self.dt = float(getattr(cfg, "rollout_dt", 0.1))
        self.planning_step = int(getattr(cfg, "planning_step", 80))
        rollout_max_steps = getattr(cfg, "rollout_max_steps", None)
        if rollout_max_steps in {None, "", "None"}:
            rollout_max_steps = getattr(cfg, "episode_len", None)
        self._configured_max_steps: Optional[int] = None if rollout_max_steps in {None, "", "None"} else int(rollout_max_steps)
        self.max_steps = 10**9
        self.current_iteration = 0
        self.scenario_steps = 0

        self.sm = self._build_scene_manager()
        self._ego_history_maxlen = max(int(round(self.sm.history_sampling.time_horizon / self.dt)) + 1, 1)
        self.ego_history: Deque[EgoState] = deque(maxlen=self._ego_history_maxlen)
        self.ctrl: Optional[TwoStageController] = None
        if scenario is not None:
            self.ctrl = instantiate(cfg.ego_controller, scenario=scenario)

    def _build_scene_manager(self) -> SceneManager:
        return SceneManager(
            time_step=self.proposal_sampling.interval_length,
            simluate_expert_trajectory=False,
            use_ref_path=True,
        )

    def set_scenario(self, scenario: AbstractScenario) -> None:
        self.scenario = scenario
        self.ctrl = instantiate(self.cfg.ego_controller, scenario=scenario)
        self.ego_history = deque(maxlen=self._ego_history_maxlen)

    def _set_ego_history(self, ego_states: List[EgoState]) -> None:
        self.ego_history = deque(ego_states[-self._ego_history_maxlen :], maxlen=self._ego_history_maxlen)

    def _initialize_ego_history(self, initial_ego_state: EgoState) -> None:
        if self.scenario is None:
            self._set_ego_history([initial_ego_state])
            return

        past_ego_states = list(
            self.scenario.get_ego_past_trajectory(
                iteration=0,
                time_horizon=3.0,
                num_samples=30,
            )
        )
        past_ego_states = [ego_state for ego_state in past_ego_states if ego_state is not None]
        past_ego_states.append(initial_ego_state)
        self._set_ego_history(past_ego_states)

    def append_ego_state(self, ego_state: EgoState) -> None:
        self.ego_history.append(ego_state)

    def get_ego_history(self) -> List[EgoState]:
        return list(self.ego_history)

    def _extract_scene_feature(self, ego_state: EgoState, iteration: int) -> SceneFeature:
        if self.scenario is None:
            raise RuntimeError("Scenario must be set before extracting scene features.")
        if self.sm.lane_map.initialized:
            self.sm.lane_map.step_with_planner_init(
                ego_state=ego_state,
                scenario=self.scenario,
                iteration=iteration,
            )
        feature_dict, _ = self.sm.extract_feature_target_from_scenario(
            scenario=self.scenario,
            iteration=iteration,
            ego_state=ego_state,
            ego_history=self.get_ego_history(),
            use_route_correction=True,
        )
        return SceneFeature.deserialize(feature_dict)

    def reset(self) -> SceneFeature:
        if self.scenario is None:
            raise RuntimeError("Scenario must be set before reset().")
        if self.ctrl is None:
            self.ctrl = instantiate(self.cfg.ego_controller, scenario=self.scenario)

        self.sm = self._build_scene_manager()
        self.ctrl.reset()
        self.current_iteration = 0
        self.scenario_steps = self.scenario.get_number_of_iterations()
        scenario_max_steps = max(int(self.scenario_steps) - 1, 0)
        self.max_steps = scenario_max_steps if self._configured_max_steps is None else min(self._configured_max_steps, scenario_max_steps)

        initial_ego_state = self.scenario.get_ego_state_at_iteration(0)
        self._initialize_ego_history(initial_ego_state)
        return self._extract_scene_feature(initial_ego_state, iteration=0)

    def get_state(self) -> EgoState:
        if self.ctrl is None:
            raise RuntimeError("Controller is not initialized.")
        return self.ctrl.get_state()

    def step(self, trajectory: AbstractTrajectory) -> tuple[SceneFeature, bool]:
        if self.scenario is None or self.ctrl is None:
            raise RuntimeError("Environment is not initialized.")
        if self.current_iteration >= self.max_steps:
            raise RuntimeError("Rollout already reached the terminal iteration.")

        self.current_iteration += 1
        current_state = self.ctrl.get_state()
        current_time_point = self.scenario.get_time_point(self.current_iteration - 1)
        next_time_point = self.scenario.get_time_point(self.current_iteration)
        current_iteration = SimulationIteration(current_time_point, self.current_iteration - 1)
        next_iteration = SimulationIteration(next_time_point, self.current_iteration)
        self.ctrl.update_state(current_iteration, next_iteration, current_state, trajectory)

        next_state = self.ctrl.get_state()
        self.append_ego_state(next_state)
        next_scene_feature = self._extract_scene_feature(next_state, iteration=self.current_iteration)
        done = self.current_iteration >= self.max_steps
        return next_scene_feature, done


def resolve_rollout_root(cfg: DictConfig) -> Path:
    root_dir = getattr(cfg, "rollout_cache_dir", None)
    if root_dir is None:
        root_dir = getattr(cfg, "temp_cache_path", "rollout_cache")
    return Path(root_dir)
