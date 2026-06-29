from typing import Any, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimeDuration, TimePoint
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from .utils.kinematic_bicycle import BatchKinematicBicycleModel
from .utils.control import BatchLQRTracker

from .utils.control_utils import DynamicStateIndex, StateIndex, trajectories_to_states


DEFAULT_SIMULATION_DT = 0.1
SIM_DTYPE = np.float32

class BatchSimulator:
    """
    Borrowed and revised from tu_garage
    Re-implementation of nuPlan's simulation pipeline. Enables batch-wise simulation.
    """

    def __init__(
        self,
        proposal_sampling: TrajectorySampling,
        default_dt: float = DEFAULT_SIMULATION_DT,
        tracker: Optional[Any] = None,
    ):
        """
        Constructor of PDMSimulator.
        :param proposal_sampling: Sampling parameters for proposals
        """

        # time parameters
        self._proposal_sampling = proposal_sampling
        self._default_dt = default_dt
        self._simulation_sampling = self._build_effective_sampling(
            proposal_sampling,
            default_dt,
        )
        self._needs_resampling = not np.isclose(
            self._proposal_sampling.interval_length,
            self._simulation_sampling.interval_length,
        )
        if self._needs_resampling:
            self._source_times = (
                np.arange(self._proposal_sampling.num_poses + 1, dtype=SIM_DTYPE)
                * self._proposal_sampling.interval_length
            )
            self._target_times = (
                np.arange(self._simulation_sampling.num_poses + 1, dtype=SIM_DTYPE)
                * self._simulation_sampling.interval_length
            )
        else:
            self._source_times = None
            self._target_times = None

        # simulation objects
        self._motion_model = BatchKinematicBicycleModel()
        self._motion_model._vehicle = get_pacifica_parameters()
        self._tracker = tracker or BatchLQRTracker(
            discretization_time=self._simulation_sampling.interval_length
        )

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
    def _wrap_angle(angles: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.arctan2(np.sin(angles), np.cos(angles))

    def _resample_trajectories(
        self,
        trajectories: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        if not self._needs_resampling:
            return trajectories.astype(SIM_DTYPE, copy=False)

        resampled = np.empty(
            (trajectories.shape[0], len(self._target_times), trajectories.shape[2]),
            dtype=SIM_DTYPE,
        )

        for state_idx in range(trajectories.shape[2]):
            values = trajectories[..., state_idx].astype(SIM_DTYPE, copy=False)
            if state_idx == StateIndex.HEADING:
                values = np.unwrap(values, axis=1)

            interpolated = interp1d(
                self._source_times,
                values,
                axis=1,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
                assume_sorted=True,
            )(self._target_times)

            if state_idx == StateIndex.HEADING:
                interpolated = self._wrap_angle(interpolated)

            resampled[..., state_idx] = interpolated

        return resampled

    def simulate(
        self, 
        trajectories: npt.NDArray[np.float32],
        ego_state: Optional[EgoState] = None,
        return_command_states: bool = False,
    ) -> Union[npt.NDArray[np.float32], Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]:
        """
        Simulate all proposals over batch-dim
        :param trajectories: proposal trajectories as array (include the initial ego state)
        :param ego_state: initial state of the ego vehicle
        :param return_command_states: whether to also return the command sequence applied at each step.
        :return: simulated proposal states as array, and optionally the applied command states.
        """

        trajectories = self._resample_trajectories(trajectories)
        states: npt.NDArray[np.float32] = trajectories_to_states(
            trajectories,
            self._simulation_sampling,
        )

        proposal_states = states
        self._tracker.update(proposal_states)

        # state array representation for simulated vehicle states
        simulated_states = np.empty(proposal_states.shape, dtype=SIM_DTYPE)
        simulated_states[:, 0] = proposal_states[:, 0]
        command_state_history = np.zeros(
            (proposal_states.shape[0], self._simulation_sampling.num_poses, len(DynamicStateIndex)),
            dtype=SIM_DTYPE,
        )

        if ego_state is not None:
            simulated_states[:, 0, StateIndex.VELOCITY_X] = ego_state.dynamic_car_state.speed
            simulated_states[:, 0, StateIndex.STEERING_ANGLE] = ego_state.tire_steering_angle
            simulated_states[:, 0, StateIndex.STEERING_RATE] = ego_state.dynamic_car_state.tire_steering_rate
            simulated_states[:, 0, StateIndex.ACCELERATION_X] = ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
            simulated_states[:, 0, StateIndex.ANGULAR_VELOCITY] = ego_state.dynamic_car_state.angular_velocity
        else:
            simulated_states[:, 0, StateIndex.VELOCITY_X] = trajectories[:, 0, 3]
            simulated_states[:, 0, StateIndex.ACCELERATION_X] = trajectories[:, 0, 4]
            simulated_states[:, 0, StateIndex.ANGULAR_VELOCITY] = trajectories[:, 0, 5]
        
        # timing objects
        current_time_point = TimePoint(int(0*1e6))
        delta_time_point = TimeDuration.from_s(self._simulation_sampling.interval_length)

        current_iteration = SimulationIteration(current_time_point, 0)
        next_iteration = SimulationIteration(current_time_point + delta_time_point, 1)

        for time_idx in range(1, self._simulation_sampling.num_poses + 1):
            sampling_time: TimePoint = (
                next_iteration.time_point - current_iteration.time_point
            )

            command_states = self._tracker.track_trajectory(
                current_iteration,
                next_iteration,
                simulated_states[:, time_idx - 1],
            )
            command_state_history[:, time_idx - 1] = command_states.astype(SIM_DTYPE, copy=False)

            simulated_states[:, time_idx] = self._motion_model.propagate_state(
                states=simulated_states[:, time_idx - 1],
                command_states=command_states,
                sampling_time=sampling_time,
            )

            current_iteration = next_iteration
            next_iteration = SimulationIteration(
                current_iteration.time_point + delta_time_point, 1 + time_idx
            )

        if return_command_states:
            return simulated_states, command_state_history

        return simulated_states
