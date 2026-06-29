from typing import List, Type

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimeDuration, TimePoint, StateSE2, StateVector2D
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class Simulator:
    """
    extracted from nuplan-devkit
    """

    def __init__(self, 
                 simulation_sampling_time : float = 0.1
                 ):
        """
        Constructor of PDMSimulator.
        :param proposal_sampling: Sampling parameters for proposals
        """

        # time parameters
        self.vehicle = get_pacifica_parameters()
        self.simulationg_sampling_time =  TimePoint(int(simulation_sampling_time * 1e6))

        # simulation objects
        self._motion_model = KinematicBicycleModel(self.vehicle)
        self.tracker = LQRTracker(
            q_longitudinal=[10.0],
            r_longitudinal=[1.0],
            q_lateral=[1.0, 10.0, 0.0],
            r_lateral=[1.0],
            discretization_time=0.1,
            tracking_horizon=10,
            jerk_penalty=1e-4,
            curvature_rate_penalty=1e-2,
            stopping_proportional_gain=0.5,
            stopping_velocity=0.2,
            )

    def simulate(
        self, trajectory: list[EgoState], initial_ego_state: EgoState
        ) -> List[EgoState]:

        trajectory.insert(0,initial_ego_state)
        current_time_point = initial_ego_state.time_point
        interpolated_trajectory = InterpolatedTrajectory(trajectory)
        current_iteration = SimulationIteration(current_time_point, 0)
        next_iteration = SimulationIteration(current_time_point + self.simulationg_sampling_time, 1)
        state = initial_ego_state
        simulated_trajectory = []

        # for _ in np.arange( self.simulationg_sampling_time.time_s,
        #                     interpolated_trajectory.end_time.time_s - current_time_point.time_s,
        #                     self.simulationg_sampling_time.time_s,
        #                     ):
        
        for _ in range(len(trajectory)-1):
            
            command_states = self.tracker.track_trajectory(current_iteration,
                                                            next_iteration,
                                                            state,
                                                            interpolated_trajectory)
            
            state = self._motion_model.propagate_state(state,
                                                        command_states,
                                                        self.simulationg_sampling_time)
            current_iteration = next_iteration
            next_iteration = SimulationIteration(current_iteration.time_point + self.simulationg_sampling_time
                                                    , current_iteration.index + 1)
            
            simulated_trajectory.append(state)



        return simulated_trajectory


