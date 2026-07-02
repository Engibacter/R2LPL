from typing import Deque
import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _se2_vel_acc_to_ego_state,
    _get_velocity_and_acceleration as _get_ml_planner_velocity_and_acceleration,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory



def trajectory_to_interpolated_trajectory(
    trajectory: npt.NDArray[np.float32],
    ego_history: Deque[EgoState],
    future_horizon: float,
    step_interval: float,
    use_anchor_velocity: bool = False,
    debug: bool = False,
):
    ego_state = ego_history[-1]
    timesteps = _get_fixed_timesteps(ego_state, future_horizon, step_interval)
    trajectory_states = [StateSE2.deserialize(pose) for pose in trajectory[..., :3]]

    if use_anchor_velocity:
        velocity_x = trajectory[..., 3]
        velocity_y = np.zeros_like(velocity_x)  # Assume zero lateral velocity.
        velocities = np.concatenate([velocity_x[:, np.newaxis], velocity_y[:, np.newaxis]], axis=-1)
        acceleration_x = trajectory[..., 5]
        acceleration_y = np.zeros_like(acceleration_x)  # Assume zero lateral acceleration.
        accelerations = np.concatenate([acceleration_x[:, np.newaxis], acceleration_y[:, np.newaxis]], axis=-1)
    else:
        velocities, accelerations = _get_ml_planner_velocity_and_acceleration(
            trajectory_states, ego_history, timesteps
        )

    ego_states = [
        _se2_vel_acc_to_ego_state(
            state,
            velocity,
            acceleration,
            timestep,
            ego_state.car_footprint.vehicle_parameters,
        )
        for state, velocity, acceleration, timestep in zip(
            trajectory_states, velocities, accelerations, timesteps
        )
    ]


    ego_states.insert(0, ego_state)

    return InterpolatedTrajectory(ego_states)

def local_trajectory_to_abstract_trajectory(
    trajectory: npt.NDArray[np.float32],
    ego_state: EgoState,
    future_horizon: float,
    step_interval: float,
) -> InterpolatedTrajectory:
    
     # convert relative trajectories to absolute trajectories
    ego_xy = ego_state.rear_axle.array
    ego_yaw = ego_state.rear_axle.heading
    rot_mat = np.array([[np.cos(ego_yaw), -np.sin(ego_yaw)],
                        [np.sin(ego_yaw),  np.cos(ego_yaw)]], dtype=float)
    trajectory_global = np.zeros_like(trajectory,dtype=np.float64)
    trajectory_global[..., :2] = trajectory[..., :2] @ rot_mat.T + ego_xy
    trajectory_global[..., 2] = trajectory[..., 2] + ego_yaw
    trajectory_global[..., 2] = np.arctan2(np.sin(trajectory_global[..., 2]), np.cos(trajectory_global[..., 2]))

    timesteps = _get_fixed_timesteps(ego_state, future_horizon, step_interval)

    trajectory_states = [StateSE2.deserialize(pose) for pose in trajectory_global[..., :3]]
    velocities = np.zeros((len(trajectory_states), 2), dtype=np.float64)
    accelerations = np.zeros((len(trajectory_states), 2), dtype=np.float64)

    ego_states = [
        _se2_vel_acc_to_ego_state(
            state,
            velocity,
            acceleration,
            timestep,
            ego_state.car_footprint.vehicle_parameters,
        )
        for state, velocity, acceleration, timestep in zip(
            trajectory_states, velocities, accelerations, timesteps
        )
    ]

    ego_states.insert(0, ego_state)

    return InterpolatedTrajectory(ego_states)




def hausdorff_xy(trajs: np.ndarray, fall_back_traj: np.ndarray) -> np.ndarray:
    """
    Calculate Hausdorff distance in XY plane between multiple trajectories and a fallback trajectory.
    Args:
        trajs: [N, T, 2] array of N trajectories.
        fall_back_traj: [T, 2] array of fallback trajectory.
    Returns:
        hausdorff_distances: [N] array of Hausdorff distances.
    """
    K, T, _ = trajs.shape
    # Broadcast (K, 1, T, 2) with (1, T, 2).
    a = trajs[:, None, :, :]          # (K,1,T,2)
    b = fall_back_traj[None, :, :]    # (1,T,2)
    # Pairwise differences: (K, T, T, 2).
    diff = a - b[:, None, :, :]       # (K,T,T,2)
    D = np.sqrt((diff ** 2).sum(axis=-1))  # (K,T,T)
    # directed distances
    d_ab = D.min(axis=2).max(axis=1)  # (K,)
    d_ba = D.min(axis=1).max(axis=1)  # (K,)
    return np.maximum(d_ab, d_ba)     # (K,)


