from typing import Tuple, List

import numpy as np
import numpy.typing as npt

from enum import IntEnum
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.planning.metrics.utils.state_extractors import approximate_derivatives

CONTROL_DTYPE = np.float32

class StateIndex:
    """Index mapping for array representation of ego states."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _VELOCITY_X = 3
    _VELOCITY_Y = 4
    _ACCELERATION_X = 5
    _ACCELERATION_Y = 6
    _STEERING_ANGLE = 7
    _STEERING_RATE = 8
    _ANGULAR_VELOCITY = 9
    _ANGULAR_ACCELERATION = 10

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_")
            and not attribute.startswith("__")
            and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def VELOCITY_X(cls):
        return cls._VELOCITY_X

    @classmethod
    @property
    def VELOCITY_Y(cls):
        return cls._VELOCITY_Y

    @classmethod
    @property
    def ACCELERATION_X(cls):
        return cls._ACCELERATION_X

    @classmethod
    @property
    def ACCELERATION_Y(cls):
        return cls._ACCELERATION_Y

    @classmethod
    @property
    def STEERING_ANGLE(cls):
        return cls._STEERING_ANGLE

    @classmethod
    @property
    def STEERING_RATE(cls):
        return cls._STEERING_RATE

    @classmethod
    @property
    def ANGULAR_VELOCITY(cls):
        return cls._ANGULAR_VELOCITY

    @classmethod
    @property
    def ANGULAR_ACCELERATION(cls):
        return cls._ANGULAR_ACCELERATION

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)

    @classmethod
    @property
    def VELOCITY_2D(cls):
        # assumes velocity X, Y have subsequent indices
        return slice(cls._VELOCITY_X, cls._VELOCITY_Y + 1)

    @classmethod
    @property
    def ACCELERATION_2D(cls):
        # assumes acceleration X, Y have subsequent indices
        return slice(cls._ACCELERATION_X, cls._ACCELERATION_Y + 1)
    
class DynamicStateIndex(IntEnum):
    """Index mapping for dynamic car state (output of controller)."""

    ACCELERATION_X = 0
    STEERING_RATE = 1

# Util functions for BatchLQRTracker
# Code re-written based on nuPlan's implementation:
# https://github.com/motional/nuplan-devkit

def normalize_angle(angle):
    """
    Map a angle in range [-π, π]
    :param angle: any angle as float
    :return: normalized angle
    """
    return np.arctan2(np.sin(angle), np.cos(angle))

# Default regularization weight for initial curvature fit.  Users shouldn't really need to modify this,
# we just want it positive and small for improved conditioning of the associated least squares problem.
INITIAL_CURVATURE_PENALTY = 1e-10

# helper function to apply matrix multiplication over a batch-dim
batch_matmul = lambda a, b: np.einsum("bij, bjk -> bik", a, b)


def _generate_profile_from_initial_condition_and_derivatives(
    initial_condition: npt.NDArray[np.float32],
    derivatives: npt.NDArray[np.float32],
    discretization_time: float,
) -> npt.NDArray[np.float32]:
    """
    Returns the corresponding profile (i.e. trajectory) given an initial condition and derivatives at
    multiple timesteps by integration.
    :param initial_condition: The value of the variable at the initial timestep.
    :param derivatives: The trajectory of time derivatives of the variable at timesteps 0,..., N-1.
    :param discretization_time: [s] Time discretization used for integration.
    :return: The trajectory of the variable at timesteps 0,..., N.
    """
    assert discretization_time > 0.0, "Discretization time must be positive."
    cumsum = np.cumsum(derivatives * discretization_time, axis=-1)
    profile = initial_condition[..., None] + np.pad(
        cumsum, [(0, 0), (1, 0)], mode="constant"
    )
    return profile


def _get_xy_heading_displacements_from_poses(
    poses: npt.NDArray[np.float32],
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Returns position and heading displacements given a pose trajectory.
    :param poses: <np.ndarray: num_poses, 3> A trajectory of poses (x, y, heading).
    :return: Tuple of xy displacements with shape (num_poses-1, 2) and heading displacements with shape (num_poses-1,).
    """
    assert (
        len(poses.shape) == 3
    ), "Expect a 2D matrix representing a trajectory of poses."
    assert (
        poses.shape[1] > 1
    ), "Cannot get displacements given an empty or single element pose trajectory."
    assert poses.shape[2] == 3, "Expect pose to have three elements (x, y, heading)."

    # Compute displacements that are used to complete the kinematic state and input.
    pose_differences = np.diff(poses, axis=1)  # (b, num_poses-1, 3)
    xy_displacements = pose_differences[..., :2]
    heading_displacements = normalize_angle(pose_differences[..., 2])

    return xy_displacements, heading_displacements


def _make_banded_difference_matrix(number_rows: int) -> npt.NDArray[np.float32]:
    """
    Returns a banded difference matrix with specified number_rows.
    When applied to a vector [x_1, ..., x_N], it returns [x_2 - x_1, ..., x_N - x_{N-1}].
    :param number_rows: The row dimension of the banded difference matrix (e.g. N-1 in the example above).
    :return: A banded difference matrix with shape (number_rows, number_rows+1).
    """
    banded_matrix = np.zeros((number_rows, number_rows + 1), dtype=CONTROL_DTYPE)
    eye = np.eye(number_rows, dtype=CONTROL_DTYPE)
    banded_matrix[:, 1:] = eye
    banded_matrix[:, :-1] = -eye
    return banded_matrix


def _fit_initial_velocity_and_acceleration_profile(
    xy_displacements: npt.NDArray[np.float32],
    heading_profile: npt.NDArray[np.float32],
    discretization_time: float,
    jerk_penalty: float,
) -> Tuple[float, npt.NDArray[np.float32]]:
    """
    Estimates initial velocity (v_0) and acceleration ({a_0, ...}) using least squares with jerk penalty regularization.
    :param xy_displacements: [m] Deviations in x and y occurring between M+1 poses, a M by 2 matrix.
    :param heading_profile: [rad] Headings associated to the starting timestamp for xy_displacements, a M-length vector.
    :param discretization_time: [s] Time discretization used for integration.
    :param jerk_penalty: A regularization parameter used to penalize acceleration differences.  Should be positive.
    :return: Least squares solution for initial velocity (v_0) and acceleration profile ({a_0, ..., a_M-1})
             for M displacement values.
    """
    assert discretization_time > 0.0, "Discretization time must be positive."
    assert jerk_penalty > 0, "Should have a positive jerk_penalty."

    assert len(xy_displacements.shape) == 3, "Expect xy_displacements to be a matrix."
    assert xy_displacements.shape[2] == 2, "Expect xy_displacements to have 2 columns."

    num_displacements = xy_displacements.shape[1]  # aka M in the docstring
    assert heading_profile.shape[0] == xy_displacements.shape[0]

    batch_size = heading_profile.shape[0]
    # Core problem: minimize_x ||y-Ax||_2
    y = xy_displacements.reshape(
        batch_size, -1
    )  # Flatten to a vector, [delta x_0, delta y_0, ...]

    headings = np.array(heading_profile, dtype=CONTROL_DTYPE)
    A_column = np.zeros(y.shape, dtype=CONTROL_DTYPE)
    A_column[:, 0::2] = np.cos(headings)
    A_column[:, 1::2] = np.sin(headings)

    A = np.repeat(
        A_column[..., None] * discretization_time**2, num_displacements, axis=2
    )
    A[..., 0] = A_column * discretization_time

    upper_triangle_mask = np.triu(
        np.ones((num_displacements, num_displacements), dtype=bool), k=1
    )
    upper_triangle_mask = np.repeat(upper_triangle_mask, 2, axis=0)
    A[:, upper_triangle_mask] = 0.0

    # Regularization using jerk penalty, i.e. difference of acceleration values.
    # If there are M displacements, then we have M - 1 acceleration values.
    # That means we have M - 2 jerk values, thus we make a banded difference matrix of that size.
    banded_matrix = _make_banded_difference_matrix(num_displacements - 2)
    R = np.block(
        [np.zeros((len(banded_matrix), 1)), banded_matrix]
    ).astype(CONTROL_DTYPE, copy=False)
    R = np.repeat(R[None, ...], batch_size, axis=0)

    A_T, R_T = np.transpose(A, (0, 2, 1)), np.transpose(R, (0, 2, 1))

    # Compute regularized least squares solution.
    intermediate_solution = batch_matmul(
        np.linalg.pinv(batch_matmul(A_T, A) + jerk_penalty * batch_matmul(R_T, R)), A_T
    )
    x = np.einsum("bij, bj -> bi", intermediate_solution, y)

    # Extract profile from solution.
    initial_velocity = x[:, 0]
    acceleration_profile = x[:, 1:]

    return initial_velocity, acceleration_profile


def _fit_initial_curvature_and_curvature_rate_profile(
    heading_displacements: npt.NDArray[np.float32],
    velocity_profile: npt.NDArray[np.float32],
    discretization_time: float,
    curvature_rate_penalty: float,
    initial_curvature_penalty: float = INITIAL_CURVATURE_PENALTY,
) -> Tuple[float, npt.NDArray[np.float32]]:
    """
    Estimates initial curvature (curvature_0) and curvature rate ({curvature_rate_0, ...})
    using least squares with curvature rate regularization.
    :param heading_displacements: [rad] Angular deviations in heading occuring between timesteps.
    :param velocity_profile: [m/s] Estimated or actual velocities at the timesteps matching displacements.
    :param discretization_time: [s] Time discretization used for integration.
    :param curvature_rate_penalty: A regularization parameter used to penalize curvature_rate.  Should be positive.
    :param initial_curvature_penalty: A regularization parameter to handle zero initial speed.  Should be positive and small.
    :return: Least squares solution for initial curvature (curvature_0) and curvature rate profile
             (curvature_rate_0, ..., curvature_rate_{M-1}) for M heading displacement values.
    """
    assert discretization_time > 0.0, "Discretization time must be positive."
    assert (
        curvature_rate_penalty > 0.0
    ), "Should have a positive curvature_rate_penalty."
    assert (
        initial_curvature_penalty > 0.0
    ), "Should have a positive initial_curvature_penalty."

    # Core problem: minimize_x ||y-Ax||_2
    y = heading_displacements
    batch_dim, dim = y.shape

    A = np.repeat(
        np.tri(dim, dtype=CONTROL_DTYPE)[None, ...], batch_dim, axis=0
    )  # lower triangular matrix

    A[:, :, 0] = velocity_profile * discretization_time

    velocity = velocity_profile * discretization_time**2
    A[:, 1:, 1:] *= velocity[:, None, 1:].transpose(0, 2, 1)

    # Regularization on curvature rate.  We add a small but nonzero weight on initial curvature too.
    # This is since the corresponding row of the A matrix might be zero if initial speed is 0, leading to singularity.
    # We guarantee that Q is positive definite such that the minimizer of the least squares problem is unique.
    Q = curvature_rate_penalty * np.eye(dim, dtype=CONTROL_DTYPE)
    Q[0, 0] = initial_curvature_penalty

    # Compute regularized least squares solution.
    A_T = A.transpose(0, 2, 1)

    intermediate = batch_matmul(np.linalg.pinv(batch_matmul(A_T, A) + Q), A_T)
    x = np.einsum("bij,bj->bi", intermediate, y)

    # Extract profile from solution.
    initial_curvature = x[:, 0]
    curvature_rate_profile = x[:, 1:]

    return initial_curvature, curvature_rate_profile


def get_velocity_curvature_profiles_with_derivatives_from_poses(
    discretization_time: float,
    poses: npt.NDArray[np.float32],
    jerk_penalty: float,
    curvature_rate_penalty: float,
) -> Tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """
    Main function for joint estimation of velocity, acceleration, curvature, and curvature rate given N poses
    sampled at discretization_time.  This is done by solving two least squares problems with the given penalty weights.
    :param discretization_time: [s] Time discretization used for integration.
    :param poses: <np.ndarray: num_poses, 3> A trajectory of N poses (x, y, heading).
    :param jerk_penalty: A regularization parameter used to penalize acceleration differences.  Should be positive.
    :param curvature_rate_penalty: A regularization parameter used to penalize curvature_rate.  Should be positive.
    :return: Profiles for velocity (N-1), acceleration (N-2), curvature (N-1), and curvature rate (N-2).
    """
    xy_displacements, heading_displacements = _get_xy_heading_displacements_from_poses(
        poses
    )

    (
        initial_velocity,
        acceleration_profile,
    ) = _fit_initial_velocity_and_acceleration_profile(
        xy_displacements=xy_displacements,
        heading_profile=poses[:, :-1, 2],
        discretization_time=discretization_time,
        jerk_penalty=jerk_penalty,
    )

    velocity_profile = _generate_profile_from_initial_condition_and_derivatives(
        initial_condition=initial_velocity,
        derivatives=acceleration_profile,
        discretization_time=discretization_time,
    )

    # Compute initial curvature + curvature rate least squares solution and extract results.  It relies on velocity fit.
    (
        initial_curvature,
        curvature_rate_profile,
    ) = _fit_initial_curvature_and_curvature_rate_profile(
        heading_displacements=heading_displacements,
        velocity_profile=velocity_profile,
        discretization_time=discretization_time,
        curvature_rate_penalty=curvature_rate_penalty,
    )

    curvature_profile = _generate_profile_from_initial_condition_and_derivatives(
        initial_condition=initial_curvature,
        derivatives=curvature_rate_profile,
        discretization_time=discretization_time,
    )

    return (
        velocity_profile,
        acceleration_profile,
        curvature_profile,
        curvature_rate_profile,
    )

def trajectories_to_states(trajectories: npt.NDArray[np.float32],
                           future_sampling: TrajectorySampling) -> npt.NDArray[np.float32]:
    """_summary_

    Args:
        trajectories (npt.NDArray[np.float64]): [K, T, S] K paths, T steps, S state size

    Returns:
        npt.NDArray[np.float64]:[K, T, StateIndex.size()] K paths, T steps, full state size
    """
    assert len(trajectories.shape) == 3, "Expect trajectories to be a 3D array."
    assert trajectories.shape[1] == int(future_sampling.num_poses) + 1, "Mismatch in number of timesteps, expected {} but got {}.".format(int(future_sampling.num_poses) + 1, trajectories.shape[1])

    current_state = np.zeros((trajectories.shape[0], StateIndex.size()), dtype=np.float32)
    current_state[:, StateIndex.VELOCITY_X] = trajectories[:, 0, 3]
    current_state[:, StateIndex.ACCELERATION_X] = trajectories[:, 0, 4]
    
    timesteps = np.arange(0.0, future_sampling.time_horizon, future_sampling.interval_length) + future_sampling.interval_length
    velocities, accelerations = _get_velocity_and_acceleration(
            trajectories, current_state, timesteps
        )
    
    
    states = np.zeros((trajectories.shape[0], trajectories.shape[1], StateIndex.size()), dtype=np.float32)
    states[..., :3] = trajectories[..., :3]
    states[..., StateIndex.VELOCITY_2D] = velocities
    states[..., StateIndex.ACCELERATION_2D] = accelerations
    states[:, 0, StateIndex.ANGULAR_VELOCITY] = trajectories[:, 0, 5]

    return states

def _project_from_global_to_ego_centric_ds(
    ego_poses: npt.NDArray[np.float32], values: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Project value from the global xy frame to the ego centric ds frame.

    :param ego_poses: [x, y, heading] with size [planned steps, 3].
    :param values: values in global frame with size [planned steps, 2]
    :return: values projected onto the new frame with size [planned steps, 2]
    """
    headings = ego_poses[..., -1:]

    values_lon = values[..., :1] * np.cos(headings) + values[..., 1:2] * np.sin(headings)
    values_lat = values[..., :1] * np.sin(headings) - values[..., 1:2] * np.cos(headings)
    values = np.concatenate((values_lon, values_lat), axis=-1)
    return values


def _get_velocity_and_acceleration(
    trajectory_states: npt.NDArray[np.float32], 
    current_state: npt.NDArray[np.float32], 
    timesteps: npt.NDArray[np.float32],
    interp_dt: float = 0.1
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Batch Savitzky-Golay smoothing + derivatives.
    Inputs:
      - trajectory_states: [K, T+1, S]
      - current_state:     [K, S]
      - timesteps:         [T]
    Outputs:
      - velocities_ds:     [K, T, 2]  (v_lon, v_lat) at steps 1..T projected到车体系
      - accelerations_ds:  [K, T, 2]  (a_lon, a_lat)
    """
    assert trajectory_states.ndim == 3, "trajectory_states must be [K, T+1, S]"
    K, Tp1, S = trajectory_states.shape
    T = len(timesteps)
    assert Tp1 == T + 1, f"Mismatch: states Tp1={Tp1}, timesteps T={T}"
    assert current_state.shape[0] == K, "batch mismatch"
    assert T >= 1, "Need at least one planned step"
    
    # Planned
    dt = float(timesteps[1] - timesteps[0]) if T > 1 else float(timesteps[0])
    pos = trajectory_states[:, :, StateIndex.STATE_SE2].astype(CONTROL_DTYPE)   # [K, T+1, 3]
    timesteps_current_planned: npt.NDArray[np.float32] = np.array([0.0] + timesteps.tolist())
    
    # Interpolate to denser timesteps for better smoothing
    interp_timesteps = np.arange(
        timesteps_current_planned[0],
        timesteps_current_planned[-1] + interp_dt,
        interp_dt,
    )
    
    pos_interpolate = interp1d(
        timesteps_current_planned, pos, axis=1, fill_value='extrapolate'
    )
    pos_interp = pos_interpolate(interp_timesteps)  # [K, T_interp, 3]
    Tp1_interp = pos_interp.shape[1]

    default_win = 5
    polyorder = 2
    win = min(default_win, Tp1_interp)
    if win % 2 == 0:
        win = max(1, win - 1)

    # Take derivatives
    vel_global_interp = savgol_filter(pos_interp[..., :2], window_length=win, polyorder=polyorder,
                                deriv=1, delta=interp_dt, axis=1, mode='interp')  # [K, T+1, 2]
    acc_global_interp = savgol_filter(pos_interp[..., :2], window_length=win, polyorder=polyorder,
                                deriv=2, delta=interp_dt, axis=1, mode='interp')  # [K, T+1, 2]
    
    # Set current velocity/acceleration to be consistent with current state
    vel_global_interp[:, 0, 0] = current_state[:, StateIndex.VELOCITY_X]
    acc_global_interp[:, 0, 0] = current_state[:, StateIndex.ACCELERATION_X]

    # Projection to ego centric ds frame
    ego_velocity_planned_ds = _project_from_global_to_ego_centric_ds(
        pos_interp, vel_global_interp
    )
    ego_acceleration_planned_ds = _project_from_global_to_ego_centric_ds(
        pos_interp, acc_global_interp
    )

    # Interpolate back
    ego_velocity_interp_back = interp1d(
        interp_timesteps, 
        ego_velocity_planned_ds, 
        axis=1, 
        fill_value='extrapolate'
    )
    ego_acceleration_interp_back = interp1d(
        interp_timesteps,
        ego_acceleration_planned_ds,
        axis=1,
        fill_value='extrapolate',
    )

    ego_velocity_planned_ds = ego_velocity_interp_back(timesteps_current_planned)
    ego_acceleration_planned_ds = ego_acceleration_interp_back(timesteps_current_planned)

    return ego_velocity_planned_ds, ego_acceleration_planned_ds