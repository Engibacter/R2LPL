from typing import Optional
import numpy as np
import numpy.typing as npt

from scipy.signal import savgol_filter

from lpl_planner.planning.scene.evaluate.utils.control_utils import StateIndex
from lpl_planner.planning.scene.trajectory_library import TrajectoryState



MAX_ABS_MAG_JERK = 8.37  # [m/s^3]
MAX_ABS_LAT_ACCEL = 4.89  # [m/s^2]
MAX_LON_ACCEL = 2.30  # [m/s^2]
MIN_LON_ACCEL = -4.00 # [m/s^2]
MAX_ABS_YAW_ACCEL = 1.93  # [rad/s^2]
MAX_ABS_LON_JERK = 4.13  # [m/s^3]
MAX_ABS_YAW_RATE = 0.95  # [rad/s]
MAX_ABS_MAG_JERK = 8.37  # [m/s^3]

REAR_AXLE_TO_CENTER = 1.461

def _extract_ego_acceleration(
    states: npt.NDArray[np.float64],
    acceleration_coordinate: str,
    decimals: int = 8,
    poly_order: int = 2,
    window_length: int = 8,
) -> npt.NDArray[np.float32]:
    """
    Extract acceleration of ego pose in simulation history over batch-dim
    :param states: array representation of ego state values
    :param acceleration_coordinate: string of axis to extract
    :param decimals: decimal precision, defaults to 8
    :param poly_order: polynomial order, defaults to 2
    :param window_length: window size for extraction, defaults to 8
    :raises ValueError: when coordinate not available
    :return: array containing acceleration values
    """

    n_batch, n_time, n_states = states.shape
    if acceleration_coordinate == "x":
        acceleration: npt.NDArray[np.float64] = states[..., StateIndex.ACCELERATION_X]
        acceleration = acceleration + REAR_AXLE_TO_CENTER * states[..., StateIndex.ANGULAR_VELOCITY] ** 2
        acceleration = acceleration + REAR_AXLE_TO_CENTER * states[..., StateIndex.ANGULAR_ACCELERATION]

    elif acceleration_coordinate == "y":
        acceleration: npt.NDArray[np.float64] = states[..., StateIndex.ACCELERATION_Y]
        acceleration = acceleration + 0 * states[..., StateIndex.ANGULAR_VELOCITY] ** 2
        acceleration = acceleration + 0 * states[..., StateIndex.ANGULAR_ACCELERATION]

    elif acceleration_coordinate == "magnitude":
        acceleration: npt.NDArray[np.float64] = np.hypot(
            states[..., StateIndex.ACCELERATION_X],
            states[..., StateIndex.ACCELERATION_Y],
        )
    else:
        raise ValueError(
            f"acceleration_coordinate option: {acceleration_coordinate} not available. "
            f"Available options are: x, y or magnitude"
        )

    acceleration = savgol_filter(
        acceleration,
        polyorder=poly_order,
        window_length=min(window_length, n_time),
        axis=-1,
    )
    acceleration = np.round(acceleration, decimals=decimals)
    return acceleration


def _extract_ego_jerk(
    states: npt.NDArray[np.float64],
    acceleration_coordinate: str,
    time_steps_s: npt.NDArray[np.float64],
    decimals: int = 8,
    deriv_order: int = 1,
    poly_order: int = 2,
    window_length: int = 15,
) -> npt.NDArray[np.float32]:
    """
    Extract jerk of ego pose in simulation history over batch-dim
    :param states: array representation of ego state values
    :param acceleration_coordinate: string of axis to extract
    :param time_steps_s: time steps [s] of time dim
    :param decimals: decimal precision, defaults to 8
    :param deriv_order: order of derivative, defaults to 1
    :param poly_order: polynomial order, defaults to 2
    :param window_length: window size for extraction, defaults to 15
    :return: array containing jerk values
    """
    n_batch, n_time, n_states = states.shape
    ego_acceleration = _extract_ego_acceleration(
        states, acceleration_coordinate=acceleration_coordinate
    )
    jerk = _approximate_derivatives(
        ego_acceleration,
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
        window_length=min(window_length, n_time),
    )
    jerk = np.round(jerk, decimals=decimals)
    return jerk


def _extract_ego_yaw_rate(
    states: npt.NDArray[np.float64],
    time_steps_s: npt.NDArray[np.float64],
    deriv_order: int = 1,
    poly_order: int = 2,
    decimals: int = 8,
    window_length: int = 15,
) -> npt.NDArray[np.float32]:
    """
    Extract yaw-rate of simulation history over batch-dim
    :param states: array representation of ego state values
    :param time_steps_s: time steps [s] of time dim
    :param deriv_order: order of derivative, defaults to 1
    :param poly_order: polynomial order, defaults to 2
    :param decimals:  decimal precision, defaults to 8
    :param window_length: window size for extraction, defaults to 15
    :return: array containing ego's yaw rate
    """
    ego_headings = states[..., StateIndex.HEADING]
    ego_yaw_rate = _approximate_derivatives(
        _phase_unwrap(ego_headings),
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
    )  # convert to seconds
    ego_yaw_rate = np.round(ego_yaw_rate, decimals=decimals)
    return ego_yaw_rate


def _phase_unwrap(headings: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Returns an array of heading angles equal mod 2 pi to the input heading angles,
    and such that the difference between successive output angles is less than or
    equal to pi radians in absolute value
    :param headings: An array of headings (radians)
    :return The phase-unwrapped equivalent headings.
    """
    # There are some jumps in the heading (e.g. from -np.pi to +np.pi) which causes approximation of yaw to be very large.
    # We want unwrapped[j] = headings[j] - 2*pi*adjustments[j] for some integer-valued adjustments making the absolute value of
    # unwrapped[j+1] - unwrapped[j] at most pi:
    # -pi <= headings[j+1] - headings[j] - 2*pi*(adjustments[j+1] - adjustments[j]) <= pi
    # -1/2 <= (headings[j+1] - headings[j])/(2*pi) - (adjustments[j+1] - adjustments[j]) <= 1/2
    # So adjustments[j+1] - adjustments[j] = round((headings[j+1] - headings[j]) / (2*pi)).
    two_pi = 2.0 * np.pi
    adjustments = np.zeros_like(headings)
    adjustments[..., 1:] = np.cumsum(
        np.round(np.diff(headings, axis=-1) / two_pi), axis=-1
    )
    unwrapped = headings - two_pi * adjustments
    return unwrapped


def _approximate_derivatives(
    y: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
    window_length: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
    axis: int = -1,
) -> npt.NDArray[np.float32]:
    """
    Given two equal-length sequences y and x, compute an approximation to the n-th
    derivative of some function interpolating the (x, y) data points, and return its
    values at the x's.  We assume the x's are increasing and equally-spaced.
    :param y: The dependent variable (say of length n)
    :param x: The independent variable (must have the same length n).  Must be strictly
        increasing and equally-spaced.
    :param window_length: The order (default 5) of the Savitsky-Golay filter used.
        (Ignored if the x's are not equally-spaced.)  Must be odd and at least 3
    :param poly_order: The degree (default 2) of the filter polynomial used.  Must
        be less than the window_length
    :param deriv_order: The order of derivative to compute (default 1)
    :param axis: The axis of the array x along which the filter is to be applied. Default is -1.
    :return Derivatives.
    """
    window_length = min(window_length, len(x))

    if not (poly_order < window_length):
        raise ValueError(f"{poly_order} < {window_length} does not hold!")

    dx = np.diff(x, axis=-1)
    if not (dx > 0).all():
        raise RuntimeError("dx is not monotonically increasing!")

    dx = dx.mean()
    derivative: npt.NDArray[np.float32] = savgol_filter(
        y,
        polyorder=poly_order,
        window_length=window_length,
        deriv=deriv_order,
        delta=dx,
        axis=axis,
    )
    return derivative

def mean_ratio_within_bound(
    metric: npt.NDArray[np.float64],
    min_bound: Optional[float] = None,
    max_bound: Optional[float] = None,
) -> npt.NDArray[np.bool_]:
    """
    Calculate the ratio if within bound in batch-dim are within bounds.
    :param metric: metric values
    :param min_bound: minimum bound, defaults to None
    :param max_bound: maximum bound, defaults to None
    :return: NDarray of mean ratio of metrics [0 1), if exceed bound then 1
    """
    min_bound = min_bound if min_bound else float(-np.inf)
    max_bound = max_bound if max_bound else float(np.inf)
    metric_values = np.array(metric)
    metric_ratio = np.where(metric_values>0, metric_values/max_bound,metric_values/min_bound)
    metric_ratio_within_bound = np.all(metric_ratio<1,axis=-1)
    metric_mean_ratio = np.mean(metric_ratio,axis=-1)
    return np.where(metric_ratio_within_bound,metric_mean_ratio,1)

def score_metric_within_bounds(
    metric: npt.NDArray[np.float64],
    min_bound: Optional[float] = None,
    max_bound: Optional[float] = None,
) -> npt.NDArray[np.bool_]:
    """
    Calculate the ratio if within bound in batch-dim are within bounds.
    :param metric: metric values
    :param min_bound: minimum bound, defaults to None
    :param max_bound: maximum bound, defaults to None
    :return: NDarray of score of metrics (0, 1), the higher the better
    """
    min_bound = min_bound if min_bound else float(-np.inf)
    max_bound = max_bound if max_bound else float(np.inf)
    metric_values = np.array(metric) # [B,T]
    metric_ratio = np.where(metric_values>0, metric_values/max_bound,metric_values/min_bound) # [B,T]
    metric_score = np.where(metric_ratio<1, 1 - metric_ratio, 0) # [B,T]
    return metric_score #[B,T]


def mean_ratio_of_trajectories(trajectories: npt.NDArray[np.float64],
                               time_steps_s: npt.NDArray[np.float64]
                               ) -> npt.NDArray[np.float64]:
    """
    Calculate the mean ratio of trajectories in batch-dim.
    :param trajectories: trajectory values
    :return: NDarray of mean ratio of trajectories [0 1)
    """
    n_batch, n_time, n_states = trajectories.shape

    lon_acceleration = _extract_ego_acceleration(
        trajectories, acceleration_coordinate="x", window_length=n_time
    )
    lon_acc_score = mean_ratio_within_bound(
        lon_acceleration,
        min_bound=MIN_LON_ACCEL,
        max_bound=MAX_LON_ACCEL,
    )
    
    lat_acceleration = _extract_ego_acceleration(
        trajectories, acceleration_coordinate="y", window_length=n_time
    )
    lat_acc_score = mean_ratio_within_bound(
        lat_acceleration,
        min_bound=-MAX_ABS_LAT_ACCEL,
        max_bound=MAX_ABS_LAT_ACCEL,
    )

    lon_jerk = _extract_ego_jerk(
        trajectories,
        acceleration_coordinate="x",
        time_steps_s=time_steps_s,
        window_length=n_time,
    )
    lon_jerk_score = mean_ratio_within_bound(
        lon_jerk,
        min_bound=-MAX_ABS_LON_JERK,
        max_bound=MAX_ABS_LON_JERK,
    )

    jerk_mag = _extract_ego_jerk(
        trajectories,
        acceleration_coordinate="magnitude",
        time_steps_s=time_steps_s,
        window_length=n_time,
    )
    mag_jerk_score = mean_ratio_within_bound(
        jerk_mag,
        min_bound=-MAX_ABS_MAG_JERK,
        max_bound=MAX_ABS_MAG_JERK,
    )

    yaw_rate = _extract_ego_yaw_rate(trajectories, time_steps_s, window_length=n_time)
    yaw_rate_score = mean_ratio_within_bound(
        yaw_rate,
        min_bound=-MAX_ABS_YAW_RATE,
        max_bound=MAX_ABS_YAW_RATE,
    )

    yaw_accel = _extract_ego_yaw_rate(
        trajectories, time_steps_s, deriv_order=2, poly_order=3, window_length=n_time
    )
    yaw_acc_score = mean_ratio_within_bound(
        yaw_accel,
        min_bound=-MAX_ABS_YAW_ACCEL,
        max_bound=MAX_ABS_YAW_ACCEL,
    )
    comfortable_value = np.stack(
        [
            lon_acc_score,
            lat_acc_score,
            lon_jerk_score,
            mag_jerk_score,
            yaw_rate_score,
            yaw_acc_score,
        ],
        axis=-1,
    ) #[B,6]
    return comfortable_value

def timed_score_of_trajectories(trajectories: npt.NDArray[np.float64],
                               time_steps_s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Calculate the timed score of trajectories in batch-dim.
    :param trajectories: trajectory values [B,T,S]
    :return: NDarray of timed score of trajectories (-1, 1)
    """
    n_batch, n_time, n_states = trajectories.shape

    lon_acceleration = _extract_ego_acceleration(
        trajectories, acceleration_coordinate="x", window_length=n_time
    )
    lon_acc_score = score_metric_within_bounds(
        lon_acceleration,
        min_bound=MIN_LON_ACCEL,
        max_bound=MAX_LON_ACCEL,
    )
    
    lat_acceleration = _extract_ego_acceleration(
        trajectories, acceleration_coordinate="y", window_length=n_time
    )
    lat_acc_score = score_metric_within_bounds(
        lat_acceleration,
        min_bound=-MAX_ABS_LAT_ACCEL,
        max_bound=MAX_ABS_LAT_ACCEL,
    )

    lon_jerk = _extract_ego_jerk(
        trajectories,
        acceleration_coordinate="x",
        time_steps_s=time_steps_s,
        window_length=n_time,
    )
    lon_jerk_score = score_metric_within_bounds(
        lon_jerk,
        min_bound=-MAX_ABS_LON_JERK,
        max_bound=MAX_ABS_LON_JERK,
    )

    jerk_mag = _extract_ego_jerk(
        trajectories,
        acceleration_coordinate="magnitude",
        time_steps_s=time_steps_s,
        window_length=n_time,
    )
    mag_jerk_score = score_metric_within_bounds(
        jerk_mag,
        min_bound=-MAX_ABS_MAG_JERK,
        max_bound=MAX_ABS_MAG_JERK,
    )

    yaw_rate = _extract_ego_yaw_rate(trajectories, time_steps_s, window_length=n_time)
    yaw_rate_score = score_metric_within_bounds(
        yaw_rate,
        min_bound=-MAX_ABS_YAW_RATE,
        max_bound=MAX_ABS_YAW_RATE,
    )

    yaw_accel = _extract_ego_yaw_rate(
        trajectories, time_steps_s, deriv_order=2, poly_order=3, window_length=n_time
    )
    yaw_acc_score = score_metric_within_bounds(
        yaw_accel,
        min_bound=-MAX_ABS_YAW_ACCEL,
        max_bound=MAX_ABS_YAW_ACCEL,
    )
    
    comfortable_value = np.stack(
        [
            lon_acc_score,
            lat_acc_score,
            lon_jerk_score,
            mag_jerk_score,
            yaw_rate_score,
            yaw_acc_score,
        ],
        axis=-1,
    )#[B,T,6]
    return comfortable_value

def _finite_diff(x: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
    """
    Centered finite differences, with forward/backward differences at endpoints.
    x: [..., T, D]
    """
    x = np.asarray(x, dtype=np.float64)
    dx = np.zeros_like(x)
    # Interior points.
    dx[..., 1:-1, :] = (x[..., 2:, :] - x[..., :-2, :]) / (2.0 * dt)
    # Endpoints.
    dx[..., 0, :] = (x[..., 1, :] - x[..., 0, :]) / dt
    dx[..., -1, :] = (x[..., -1, :] - x[..., -2, :]) / dt
    return dx

def _finite_diff_scalar(x: npt.NDArray[np.float64], dt: float) -> npt.NDArray[np.float64]:
    """
    Centered finite differences for scalar sequences, with forward/backward endpoints.
    x: [..., T]
    """
    x = np.asarray(x, dtype=np.float64)
    dx = np.zeros_like(x)
    dx[..., 1:-1] = (x[..., 2:] - x[..., :-2]) / (2.0 * dt)
    dx[..., 0] = (x[..., 1] - x[..., 0]) / dt
    dx[..., -1] = (x[..., -1] - x[..., -2]) / dt
    return dx

def mean_ratio_of_trajectories_from_xyyaw(
    trajectories: npt.NDArray[np.float64],
    dt: float,
    min_lon_acc: float,
    max_lon_acc: float,
    max_lat_acc: float,
    max_lon_jerk: float,
    max_mag_jerk: float,
    max_yaw_rate: float,
    max_yaw_acc: float,
) -> npt.NDArray[np.float64]:
    B, T, S = trajectories.shape
    eps = 1e-6

    # Extract position and heading.
    pos = trajectories[:, :, TrajectoryState.POINT()]            # [B,T,2]
    yaw = trajectories[:, :, TrajectoryState.HEADING]            # [B,T]
    # Unwrap angles to avoid discontinuities.
    yaw_unwrapped = np.unwrap(yaw, axis=1)

    # Savitzky-Golay parameters with an odd window adapted to T.
    win = 9     # base window
    poly = 2
    # Keep the window odd and no longer than T.
    eff_win = min(win, T if (T % 2 == 1) else T - 1)
    if eff_win < 3:
        eff_win = 3
    if eff_win <= poly:
        poly = max(1, eff_win - 2)

    tvec = (np.arange(T, dtype=np.float32) * float(dt))

    if T >= 5:
        # First/second/third derivatives of position: velocity, acceleration, jerk.
        v_xy = approximate_derivatives(pos, tvec, window_length=eff_win, poly_order=poly, deriv_order=1, axis=1)  # [B,T,2]
        a_xy = approximate_derivatives(pos, tvec, window_length=eff_win, poly_order=poly, deriv_order=2, axis=1)  # [B,T,2]
        j_xy = approximate_derivatives(pos, tvec, window_length=eff_win, poly_order=poly, deriv_order=3, axis=1)  # [B,T,2]

        # First/second derivatives of heading: yaw rate, yaw acceleration.
        yaw_rate = approximate_derivatives(yaw_unwrapped, tvec, window_length=eff_win, poly_order=poly, deriv_order=1, axis=1)  # [B,T]
        yaw_acc  = approximate_derivatives(yaw_unwrapped, tvec, window_length=eff_win, poly_order=poly, deriv_order=2, axis=1)  # [B,T]
    else:
        # Fall back to finite differences for short horizons to avoid ill-conditioned filtering.
        v_xy = _finite_diff(pos, dt)
        a_xy = _finite_diff(v_xy, dt)
        j_xy = _finite_diff(a_xy, dt)
        yaw_rate = _finite_diff_scalar(yaw_unwrapped, dt)
        yaw_acc = _finite_diff_scalar(yaw_rate, dt)

    # Scalar speed.
    v = np.linalg.norm(v_xy, axis=-1)                            # [B,T]

    # Anchor the first frame with the input longitudinal velocity/acceleration.
    v0_long = trajectories[:, 0, 3]     # [B]
    t_hat = np.stack([np.cos(yaw_unwrapped[:, 0]), np.sin(yaw_unwrapped[:, 0])], axis=-1)  # [B,2]
    v_xy[:, 0, :] = (v0_long[:, None] * t_hat)
    v[:, 0] = v0_long
    # Re-estimate first-frame acceleration with a forward difference.
    if T > 1:
        a_xy[:, 0, :] = (v_xy[:, 1, :] - v_xy[:, 0, :]) / (tvec[1] - tvec[0] + eps)

    a_lon = approximate_derivatives(v, tvec, window_length=eff_win, poly_order=poly, deriv_order=1, axis=1) if T >= 5 else _finite_diff_scalar(v, dt)  # [B,T]
    a0_long = trajectories[:, 0, 4]
    a_lon[:, 0] = a0_long

    # Lateral acceleration: a_lat = v * yaw_rate (= v^2 * curvature).
    a_lat = v * yaw_rate

    # Longitudinal jerk and jerk magnitude.
    j_lon = approximate_derivatives(a_lon, tvec, window_length=eff_win, poly_order=poly, deriv_order=1, axis=1) if T >= 5 else _finite_diff_scalar(a_lon, dt)  # [B,T]
    j_mag = np.linalg.norm(j_xy, axis=-1) 

    # Scores.
    lon_acc_score = mean_ratio_within_bound(a_lon, min_bound=min_lon_acc, max_bound=max_lon_acc)
    lat_acc_score = mean_ratio_within_bound(a_lat, min_bound=-max_lat_acc, max_bound=max_lat_acc)
    lon_jerk_score = mean_ratio_within_bound(j_lon, min_bound=-max_lon_jerk, max_bound=max_lon_jerk)
    mag_jerk_score = mean_ratio_within_bound(j_mag, min_bound=-max_mag_jerk, max_bound=max_mag_jerk)
    yaw_rate_score = mean_ratio_within_bound(yaw_rate, min_bound=-max_yaw_rate, max_bound=max_yaw_rate)
    yaw_acc_score = mean_ratio_within_bound(yaw_acc, min_bound=-max_yaw_acc, max_bound=max_yaw_acc)

    comfortable_value = np.stack(
        [lon_acc_score, lat_acc_score, lon_jerk_score, mag_jerk_score, yaw_rate_score, yaw_acc_score],
        axis=-1,
    )  # [B,6]
    return comfortable_value



def timed_score_of_trajectories_from_xyyaw(trajectories: npt.NDArray[np.float64],
                                    dt: float,
                                    min_lon_acc: float,
                                    max_lon_acc: float,
                                    max_lat_acc: float,
                                    max_lon_jerk: float,
                                    max_mag_jerk: float,
                                    max_yaw_rate: float,
                                    max_yaw_acc: float) -> npt.NDArray[np.float64]:
    """
    Calculate the timed score of trajectories in batch-dim from x,y,yaw.
    Use Savitzky-Golay filtering to approximate derivatives and reduce difference noise.
    :param trajectories: trajectory values [B,T,S]
    :return: NDarray of timed score of trajectories (-1, 1) with shape [B,T,6]
    """
    B, T, S = trajectories.shape
    eps = 1e-6

    # Position and heading.
    pos = trajectories[:, :, TrajectoryState.POINT()]     # [B,T,2]
    yaw = trajectories[:, :, TrajectoryState.HEADING]     # [B,T]
    yaw_unwrapped = np.unwrap(yaw, axis=1)

    # Adaptive Savitzky-Golay parameters.
    win = 9
    poly = 2
    eff_win = min(win, T if (T % 2 == 1) else T - 1)
    if eff_win < 3:
        eff_win = 3
    if eff_win <= poly:
        poly = max(1, eff_win - 2)
    tvec = (np.arange(T, dtype=np.float32) * float(dt))

    if T >= 5:
        v_xy = approximate_derivatives(pos, tvec, window_length=eff_win, poly_order=poly, deriv_order=1, axis=1)  # [B,T,2]
        a_xy = approximate_derivatives(pos, tvec, window_length=eff_win, poly_order=poly, deriv_order=2, axis=1)  # [B,T,2]
        j_xy = approximate_derivatives(pos, tvec, window_length=eff_win, poly_order=poly, deriv_order=3, axis=1)  # [B,T,2]
        yaw_rate = approximate_derivatives(yaw_unwrapped, tvec, window_length=eff_win, poly_order=poly, deriv_order=1, axis=1)  # [B,T]
        yaw_acc  = approximate_derivatives(yaw_unwrapped, tvec, window_length=eff_win, poly_order=poly, deriv_order=2, axis=1)  # [B,T]
    else:
        v_xy = _finite_diff(pos, dt)
        a_xy = _finite_diff(v_xy, dt)
        j_xy = _finite_diff(a_xy, dt)
        yaw_rate = _finite_diff_scalar(yaw_unwrapped, dt)
        yaw_acc = _finite_diff_scalar(yaw_rate, dt)

    # Speed magnitude and unit tangent/normal in the instantaneous body frame.
    v = np.linalg.norm(v_xy, axis=-1)  # [B,T]
    t_hat = np.stack([np.cos(yaw_unwrapped), np.sin(yaw_unwrapped)], axis=-1)        # [B,T,2]
    # Optional normal vector: n_hat = [-sin(yaw), cos(yaw)].

    # Longitudinal acceleration (tangent): a_lon = dv/dt, which is more stable than direct projection.
    if T >= 5:
        a_lon = approximate_derivatives(v, tvec, window_length=eff_win, poly_order=poly, deriv_order=1, axis=1)  # [B,T]
    else:
        a_lon = _finite_diff_scalar(v, dt)

    # Lateral acceleration (normal): a_lat = v * yaw_rate.
    a_lat = v * yaw_rate  # [B,T]

    # Longitudinal jerk: j_lon = d(a_lon)/dt.
    if T >= 5:
        j_lon = approximate_derivatives(a_lon, tvec, window_length=eff_win, poly_order=poly, deriv_order=1, axis=1)  # [B,T]
    else:
        j_lon = _finite_diff_scalar(a_lon, dt)

    # Jerk magnitude: norm of the third-order position derivative.
    j_mag = np.linalg.norm(j_xy, axis=-1)  # [B,T]

    # Scores.
    lon_acc_score = score_metric_within_bounds(a_lon, min_bound=min_lon_acc, max_bound=max_lon_acc)
    lat_acc_score = score_metric_within_bounds(a_lat, min_bound=-max_lat_acc, max_bound=max_lat_acc)
    lon_jerk_score = score_metric_within_bounds(j_lon, min_bound=-max_lon_jerk, max_bound=max_lon_jerk)
    mag_jerk_score = score_metric_within_bounds(j_mag, min_bound=-max_mag_jerk, max_bound=max_mag_jerk)
    yaw_rate_score = score_metric_within_bounds(yaw_rate, min_bound=-max_yaw_rate, max_bound=max_yaw_rate)
    yaw_acc_score = score_metric_within_bounds(yaw_acc, min_bound=-max_yaw_acc, max_bound=max_yaw_acc)

    comfortable_value = np.stack(
        [lon_acc_score, lat_acc_score, lon_jerk_score, mag_jerk_score, yaw_rate_score, yaw_acc_score],
        axis=-1,
    )  # [B, T, 6]
    return comfortable_value


def rotate_poly(poly, yaw, center):
        """"""
        c, s = np.cos(yaw), np.sin(yaw)
        rot_mat = np.array([[c, -s], [s, c]])
        return (poly - center) @ rot_mat.T + center

def approximate_derivatives(
    y: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
    window_length: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
    axis: int = -1,
) -> npt.NDArray[np.float32]:
    """
    Given two equal-length sequences y and x, compute an approximation to the n-th
    derivative of some function interpolating the (x, y) data points, and return its
    values at the x's.  We assume the x's are increasing and equally-spaced.
    :param y: The dependent variable (say of length n)
    :param x: The independent variable (must have the same length n).  Must be strictly
        increasing and equally-spaced.
    :param window_length: The order (default 5) of the Savitsky-Golay filter used.
        (Ignored if the x's are not equally-spaced.)  Must be odd and at least 3
    :param poly_order: The degree (default 2) of the filter polynomial used.  Must
        be less than the window_length
    :param deriv_order: The order of derivative to compute (default 1)
    :param axis: The axis of the array x along which the filter is to be applied. Default is -1.
    :return Derivatives.
    """
    window_length = min(window_length, len(x))

    if not (poly_order < window_length):
        raise ValueError(f'{poly_order} < {window_length} does not hold!')

    dx = np.diff(x)
    if not (dx > 0).all():
        raise RuntimeError('dx is not monotonically increasing!')

    dx = dx.mean()
    derivative: npt.NDArray[np.float32] = savgol_filter(
        y, polyorder=poly_order, window_length=window_length, deriv=deriv_order, delta=dx, axis=axis
    )
    return derivative
