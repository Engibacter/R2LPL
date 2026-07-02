
import numpy.typing as npt
import numpy as np
from enum import IntEnum

from scipy.signal import savgol_filter


class BBCoordsIndex(IntEnum):
    """Index mapping for corners and center of bounding boxes."""

    FRONT_LEFT = 0
    REAR_LEFT = 1
    REAR_RIGHT = 2
    FRONT_RIGHT = 3
    CENTER = 4

def translate_lon_and_lat(
    centers: npt.NDArray,
    headings: npt.NDArray,
    lon: float,
    lat: float,
) -> npt.NDArray:
    """
    Translate the position component of an centers point array
    :param centers: array to be translated
    :param headings: array with heading angles
    :param lon: [m] distance by which a point should be translated in longitudinal direction
    :param lat: [m] distance by which a point should be translated in lateral direction
    :return array of translated coordinates
    """
    half_pi = np.pi / 2.0
    headings = headings #+ half_pi
    translation: npt.NDArray = np.stack(
        [
            (lon * np.cos(headings)) - (lat * np.sin(headings)),
            (lon * np.sin(headings)) + (lat * np.cos(headings)),
        ],
        axis=-1,
    )
    return centers + translation

def state_array_to_box_array(
    states: npt.NDArray, 
    half_length = 1.5 , 
    half_width = 1.1, 
) -> npt.NDArray:
    """
    TODO: complete description info
    Converts multi-dim tensor representation of ego states to bounding box coordinates
    :param state: tensor representation of ego states
    :param 
    :return: multi-dim tensor bounding box coordinates
    """
    n_horizon, n_states = states.shape


    half_length, half_width = (
        half_length,
        half_width,
    )

    headings = states[..., 2]
    cos, sin = np.cos(headings), np.sin(headings)

    agent_centers: npt.NDArray = (
        states[..., :2] 
    )

    coords_array: npt.NDArray = np.zeros(
        (n_horizon, len(BBCoordsIndex), 2)
    )

    coords_array[..., BBCoordsIndex.CENTER,:] = agent_centers

    coords_array[..., BBCoordsIndex.FRONT_LEFT,:] = translate_lon_and_lat(
        agent_centers, headings, half_length-half_width/2, half_width/2
    )
    coords_array[..., BBCoordsIndex.FRONT_RIGHT,:] = translate_lon_and_lat(
        agent_centers, headings, half_length-half_width/2, -half_width/2
    )
    coords_array[..., BBCoordsIndex.REAR_LEFT,:] = translate_lon_and_lat(
        agent_centers, headings, -half_length+half_width/2, half_width/2
    )
    coords_array[..., BBCoordsIndex.REAR_RIGHT,:] = translate_lon_and_lat(
        agent_centers, headings, -half_length+half_width/2, -half_width/2
    )

    return coords_array

def _smooth_1d(arr: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Safely smooth along time axis (last dim). If too short, return original."""
    T = arr.shape[-1]
    if T < 3 or window <= 1:
        return arr
    # The window must be odd and no longer than T.
    w = min(window if window % 2 == 1 else window - 1, T if T % 2 == 1 else T - 1)
    if w < 3:
        return arr
    p = min(poly, w - 1)
    return savgol_filter(arr, window_length=w, polyorder=p, axis=-1, mode="interp")

def process_agent_states(agent_states: np.ndarray,
                         dt: float = 0.1,
                         vel_window: int = 5,
                         yawrate_window: int = 5,
                         max_yaw_rate: float = 2.0) -> np.ndarray:
    """
    process agent states to compute smoothed velocity, acceleration, and yaw rate
    :param agent_states: (N,T,5+) array of agent states with at least 5 channels (x,y,yaw,vx,vy)
    :param dt: time step between consecutive states
    :param vel_window: window size for velocity smoothing
    :param yawrate_window: window size for yaw rate smoothing
    :param max_yaw_rate: maximum yaw rate to clip to
    :return: (N,T,6) array of processed agent states with channels (x,y,yaw,smoothed_v,acceleration,smoothed_yaw_rate)
    """
    assert agent_states.shape[-1] >= 5, f"expected (N,T,5), got {agent_states.shape}"
    
    if agent_states.ndim == 2:
        T, _ = agent_states.shape
        agent_states = agent_states[np.newaxis, ...]
        N = 1
    elif agent_states.ndim == 3:
        N, T, _ = agent_states.shape
    else:
        raise ValueError(f"expected (N,T,5) or (T,5), got {agent_states.shape}")

    x = agent_states[..., 0]
    y = agent_states[..., 1]
    yaw = agent_states[..., 2]

    vx = agent_states[..., 3]
    vy = agent_states[..., 4]

    v = np.sqrt(vx * vx + vy * vy)
    v_smooth = _smooth_1d(v, window=vel_window, poly=2)

    a = np.gradient(v_smooth, dt, axis=1)

    yaw_unwrap = np.unwrap(yaw, axis=1)
    dyaw = np.diff(yaw_unwrap, axis=1)
    max_step = max_yaw_rate * dt
    dyaw = np.clip(dyaw, -max_step, max_step)
    yaw_unwrap = yaw_unwrap[:, :1] + np.cumsum(np.concatenate([np.zeros_like(dyaw[:, :1]), dyaw], axis=1), axis=1)
    yaw_rate = np.gradient(yaw_unwrap, dt, axis=1)
    yaw_rate_smooth = _smooth_1d(yaw_rate, window=yawrate_window, poly=2)

    out = np.stack([x, y, yaw, v_smooth, a, yaw_rate_smooth], axis=-1)
    # Sanitize numerical outliers.
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out
