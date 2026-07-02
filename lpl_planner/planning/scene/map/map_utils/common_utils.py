from typing import List
import torch
import numpy as np
from nuplan.common.actor_state.state_representation import StateSE2

from shapely.geometry import Polygon

def normalize_angle(angle):
    """
    Map a angle in range [-π, π]
    :param angle: any angle as float
    :return: normalized angle
    """
    return np.arctan2(np.sin(angle), np.cos(angle))

# Function to resample discrete lanes
def resample_discrete_path(path: List[StateSE2], num_points: int, return_headings=False):

    if num_points <= 0:
        raise ValueError("Number of points must be positive")
    
    path_points = np.array([state.array for state in path])
    cumulative_distances = np.cumsum(np.linalg.norm(np.diff(path_points, axis=0), axis=1))
    cumulative_distances = np.insert(cumulative_distances, 0, 0)
    distances = np.linspace(0, cumulative_distances[-1], num_points)
    # Interpolate x, y
    resampled_x = np.interp(distances, cumulative_distances, path_points[:, 0])
    resampled_y = np.interp(distances, cumulative_distances, path_points[:, 1])
    
    if return_headings:
        # Interpolate heading (normalize angles to avoid discontinuities)
        headings = np.unwrap(path_points[:, 2])  # Unwrap to avoid angle jumps
        resampled_heading = np.interp(distances, cumulative_distances, headings)
        resampled_heading = normalize_angle(resampled_heading)  # Normalize back to [-π, π]
        resampled_path = np.vstack([resampled_x, resampled_y, resampled_heading]).T
    else:
        resampled_path = np.vstack([resampled_x, resampled_y]).T

    return resampled_path

def build_ref_path_roi(ref_path: np.ndarray, s_min: float, s_max: float, width: float, extrapolate_back: bool = True):
    """
    Build a polygonal ROI centered on the reference path over [s_min, s_max].

    If s_min is negative, optionally extrapolate the reference path backward so
    the ROI also covers space behind the first path point.
    :param ref_path: [N, 2] or [N, 3] (x, y, ...)
    :param s_min: ROI start distance relative to the first path point.
    :param s_max: ROI end distance relative to the first path point.
    :param width: Total ROI width.
    :param extrapolate_back: Whether to extrapolate backward for negative s_min.
    :return: shapely Polygon
    """
    diff = np.diff(ref_path[:, :2], axis=0)
    ds = np.linalg.norm(diff, axis=1)
    s = np.concatenate([[0], np.cumsum(ds)])

    if extrapolate_back and s_min < 0:
        step = ds[0] if len(ds) > 0 else 1.0
        n_extrapolate = int(np.ceil(abs(s_min) / (step + 1e-8)))
        # Extend backward along the initial tangent.
        tangent = ref_path[1, :2] - ref_path[0, :2]
        tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
        virtual_points = [ref_path[0, :2] - tangent * (i * step) for i in range(n_extrapolate, 0, -1)]
        ref_path = np.vstack([virtual_points, ref_path])
        diff = np.diff(ref_path[:, :2], axis=0)
        ds = np.linalg.norm(diff, axis=1)
        s = np.concatenate([[0], np.cumsum(ds)])
        s_min = 0
        s_max = s_max + abs(s_min)

    idx = np.where((s >= s_min) & (s <= s_max))[0]
    if len(idx) < 2:
        raise ValueError("ref_path interval is too short")
    ref_seg = ref_path[idx, :2]

    tangents = np.diff(ref_seg, axis=0)
    tangents = np.vstack([tangents, tangents[-1]])
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / (norms + 1e-8)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    left = ref_seg + normals * (width / 2)
    right = ref_seg - normals * (width / 2)
    roi_poly = np.vstack([left, right[::-1]])

    return Polygon(roi_poly)


def torch_interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation equivalent to numpy.interp(x, xp, fp).

    xp must be monotonically increasing. Values outside the support use constant
    endpoint extrapolation.
    """
    x = x.to(dtype=fp.dtype)
    xp = xp.to(dtype=fp.dtype)
    fp = fp

    n = xp.numel()
    if n == 0:
        raise ValueError("xp is empty")
    if n == 1:
        return fp.expand_as(x)

    idx_right = torch.searchsorted(xp, x, right=False)
    idx_right = idx_right.clamp(1, n - 1)
    idx_left = idx_right - 1

    x0 = xp[idx_left]
    x1 = xp[idx_right]
    y0 = fp[idx_left]
    y1 = fp[idx_right]

    denom = (x1 - x0)
    t = torch.where(denom > 0, (x - x0) / denom, torch.zeros_like(denom))
    t = t.clamp(0.0, 1.0)

    return y0 + t * (y1 - y0)
