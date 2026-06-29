from dataclasses import dataclass, field
from typing import List
import numpy as np
import numpy.typing as npt
from enum import Enum


def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to [-pi, pi].
    """
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def interp_valid_yaw(yaw: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Interpolate invalid yaw values (NaN/inf). If all are invalid, return zeros.
    """
    if np.any(np.isnan(yaw)) or np.any(np.isinf(yaw)):
        valid = ~np.isnan(yaw) & ~np.isinf(yaw)
        idx = np.arange(len(yaw))
        yaw = np.interp(idx, idx[valid], yaw[valid]) if np.any(valid) else np.zeros_like(yaw)
    return yaw


class QuinticPolynomial:
    """
    Quintic polynomial with boundary conditions on position/velocity/acceleration
    at start and end over horizon T.
    """

    def __init__(self, xs: float, vxs: float, axs: float,
                 xe: float, vxe: float, axe: float, T: float) -> None:
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([
            [T**3, T**4, T**5],
            [3 * T**2, 4 * T**3, 5 * T**4],
            [6 * T, 12 * T**2, 20 * T**3],
        ])
        b = np.array([
            xe - self.a0 - self.a1 * T - self.a2 * T**2,
            vxe - self.a1 - 2 * self.a2 * T,
            axe - 2 * self.a2,
        ])
        self.a3, self.a4, self.a5 = np.linalg.solve(A, b)

    def calc_point(self, t: float) -> float:
        return (self.a0 + self.a1 * t + self.a2 * t**2 +
                self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5)

    def calc_first_derivative(self, t: float) -> float:
        return (self.a1 + 2 * self.a2 * t +
                3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4)

    def calc_second_derivative(self, t: float) -> float:
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

    def calc_third_derivative(self, t: float) -> float:
        return 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2


class QuarticPolynomial:
    """
    Quartic polynomial with boundary conditions on position/velocity/acceleration
    at start, and velocity/acceleration at end over horizon T.
    """

    def __init__(self, xs: float, vxs: float, axs: float,
                 vxe: float, axe: float, T: float) -> None:
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([
            [3 * T**2, 4 * T**3],
            [6 * T, 12 * T**2],
        ])
        b = np.array([
            vxe - self.a1 - 2 * self.a2 * T,
            axe - 2 * self.a2,
        ])
        self.a3, self.a4 = np.linalg.solve(A, b)

    def calc_point(self, t: float) -> float:
        return self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4

    def calc_first_derivative(self, t: float) -> float:
        return self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3

    def calc_second_derivative(self, t: float) -> float:
        return 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

    def calc_third_derivative(self, t: float) -> float:
        return 6 * self.a3 + 24 * self.a4 * t

class Intent(Enum):
    KEEP = 0
    LC_LEFT = 1
    LC_RIGHT = 2
    BRAKE_KEEP = 3

@dataclass
class FrenetTraj:
    """
    Container for a single Frenet trajectory and its Cartesian projection.
    """
    t: List[float] = field(default_factory=list)
    d: List[float] = field(default_factory=list)
    d_dot: List[float] = field(default_factory=list)
    d_dotdot: List[float] = field(default_factory=list)
    d_dotdotdot: List[float] = field(default_factory=list)
    s: List[float] = field(default_factory=list)
    s_dot: List[float] = field(default_factory=list)
    s_dotdot: List[float] = field(default_factory=list)
    s_dotdotdot: List[float] = field(default_factory=list)

    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    yaw: List[float] = field(default_factory=list)
    ds: List[float] = field(default_factory=list)
    c: List[float] = field(default_factory=list)
    intent: Intent = Intent.KEEP