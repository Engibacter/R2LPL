from typing import Optional

import numpy as np
import numpy.typing as npt
from lpl_planner.planning.scene.scene_manager import SceneManager
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    StateSE2,
    StateVector2D,
    TimePoint,
)
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from lpl_planner.planning.planner.frenet_utils import (
    QuinticPolynomial,
)

class EmergencyBrake:
    """Class for emergency brake maneuver. Original implementation in PDM planner."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        time_to_infraction_threshold: float = 2.0,
        max_ego_speed: float = 5.0,
        max_long_accel: float = 2.40,
        min_long_accel: float = -4.00,
        max_lon_jerk: float = 4.00,
        infraction: str = "collision", # "collision" or "ttc"
        use_frenet: bool = False,
    ):
        """
        Constructor for EmergencyBrake
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param time_to_infraction_threshold: threshold for applying brake, defaults to 2.0
        :param max_ego_speed: maximum speed to apply brake, defaults to 5.0
        :param max_long_accel: maximum longitudinal acceleration for braking, defaults to 2.40
        :param min_long_accel: min longitudinal acceleration for braking, defaults to -4.00
        :param max_lon_jerk: maximum longitudinal jerk for braking, defaults to 4.00
        :param infraction: infraction to determine braking (collision or ttc), defaults to "collision"
        """

        # trajectory parameters
        self._trajectory_sampling = trajectory_sampling

        # braking parameters
        self._max_ego_speed: float = max_ego_speed  # [m/s]
        self._max_long_accel: float = max_long_accel  # [m/s^2]
        self._min_long_accel: float = min_long_accel  # [m/s^2]
        self._max_lon_jerk: float = max_lon_jerk  # [m/s^3]

        # braking condition parameters
        self._time_to_infraction_threshold: float = time_to_infraction_threshold
        self._infraction: str = infraction
        self._use_frenet: bool = use_frenet

        assert self._infraction in [
            "collision",
            "ttc",
        ], f"PDMEmergencyBraking: Infraction {self._infraction} not available as brake condition!"

    def _build_longitudinal_brake_profile(
        self,
        current_velocity: float,
        current_acceleration: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Generate a jerk-limited braking profile over the configured horizon."""
        dt = self._trajectory_sampling.interval_length
        horizon = self._trajectory_sampling.num_poses + 1

        distance_profile = np.zeros(horizon, dtype=np.float64)
        velocity_profile = np.zeros(horizon, dtype=np.float64)
        acceleration_profile = np.zeros(horizon, dtype=np.float64)

        velocity = max(float(current_velocity), 0.0)
        acceleration = float(
            np.clip(current_acceleration, self._min_long_accel, self._max_long_accel)
        )
        max_delta_acc = self._max_lon_jerk * dt

        velocity_profile[0] = velocity
        acceleration_profile[0] = acceleration

        for step_idx in range(1, horizon):
            target_acceleration = self._min_long_accel if velocity > 1e-3 else 0.0
            acceleration = float(
                np.clip(
                    acceleration + np.clip(target_acceleration - acceleration, -max_delta_acc, max_delta_acc),
                    self._min_long_accel,
                    self._max_long_accel,
                )
            )

            next_velocity = max(velocity + acceleration * dt, 0.0)
            if next_velocity == 0.0 and velocity > 0.0 and acceleration < 0.0:
                time_to_stop = min(dt, -velocity / acceleration)
                step_distance = velocity * time_to_stop + 0.5 * acceleration * time_to_stop**2
            else:
                step_distance = 0.5 * (velocity + next_velocity) * dt

            distance_profile[step_idx] = distance_profile[step_idx - 1] + max(step_distance, 0.0)
            velocity_profile[step_idx] = next_velocity
            acceleration_profile[step_idx] = acceleration
            velocity = next_velocity

        return distance_profile, velocity_profile, acceleration_profile

    def brake_if_emergency(
        self,
        ego_state: EgoState,
        scores: npt.NDArray[np.float64],
        scene_manager: SceneManager = None,
        collision_times: Optional[npt.NDArray[np.float64]] = None,
        ttc_times: Optional[npt.NDArray[np.float64]] = None,
    ) -> Optional[InterpolatedTrajectory]:
        """
        Applies emergency brake only if an infraction is expected within horizon.
        :param ego_state: state object of ego
        :param scores: array of proposal scores
        :param collision_times: per-proposal collision infraction times in seconds
        :param ttc_times: per-proposal TTC infraction times in seconds
        :return: brake trajectory or None
        """

        trajectory = None
        ego_speed: float = ego_state.dynamic_car_state.speed

        proposal_idx = np.argmax(scores)

        # retrieve time to infraction depending on brake detection mode
        if self._infraction == "ttc":
            if ttc_times is None:
                return trajectory
            time_to_infraction = float(ttc_times[proposal_idx])

        elif self._infraction == "collision":
            if collision_times is None:
                return trajectory
            time_to_infraction = float(collision_times[proposal_idx])

        # check time to infraction below threshold
        if (
            np.isfinite(time_to_infraction)
            and
            time_to_infraction <= self._time_to_infraction_threshold
            and ego_speed <= self._max_ego_speed
        ):
            trajectory = self._generate_trajectory(ego_state, scene_manager)

        return trajectory

    def _generate_trajectory(self, ego_state: EgoState, scene_manager: SceneManager=None) -> InterpolatedTrajectory:
        """
        Generates trajectory for reach zero velocity.
        :param ego_state: state object of ego
        :return: InterpolatedTrajectory for braking
        """
        current_time_point = ego_state.time_point
        current_velocity = ego_state.dynamic_car_state.center_velocity_2d.x
        current_acceleration = ego_state.dynamic_car_state.center_acceleration_2d.x
        longitudinal_s, longitudinal_v, longitudinal_a = self._build_longitudinal_brake_profile(
            current_velocity=current_velocity,
            current_acceleration=current_acceleration,
        )

        trajectory_states = []
        if self._use_frenet and scene_manager is not None:
            current_ego = np.array([0.0, 0.0, 0.0])
            current_sd = scene_manager.lane_map.cartesian_to_frenet(
                points=current_ego.reshape(1, 3)
            )[0]  # (2,)
            current_s = current_sd[0]
            current_d = current_sd[1]
            target_d = 0.0 if np.abs(current_d) < 0.1 else current_d  # bring back to centerline
            stop_indices = np.flatnonzero(longitudinal_v <= 1e-3)
            stop_time = (
                stop_indices[0] * self._trajectory_sampling.interval_length
                if stop_indices.size > 0
                else self._trajectory_sampling.num_poses * self._trajectory_sampling.interval_length
            )
            break_time = max(self._trajectory_sampling.interval_length, stop_time)
            lat_qp = QuinticPolynomial(
                current_d, 0.0, 0.0, target_d, 0.0, 0.0, break_time
            )
            s_plan = current_s + longitudinal_s
            d_plan = np.array(
                [
                    lat_qp.calc_point(min(sample * self._trajectory_sampling.interval_length, break_time))
                    for sample in range(self._trajectory_sampling.num_poses + 1)
                ],
                dtype=np.float64,
            )
    

        # Propagate planned trajectory for set number of samples
        for sample in range(self._trajectory_sampling.num_poses + 1):
            if self._use_frenet and scene_manager is not None:
                s_ = s_plan[sample]
                d_ = d_plan[sample]
                frenet_sd = np.array([s_, d_])
                pose_local = scene_manager.lane_map.frenet_to_cartesian(
                    frenet_points=frenet_sd.reshape(1, 2), with_yaw=True
                )[0]
                pose = relative_to_absolute_poses(
                    ego_state.center, [StateSE2(pose_local[0], pose_local[1], pose_local[2])]
                )[0]
            else:
                pose = relative_to_absolute_poses(
                    ego_state.center, [StateSE2(0.0, 0, 0)]
                )[0]

            ego_state_ = EgoState.build_from_center(
                center=pose,
                center_velocity_2d=StateVector2D(longitudinal_v[sample], 0),
                center_acceleration_2d=StateVector2D(longitudinal_a[sample], 0),
                tire_steering_angle=0.0,
                time_point=current_time_point,
                vehicle_parameters=ego_state.car_footprint.vehicle_parameters,
            )
            trajectory_states.append(ego_state_)

            current_time_point += TimePoint(
                int(self._trajectory_sampling.interval_length * 1e6)
            )

        return InterpolatedTrajectory(trajectory_states)