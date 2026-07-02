import numpy as np
import numpy.typing as npt
import shapely.creation as sc
from enum import IntEnum

from shapely import LineString, Polygon

from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.agent import Agent
from nuplan.planning.simulation.observation.idm.utils import (
    is_agent_behind,
    is_track_stopped,
)

from lpl_planner.planning.scene.trajectory_library import TrajectoryState


EVAL_UTILS_DTYPE = np.float32

class BBCoordsIndex(IntEnum):
    """Index mapping for corners and center of bounding boxes."""

    FRONT_LEFT = 0
    REAR_LEFT = 1
    REAR_RIGHT = 2
    FRONT_RIGHT = 3
    CENTER = 4

class WeightedMetricIndex(IntEnum):
    """Index mapping weighted metrics (used in Scorer)."""

    PROGRESS = 0
    TTC = 1
    COMFORTABLE = 2
    SPEED_LIMIT = 3
    LANE_CENTER_DISTANCE = 4
    HEADING_COMPLIANCE = 5

class MultiMetricIndex(IntEnum):
    """Index mapping multiplicative metrics (used in PDMScorer)."""

    NO_COLLISION = 0
    DRIVABLE_AREA = 1
    DRIVING_DIRECTION = 2
    WITHIN_LANE = 3
    RED_LIGHT_COMPLIANCE = 4
    FOLLOWING_COMPLIANCE = 5

class EgoAreaIndex(IntEnum):
    """Index mapping for area of ego agent (used in Scorer)."""

    MULTIPLE_LANES = 0
    NON_DRIVABLE_AREA = 1
    ONCOMING_TRAFFIC = 2

class CollisionType(IntEnum):
    """Enum for the types of collisions of interest."""

    STOPPED_EGO_COLLISION = 0
    STOPPED_TRACK_COLLISION = 1
    ACTIVE_FRONT_COLLISION = 2
    ACTIVE_REAR_COLLISION = 3
    ACTIVE_LATERAL_COLLISION = 4

def state_array_to_coords_array(
    states: npt.NDArray[np.float32],
    vehicle_parameters: VehicleParameters = None,
    half_length: float = 2.5880,
    half_width: float = 1.1485,
    rear_axle_to_center: float = 1.461,
) -> npt.NDArray[np.float32]:
    """
    Converts multi-dim array representation of ego states to bounding box coordinates
    :param state_array: array representation of ego states
    :param vehicle_parameters: vehicle parameter of ego
    :return: multi-dim array bounding box coordinates
    """
    n_batch, n_time, n_states = states.shape

    if vehicle_parameters is not None:
        half_length, half_width, rear_axle_to_center = (
            vehicle_parameters.half_length,
            vehicle_parameters.half_width,
            vehicle_parameters.rear_axle_to_center,
        )
    else:
        half_length = half_length
        half_width = half_width
        rear_axle_to_center = rear_axle_to_center

    headings = states[..., TrajectoryState.HEADING]
    cos, sin = np.cos(headings), np.sin(headings)

    # calculate ego center from rear axle
    rear_axle_to_center_translate = np.stack(
        [rear_axle_to_center * cos, rear_axle_to_center * sin], axis=-1
    )

    ego_centers = (
        states[..., TrajectoryState.POINT()] + rear_axle_to_center_translate
    )

    coords_array = np.zeros(
        (n_batch, n_time, len(BBCoordsIndex), 2), dtype=EVAL_UTILS_DTYPE
    )

    coords_array[:, :, BBCoordsIndex.CENTER] = ego_centers

    coords_array[:, :, BBCoordsIndex.FRONT_LEFT] = translate_lon_and_lat(
        ego_centers, headings, half_length, half_width
    )
    coords_array[:, :, BBCoordsIndex.FRONT_RIGHT] = translate_lon_and_lat(
        ego_centers, headings, half_length, -half_width
    )
    coords_array[:, :, BBCoordsIndex.REAR_LEFT] = translate_lon_and_lat(
        ego_centers, headings, -half_length, half_width
    )
    coords_array[:, :, BBCoordsIndex.REAR_RIGHT] = translate_lon_and_lat(
        ego_centers, headings, -half_length, -half_width
    )

    return coords_array

def coords_array_to_polygon_array(
    coords: npt.NDArray[np.float32],
) -> npt.NDArray[np.object_]:
    """
    Converts multi-dim array of bounding box coords of to polygons
    :param coords: bounding box coords (including corners and center)
    :return: array of shapely's polygons
    """
    # create coords copy and use center point for closed exterior
    coords_exterior = coords.copy()
    coords_exterior[..., BBCoordsIndex.CENTER, :] = coords_exterior[
        ..., BBCoordsIndex.FRONT_LEFT, :
    ]

    # load new coordinates into polygon array
    polygons = sc.polygons(coords_exterior)

    return polygons

def translate_lon_and_lat(
    centers: npt.NDArray[np.float32],
    headings: npt.NDArray[np.float32],
    lon: float,
    lat: float,
) -> npt.NDArray[np.float32]:
    """
    Translate the position component of an centers point array
    :param centers: array to be translated
    :param headings: array with heading angles
    :param lon: [m] distance by which a point should be translated in longitudinal direction
    :param lat: [m] distance by which a point should be translated in lateral direction
    :return array of translated coordinates
    """
    half_pi = np.pi / 2.0
    translation = np.stack(
        [
            (lat * np.cos(headings + half_pi)) + (lon * np.cos(headings)),
            (lat * np.sin(headings + half_pi)) + (lon * np.sin(headings)),
        ],
        axis=-1,
    ).astype(EVAL_UTILS_DTYPE, copy=False)
    return centers + translation


def get_collision_type(
    state: npt.NDArray[np.float64],
    ego_polygon: Polygon,
    tracked_object: TrackedObject = None,
    tracked_object_polygon: Polygon = None,
    stopped_speed_threshold: float = 5e-02,
    object_token: str = None,
    object_start_x = None,
    object_start_y = None,
    object_start_yaw = None,
    object_velocity = None,
) -> CollisionType:
    """
    Classify collision between ego and the track.
    :param ego_state: Ego's state at the current timestamp.
    :param tracked_object: Tracked object.
    :param stopped_speed_threshold: Threshold for 0 speed due to noise.
    :return Collision type.
    """

    ego_speed = np.hypot(
        state[TrajectoryState.VELOCITY_X],
        state[TrajectoryState.VELOCITY_Y],
    )

    is_ego_stopped = float(ego_speed) <= stopped_speed_threshold
    if tracked_object is not None:
        center_point = tracked_object_polygon.centroid
        tracked_object_center = StateSE2(
            center_point.x, center_point.y, tracked_object.box.center.heading
        )
    else:
        tracked_object_center = StateSE2(
            object_start_x,
            object_start_y,
            object_start_yaw,
        )

    ego_rear_axle_pose: StateSE2 = StateSE2(*state[TrajectoryState.STATE()])

    # Collisions at (close-to) zero ego speed
    if is_ego_stopped:
        collision_type = CollisionType.STOPPED_EGO_COLLISION

    # Collisions at (close-to) zero track speed
    elif is_track_stopped(tracked_object, agent_token=object_token, agent_speed=object_velocity):
        collision_type = CollisionType.STOPPED_TRACK_COLLISION

    # Rear collision when both ego and track are not stopped
    elif is_agent_behind(ego_rear_axle_pose, tracked_object_center):
        collision_type = CollisionType.ACTIVE_REAR_COLLISION

    # Front bumper collision when both ego and track are not stopped
    elif LineString(
        [
            ego_polygon.exterior.coords[0],
            ego_polygon.exterior.coords[3],
        ]
    ).intersects(tracked_object_polygon):
        collision_type = CollisionType.ACTIVE_FRONT_COLLISION

    # Lateral collision when both ego and track are not stopped
    else:
        collision_type = CollisionType.ACTIVE_LATERAL_COLLISION

    return collision_type

def is_track_stopped(tracked_object: TrackedObject = None, 
                     agent_speed: float = None,
                     agent_token: str = None,
                     stopped_speed_threshhold: float = 5e-02) -> bool:
    """
    Evaluates if a tracked object is stopped
    :param tracked_object: tracked_object representation
    :param stopped_speed_threshhold: Threshhold for 0 speed due to noise
    :return: True if track is stopped else False.
    """
    if tracked_object is not None:
        return (
            True
            if not isinstance(tracked_object, Agent)
            else bool(tracked_object.velocity.magnitude() <= stopped_speed_threshhold)
        )
    else:
        return (
            True
            if (not 'agent' in agent_token)
            else bool(agent_speed <= stopped_speed_threshhold)
        )








