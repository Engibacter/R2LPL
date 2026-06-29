from typing import Dict, List, Tuple

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import (
    Intersection,
    Lane,
    LaneConnector,
    PolygonMapObject,
    LaneGraphEdgeMapObject,
    RoadBlockGraphEdgeMapObject,
    StopLine,
    PolylineMapObject,
)
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import (
    STRTreeOccupancyMapFactory,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.metrics.utils.route_extractor import get_route
from .bfs_roadblock import (
    BreadthFirstSearchRoadBlock,
)
from .common_utils import normalize_angle


def _search_route_roadblock_path(
    start_roadblock_id: str,
    target_roadblock_id: str,
    map_api: AbstractMap,
    max_depth: int,
) -> Tuple[List[RoadBlockGraphEdgeMapObject], List[str], bool]:
    """Search a forward-connected roadblock path between two roadblocks."""
    if start_roadblock_id == target_roadblock_id:
        roadblock = map_api._get_roadblock(start_roadblock_id)
        roadblock = roadblock or map_api._get_roadblock_connector(start_roadblock_id)
        if roadblock is None:
            return [], [], False
        return [roadblock], [start_roadblock_id], True

    graph_search = BreadthFirstSearchRoadBlock(
        start_roadblock_id, map_api, forward_search=True
    )
    (path, path_id), path_found = graph_search.search(target_roadblock_id, max_depth)
    return path, path_id, path_found


def _get_route_roadblock(
    roadblock_id: str,
    map_api: AbstractMap,
) -> RoadBlockGraphEdgeMapObject:
    """Load a roadblock or connector by id."""
    roadblock = map_api._get_roadblock(roadblock_id)
    return roadblock or map_api._get_roadblock_connector(roadblock_id)


def _get_state_roadblock_candidate_ids(
    ego_state: EgoState,
    map_api: AbstractMap,
    radius: float = 1.0,
) -> List[str]:
    """Return all nearby roadblock/connector ids for a state, preserving map query order."""
    layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    roadblock_dict = map_api.get_proximal_map_objects(
        point=ego_state.center.point, radius=radius, layers=layers
    )
    roadblocks = (
        roadblock_dict[SemanticMapLayer.ROADBLOCK]
        + roadblock_dict[SemanticMapLayer.ROADBLOCK_CONNECTOR]
    )

    if not roadblocks:
        for layer in layers:
            roadblock_id, _ = map_api.get_distance_to_nearest_map_object(
                point=ego_state.center.point, layer=layer
            )
            roadblock = map_api.get_map_object(roadblock_id, layer)
            if roadblock is not None:
                roadblocks.append(roadblock)

    return list(dict.fromkeys(roadblock.id for roadblock in roadblocks))


def _get_state_lane_roadblock_candidate_ids(
    ego_state: EgoState,
    map_api: AbstractMap,
    radius: float = 1.0,
) -> List[str]:
    """Return roadblock ids from nearby lanes/connectors plus enclosing roadblocks."""
    roadblock_ids = _get_state_roadblock_candidate_ids(ego_state, map_api, radius)
    lane_layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    lane_dict = map_api.get_proximal_map_objects(
        point=ego_state.center.point, radius=radius, layers=lane_layers
    )
    lanes = lane_dict[SemanticMapLayer.LANE] + lane_dict[SemanticMapLayer.LANE_CONNECTOR]
    roadblock_ids.extend(lane.get_roadblock_id() for lane in lanes)
    return list(dict.fromkeys(roadblock_ids))


def _roadblock_heading_error_to_state(
    roadblock_id: str,
    ego_state: EgoState,
    map_api: AbstractMap,
) -> float:
    """Return the smallest nearest-centerline heading error for lanes in a roadblock."""
    roadblock = _get_route_roadblock(roadblock_id, map_api)
    if roadblock is None:
        return np.inf

    ego_pose = ego_state.center
    ego_position = ego_pose.point.array
    min_heading_error = np.inf
    for lane in roadblock.interior_edges:
        lane_discrete_path: List[StateSE2] = lane.baseline_path.discrete_path
        if not lane_discrete_path:
            continue

        lane_points = np.array(
            [state.point.array for state in lane_discrete_path], dtype=np.float64
        )
        lane_distances = np.linalg.norm(lane_points - ego_position[None, :], axis=-1)
        closest_idx = int(np.argmin(lane_distances))
        heading_error = float(
            abs(normalize_angle(lane_discrete_path[closest_idx].heading - ego_pose.heading))
        )
        min_heading_error = min(min_heading_error, heading_error)

    return min_heading_error


def _filter_start_roadblocks_by_heading(
    start_roadblock_ids: List[str],
    ego_state: EgoState,
    map_api: AbstractMap,
    heading_error_threshold: float = np.pi / 4,
) -> List[str]:
    """Keep start roadblocks aligned with ego heading; fallback to all if none pass."""
    aligned_roadblock_ids = [
        roadblock_id
        for roadblock_id in start_roadblock_ids
        if _roadblock_heading_error_to_state(roadblock_id, ego_state, map_api) <= heading_error_threshold
    ]
    return aligned_roadblock_ids or start_roadblock_ids


def _search_shortest_roadblock_path(
    start_roadblock_ids: List[str],
    target_roadblock_ids: List[str],
    map_api: AbstractMap,
    max_depth: int,
) -> Tuple[List[RoadBlockGraphEdgeMapObject], List[str], bool]:
    """Search the shortest forward roadblock path from any start to any target."""
    best_path: List[RoadBlockGraphEdgeMapObject] = []
    best_path_ids: List[str] = []

    for start_roadblock_id in start_roadblock_ids:
        if _get_route_roadblock(start_roadblock_id, map_api) is None:
            continue

        path, path_ids, path_found = _search_route_roadblock_path(
            start_roadblock_id, target_roadblock_ids, map_api, max_depth
        )
        if not path_found:
            continue

        if not best_path_ids or len(path_ids) < len(best_path_ids):
            best_path = path
            best_path_ids = path_ids

    return best_path, best_path_ids, bool(best_path_ids)


def _extract_expert_route_roadblocks(
    expert_route_map: List[List],
    map_api: AbstractMap,
) -> Tuple[List[RoadBlockGraphEdgeMapObject], List[str]]:
    """Extract expert-observed roadblocks in first-seen temporal order.

    We only keep the primary lane candidate for each pose so the roadblock
    sequence reflects the expert trajectory order instead of accumulating all
    candidate lanes returned by get_route().
    """
    expert_route_roadblock_ids: List[str] = []
    expert_route_roadblocks: List[RoadBlockGraphEdgeMapObject] = []
    seen_roadblock_ids = set()

    for lane_obj_list in expert_route_map:
        if not lane_obj_list:
            continue

        roadblock_id = lane_obj_list[0].get_roadblock_id()
        if roadblock_id in seen_roadblock_ids:
            continue

        roadblock = _get_route_roadblock(roadblock_id, map_api)
        if roadblock is None:
            continue

        expert_route_roadblock_ids.append(roadblock_id)
        expert_route_roadblocks.append(roadblock)
        seen_roadblock_ids.add(roadblock_id)

    return expert_route_roadblocks, expert_route_roadblock_ids


def _route_roadblocks_cover_expert_trajectory(
    expert_trajectory: List[EgoState],
    map_api: AbstractMap,
    route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject],
) -> bool:
    """Return True when every expert state has at least one on-route roadblock candidate.

    This is intentionally more permissive than get_route(...): near merges or lane
    boundaries, get_route may pick a single off-route candidate even when the
    preloaded mission route already provides a valid overlapping roadblock.
    """
    if not expert_trajectory or not route_roadblock_dict:
        return False

    for expert_state in expert_trajectory:
        _, roadblock_candidates = get_current_roadblock_candidates(
            expert_state, map_api, route_roadblock_dict.keys()
        )
        if not any(candidate.id in route_roadblock_dict for candidate in roadblock_candidates):
            return False

    return True


def _build_connected_expert_route_in_order(
    expert_route_roadblock_ids: List[str],
    map_api: AbstractMap,
    max_depth: int,
) -> Tuple[List[RoadBlockGraphEdgeMapObject], List[str]]:
    """Build a connected expert route while preserving expert visit order.

    The previous longest-subsequence logic could skip earlier expert-observed
    roadblocks and connect directly to later ones. Here we only connect
    consecutive expert-observed roadblocks, so the corrected route is a
    connected prefix that follows the expert's temporal order.
    """
    connected_path: List[RoadBlockGraphEdgeMapObject] = []
    connected_path_ids: List[str] = []
    connected_path_id_set = set()

    if not expert_route_roadblock_ids:
        return connected_path, connected_path_ids

    for target_roadblock_id in expert_route_roadblock_ids:
        if target_roadblock_id in connected_path_id_set:
            continue

        target_roadblock = _get_route_roadblock(target_roadblock_id, map_api)
        if target_roadblock is None:
            continue

        if not connected_path_ids:
            connected_path = [target_roadblock]
            connected_path_ids = [target_roadblock_id]
            connected_path_id_set = {target_roadblock_id}
            continue

        path, path_id, path_found = _search_route_roadblock_path(
            connected_path_ids[-1], target_roadblock_id, map_api, max_depth
        )
        if not path_found:
            break

        appended_path = path[1:]
        appended_path_ids = path_id[1:]

        # Stop once the forward connection would create a loop; this keeps the
        # corrected route aligned with the first monotonic expert traversal.
        if any(roadblock_id in connected_path_id_set for roadblock_id in appended_path_ids):
            break

        connected_path.extend(appended_path)
        connected_path_ids.extend(appended_path_ids)
        connected_path_id_set.update(appended_path_ids)

    return connected_path, connected_path_ids


def _extend_route_by_one_roadblock(
    route_roadblocks: List[RoadBlockGraphEdgeMapObject],
    route_roadblock_ids: List[str],
) -> Tuple[List[RoadBlockGraphEdgeMapObject], List[str]]:
    """Extend a connected route by one incoming and one outgoing roadblock when available."""
    if not route_roadblocks:
        return route_roadblocks, route_roadblock_ids

    route_roadblock_id_set = set(route_roadblock_ids)

    if route_roadblocks[0].incoming_edges:
        previous_roadblock = route_roadblocks[0].incoming_edges[0]
        if previous_roadblock.id not in route_roadblock_id_set:
            route_roadblocks = [previous_roadblock] + route_roadblocks
            route_roadblock_ids = [previous_roadblock.id] + route_roadblock_ids
            route_roadblock_id_set.add(previous_roadblock.id)

    if route_roadblocks[-1].outgoing_edges:
        next_roadblock = route_roadblocks[-1].outgoing_edges[0]
        if next_roadblock.id not in route_roadblock_id_set:
            route_roadblocks = route_roadblocks + [next_roadblock]
            route_roadblock_ids = route_roadblock_ids + [next_roadblock.id]

    return route_roadblocks, route_roadblock_ids


def get_current_roadblock_candidates(
    ego_state: EgoState,
    map_api: AbstractMap,
    route_roadblocks: List[str],
    heading_error_thresh: float = np.pi / 4,
    displacement_error_thresh: float = 3,
) -> Tuple[RoadBlockGraphEdgeMapObject, List[RoadBlockGraphEdgeMapObject]]:
    """
    Determine roadblock candidates where ego is located.

    Args:
        ego_state: class containing ego state.
        map_api: map object.
        route_roadblocks: list of on-route roadblock ids.
        heading_error_thresh: maximum heading error in radians.
        displacement_error_thresh: maximum displacement in meters.

    Returns:
        Tuple[RoadBlockGraphEdgeMapObject, List[RoadBlockGraphEdgeMapObject]]:
            most promising roadblock and all candidates.
    """
    ego_pose: StateSE2 = ego_state.rear_axle
    roadblock_candidates = []

    # Get the closest (within 1.0m) roadblock (if the car is on the road) 
    # or the closet roadblock connector (if the car is on the intersection)
    layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    roadblock_dict = map_api.get_proximal_map_objects(
        point=ego_pose.point, radius=1.0, layers=layers
    )
    roadblock_candidates = (
        roadblock_dict[SemanticMapLayer.ROADBLOCK]
        + roadblock_dict[SemanticMapLayer.ROADBLOCK_CONNECTOR]
    )

    # If the closest layer is further than 1.0m, get the closest
    if not roadblock_candidates:
        for layer in layers:
            roadblock_id_, distance = map_api.get_distance_to_nearest_map_object(
                point=ego_pose.point, layer=layer
            )
            roadblock = map_api.get_map_object(roadblock_id_, layer)

            if roadblock:
                roadblock_candidates.append(roadblock)

    on_route_candidates, on_route_candidate_displacement_errors = [], []
    candidates, candidate_displacement_errors = [], []

    roadblock_displacement_errors = []
    roadblock_heading_errors = []

    for idx, roadblock in enumerate(roadblock_candidates):
        lane_displacement_error, lane_heading_error = np.inf, np.inf

        for lane in roadblock.interior_edges:
            lane_discrete_path: List[StateSE2] = lane.baseline_path.discrete_path
            lane_discrete_points = np.array(
                [state.point.array for state in lane_discrete_path], dtype=np.float64
            )
            lane_state_distances = (
                (lane_discrete_points - ego_pose.point.array[None, ...]) ** 2.0
            ).sum(axis=-1) ** 0.5
            argmin = np.argmin(lane_state_distances)

            heading_error = np.abs(
                normalize_angle(lane_discrete_path[argmin].heading - ego_pose.heading)
            )
            displacement_error = lane_state_distances[argmin]

            if displacement_error < lane_displacement_error:
                lane_heading_error, lane_displacement_error = (
                    heading_error,
                    displacement_error,
                )

            if (
                heading_error < heading_error_thresh
                and displacement_error < displacement_error_thresh
            ):
                if roadblock.id in route_roadblocks:
                    on_route_candidates.append(roadblock)
                    on_route_candidate_displacement_errors.append(displacement_error)
                else:
                    candidates.append(roadblock)
                    candidate_displacement_errors.append(displacement_error)

        roadblock_displacement_errors.append(lane_displacement_error)
        roadblock_heading_errors.append(lane_heading_error)

    if on_route_candidates:  # prefer on-route roadblocks
        return (
            on_route_candidates[np.argmin(on_route_candidate_displacement_errors)],
            on_route_candidates,
        )
    elif candidates:  # fallback to most promising candidate
        return candidates[np.argmin(candidate_displacement_errors)], candidates

    # otherwise, just find any close roadblock
    return (
        roadblock_candidates[np.argmin(roadblock_displacement_errors)],
        roadblock_candidates,
    )


def route_roadblock_correction(
    ego_state: EgoState,
    map_api: AbstractMap,
    route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject],
    search_depth_backward: int = 15,
    search_depth_forward: int = 30,
    scenario: AbstractScenario = None,
) -> List[str]:
    """
    Applies several methods to correct route roadblocks.
    :param ego_state: class containing ego state
    :param map_api: map object
    :param route_roadblocks_dict: dictionary of on-route roadblocks
    :param search_depth_backward: depth of forward BFS search, defaults to 15
    :param search_depth_forward:  depth of backward BFS search, defaults to 30
    :return: list of roadblock id's of corrected route
    """
    # print("Applying route correction...\n"
    #       f"use scenario for correction: {scenario is not None}, ")
    route_roadblocks = list(route_roadblock_dict.values())
    route_roadblock_ids = list(route_roadblock_dict.keys())

    # Fix 0: if scenario is provided, use the 15s expert future endpoints to
    # search a connected roadblock path. This avoids trusting get_route()'s
    # per-pose first lane candidate near intersections.
    if scenario is not None:
        expert_future_traj = scenario.get_ego_future_trajectory(0, 20.0, 100)
        expert_future_states = [state for state in expert_future_traj]
        if expert_future_states:
            start_roadblock_ids = _get_state_roadblock_candidate_ids(
                expert_future_states[0], map_api
            )
            start_roadblock_ids = _filter_start_roadblocks_by_heading(
                start_roadblock_ids, expert_future_states[0], map_api
            )
            target_roadblock_ids = _get_state_lane_roadblock_candidate_ids(
                expert_future_states[-1], map_api
            )
            _, connected_route_ids, path_found = _search_shortest_roadblock_path(
                start_roadblock_ids,
                target_roadblock_ids,
                map_api,
                search_depth_forward,
            )
            if path_found:
                return connected_route_ids


    starting_block, starting_block_candidates = get_current_roadblock_candidates(
        ego_state, map_api, route_roadblock_dict.keys()
    )
    starting_block_ids = [roadblock.id for roadblock in starting_block_candidates]

    

    # Fix 1: when agent starts off-route
    if starting_block.id not in route_roadblock_ids:
        # Backward search if current roadblock not in route
        graph_search = BreadthFirstSearchRoadBlock(
            route_roadblock_ids[0], map_api, forward_search=False
        )
        (path, path_id), path_found = graph_search.search(
            starting_block_ids, max_depth=search_depth_backward
        )

        if path_found:
            route_roadblocks[:0] = path[:-1]
            route_roadblock_ids[:0] = path_id[:-1]

        else:
            # Forward search to any route roadblock
            graph_search = BreadthFirstSearchRoadBlock(
                starting_block.id, map_api, forward_search=True
            )
            (path, path_id), path_found = graph_search.search(
                route_roadblock_ids[:3], max_depth=search_depth_forward
            )

            if path_found:
                end_roadblock_idx = np.argmax(
                    np.array(route_roadblock_ids) == path_id[-1]
                )

                route_roadblocks = route_roadblocks[end_roadblock_idx + 1 :]
                route_roadblock_ids = route_roadblock_ids[end_roadblock_idx + 1 :]

                route_roadblocks[:0] = path
                route_roadblock_ids[:0] = path_id

    # Fix 2: check if roadblocks are linked, search for links if not
    roadblocks_to_append = {}
    for i in range(len(route_roadblocks) - 1):
        next_incoming_block_ids = [
            _roadblock.id for _roadblock in route_roadblocks[i + 1].incoming_edges
        ]
        is_incoming = route_roadblock_ids[i] in next_incoming_block_ids

        if is_incoming:
            continue

        graph_search = BreadthFirstSearchRoadBlock(
            route_roadblock_ids[i], map_api, forward_search=True
        )
        (path, path_id), path_found = graph_search.search(
            route_roadblock_ids[i + 1], max_depth=search_depth_forward
        )

        if path_found and path and len(path) >= 3:
            path, path_id = path[1:-1], path_id[1:-1]
            roadblocks_to_append[i] = (path, path_id)

    # append missing intermediate roadblocks
    offset = 1
    for i, (path, path_id) in roadblocks_to_append.items():
        route_roadblocks[i + offset : i + offset] = path
        route_roadblock_ids[i + offset : i + offset] = path_id
        offset += len(path)

    # Fix 3: cut route-loops
    route_roadblocks, route_roadblock_ids = remove_route_loops(
        route_roadblocks, route_roadblock_ids
    )

    return route_roadblock_ids


def remove_route_loops(
    route_roadblocks: List[RoadBlockGraphEdgeMapObject],
    route_roadblock_ids: List[str],
) -> Tuple[List[str], List[RoadBlockGraphEdgeMapObject]]:
    """
    Remove ending of route, if the roadblock are intersecting the route (forming a loop).
    :param route_roadblocks: input route roadblocks
    :param route_roadblock_ids: input route roadblocks ids
    :return: tuple of ids and roadblocks of route without loops
    """

    roadblock_occupancy_map = None
    loop_idx = None

    for idx, roadblock in enumerate(route_roadblocks):
        # loops only occur at intersection, thus searching for roadblock-connectors.
        if str(roadblock.__class__.__name__) == "NuPlanRoadBlockConnector":
            if not roadblock_occupancy_map:
                roadblock_occupancy_map = STRTreeOccupancyMapFactory.get_from_geometry(
                    [roadblock.polygon], [roadblock.id]
                )
                continue

            strtree, index_by_id = roadblock_occupancy_map._build_strtree()
            indices = strtree.query(roadblock.polygon)
            if len(indices) > 0:
                for geom in strtree.geometries.take(indices):
                    area = geom.intersection(roadblock.polygon).area
                    if area > 1:
                        loop_idx = idx
                        break
                if loop_idx:
                    break

            roadblock_occupancy_map.insert(roadblock.id, roadblock.polygon)

    if loop_idx:
        route_roadblocks = route_roadblocks[:loop_idx]
        route_roadblock_ids = route_roadblock_ids[:loop_idx]

    return route_roadblocks, route_roadblock_ids

def QueryNearestLaneLink(map_api: AbstractMap, point: Point2D, heading: float, radius: float = 2.0, heading_error: float = np.pi / 4.0, num: int = 3) -> List[LaneConnector]:
        """
        Find nearby lane connectors whose heading aligns with the query heading.

        Args:
            point: query point in map coordinates.
            heading: query heading in radians.
            radius: spatial search radius in meters.
            heading_error: maximum heading deviation in radians.
            num: maximum number of lane connectors to return.

        Returns:
            List[LaneConnector]: up to `num` lane connectors sorted by distance.
        """
        lane_link_dict = map_api.get_proximal_map_objects(point, radius, [SemanticMapLayer.LANE_CONNECTOR])
        lane_links = lane_link_dict[SemanticMapLayer.LANE_CONNECTOR]
        
        def IsHeadingAlongRefLine(point: Point2D, heading: float, ref_line: PolylineMapObject, heading_error: float) -> Tuple[bool, float]:
            nearest_pose = ref_line.get_nearest_pose_from_position(point)
            return close_angle(nearest_pose.heading, heading, heading_error), ((nearest_pose.x - point.x) ** 2 + (nearest_pose.y - point.y) ** 2)
        
        nearest_lane_links = []
        distances = []
        for lane_link in lane_links:
            if len(nearest_lane_links) >= 10:
                break
            along, distance_squared = IsHeadingAlongRefLine(point, heading, lane_link.baseline_path, heading_error)
            if along:
                nearest_lane_links.append(lane_link)
                distances.append(distance_squared)

        if len(distances) == 0:
            return []
        else:
            zipped_lists = list(zip(distances, nearest_lane_links))
            zipped_lists.sort(key=lambda x: x[0])
            distances_sorted, nearest_lane_links_sorted = zip(*zipped_lists)
            distances_sorted = list(distances_sorted)
            nearest_lane_links_sorted = list(nearest_lane_links_sorted)
            return nearest_lane_links_sorted[:num]
        

def close_angle(angle1: float, angle2: float, threshold: float) -> bool:
    """
    Check whether two angles are within a given angular distance of each other.

    Args:
        angle1: first angle in radians.
        angle2: second angle in radians.
        threshold: maximum allowed angular difference in radians.

    Returns:
        bool: True if |normalize(angle1 - angle2)| < threshold.
    """
    angle_difference = normalize_angle(angle1 - angle2)  # in [-pi, pi)
    return abs(angle_difference) < threshold
