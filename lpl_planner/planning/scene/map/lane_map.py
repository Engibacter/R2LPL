from abc import ABC
from typing import List, Dict, Tuple, Optional
import numpy.typing as npt
import torch
import numpy as np
from collections import deque
from shapely.geometry import LineString

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import (
    LaneGraphEdgeMapObject,
    RoadBlockGraphEdgeMapObject,
)
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.maps_datatypes import (
    TrafficLightStatusData,
    TrafficLightStatusType,
)

from lpl_planner.planning.scene.map.occupancy_map import OccupancyMap
from shapely.geometry import Point, Polygon
from lpl_planner.planning.scene.map.map_utils.common_utils import normalize_angle, resample_discrete_path
from lpl_planner.planning.planner.utils.int_enum import RoadType
from lpl_planner.planning.scene.map.map_utils.roi_segement import ROIMap
from lpl_planner.planning.scene.map.map_utils.frenet_path import FrenetPath
from lpl_planner.planning.scene.map.map_utils.dijkstra import Dijkstra
from lpl_planner.planning.scene.map.map_utils.route_utils import (
    route_roadblock_correction,
    QueryNearestLaneLink,
    get_current_roadblock_candidates)
from .map_utils.bfs_roadblock import (
    BreadthFirstSearchRoadBlock,
)
from lpl_planner.planning.scene.evaluate.utils.evaluate_utils import state_array_to_coords_array
import logging
logger = logging.getLogger(__name__)
_eps = 0.00001

class LaneMap(ABC):
    def __init__(self,
                 use_local_coords: bool = True,
                 use_ref_path: bool = True,
                 tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}):
        
        self.use_local_coords = use_local_coords
        self.use_ref_path = use_ref_path

        # vessels for nuplan 
        self.discrete_center_line : Dict[str, List[StateSE2]] = {}
        self._route_roadblock_ids: List[str] = []
        self._mission_goal: StateSE2 = None
        self._route_roadblock_dict : Dict[str, RoadBlockGraphEdgeMapObject] = {}
        self._route_lane_dict : Dict[str, LaneGraphEdgeMapObject] = {} # dict object of all lane in route
        self._lane_map_dict : Dict[str, LaneGraphEdgeMapObject] = {} # dict object of all lane within map radius
        self.route_plans : Dict[str, List[LaneGraphEdgeMapObject]] = {}
        self.left_lane, self.right_lane = None, None
        self.map_radius = 100.0  # default map radius in meters
        self.ego_lane = None
        self.ego_route = None
        self._map_api: AbstractMap = None
        self._tensor_args = tensor_args
        self.frenet_path_util: FrenetPath = None
        self._drivable_area_map: Optional[OccupancyMap] = None
        self._drivable_area_map_local: Optional[OccupancyMap] = None
        self._lane_map_local: Optional[OccupancyMap] = None
        self._lane_speed_limit_hash: Dict[str, float] = {}
        self._lane_connection_hash: Dict[str, frozenset[str]] = {}
        self._lane_neighbor_hash: Dict[str, Tuple[str, ...]] = {}
        self.max_route_lane_length = 0.0
        self._goal_block_id = ""

        self.initialized = False

    def _reset(self):

        self.discrete_center_line = {}
        self._route_roadblock_dict = {}
        self._route_lane_dict = {}
        self.route_plans = {}
        self.left_lane, self.right_lane = None, None
        self.ego_lane = None
        self.ego_route = None
        self.frenet_path_util = None
        self._drivable_area_map = None
        self._drivable_area_map_local = None
        self._lane_map_local = None
        self._lane_speed_limit_hash = {}
        self._lane_connection_hash = {}
        self._lane_neighbor_hash = {}
        self.max_route_lane_length = 0.0

        self.initialized = False

    def _build_lane_connection_hash(
        self,
        lane_map_dict: Dict[str, LaneGraphEdgeMapObject],
    ) -> None:
        """Build local connectivity caches over incoming/outgoing/adjacent lane relations."""
        lane_ids = set(lane_map_dict.keys())
        lane_neighbor_hash: Dict[str, Tuple[str, ...]] = {}

        for lane_id, lane in lane_map_dict.items():
            connected_lane_ids = set()
            for neighbor in list(lane.incoming_edges) + list(lane.outgoing_edges) + list(lane.adjacent_edges):
                if neighbor is None:
                    continue
                neighbor_id = getattr(neighbor, "id", None)
                if neighbor_id is None or neighbor_id == lane_id or neighbor_id not in lane_ids:
                    continue
                connected_lane_ids.add(neighbor_id)
            lane_neighbor_hash[lane_id] = tuple(sorted(connected_lane_ids))

        lane_connection_hash: Dict[str, frozenset[str]] = {}
        unvisited_lane_ids = set(lane_ids)
        while unvisited_lane_ids:
            start_lane_id = unvisited_lane_ids.pop()
            connected_component = {start_lane_id}
            queue = deque([start_lane_id])

            while queue:
                current_lane_id = queue.popleft()
                for neighbor_lane_id in lane_neighbor_hash.get(current_lane_id, ()): 
                    if neighbor_lane_id in connected_component:
                        continue
                    connected_component.add(neighbor_lane_id)
                    unvisited_lane_ids.discard(neighbor_lane_id)
                    queue.append(neighbor_lane_id)

            frozen_component = frozenset(connected_component)
            for lane_id in frozen_component:
                lane_connection_hash[lane_id] = frozen_component

        self._lane_neighbor_hash = lane_neighbor_hash
        self._lane_connection_hash = lane_connection_hash

    def lanes_are_connected(self, lane_ids: List[str]) -> bool:
        """Return True when all lane ids belong to the same local connected component."""
        if len(lane_ids) <= 1:
            return True

        first_lane_id = lane_ids[0]
        connected_component = self._lane_connection_hash.get(first_lane_id)
        if connected_component is None:
            return False

        return all(lane_id in connected_component for lane_id in lane_ids[1:])
    

    def init_from_planner_init(self, 
                               map_api: AbstractMap,
                               route_roadblock_ids: List[str],
                               mission_goal: Optional[StateSE2] = None,
                               scenario: Optional[AbstractScenario] = None,):
        """
        Initialize the lane map with planner input.
        :param planner_input: PlannerInput object containing ego state and map API
        """
        self._map_api = map_api
        self._load_route_dicts(route_roadblock_ids)
        self._mission_goal = mission_goal
        if scenario is not None:
            route_roadblock_ids = route_roadblock_correction(
                scenario.initial_ego_state, map_api, self._route_roadblock_dict, scenario=scenario
            )
            self._load_route_dicts(route_roadblock_ids)
        self.ego_route = None
        self.ego_lane = None
        self.frenet_path_util = None
        self.no_route = None

    def step_with_planner_init(self, 
                               ego_state: EgoState, 
                               scenario: AbstractScenario = None,
                               iteration: Optional[int] = None) ->None:
        """
        Update the lane map with the current ego state.
        """
        if self._map_api is None:
            raise RuntimeError("LaneMap.step_with_planner_init() called before init_from_planner_init(): map_api is None.")
        

        self._ego_state = ego_state
        self._drivable_area_map, self._drivable_area_map_local = self._update_lane_map_dict(ego_state)
        ego_lane = self.get_starting_lane(ego_state)
        remaining_route_len = 0.0 if self.frenet_path_util is None else self.frenet_path_util.cumulative_s[-1]

        # skip route search if has been searched
        if self.ego_route is None or remaining_route_len < 60.0:

            ego_roadblock_ids = ego_lane.get_roadblock_id()
            if ego_roadblock_ids not in self._route_roadblock_dict.keys():
                # logger.warning(f"Ego lane roadblock id {ego_roadblock_ids} not in route roadblock dict keys.")
                self.ego_route = self.get_free_drive_route_plan(ego_lane, ego_state, scenario=scenario)
            else:
                self.ego_route = self.get_route_plan_by_search(ego_lane)

            self.ego_route = self.refine_route(ego_state, self.ego_route, scenario)

        self._update_ref_path(ego_state)
        self.frenet_path_util = FrenetPath(self.ref_path)

    def init_from_scenario(self, 
                            ego_state: EgoState,
                            map_api: AbstractMap,
                            route_roadblock_ids: List[str],
                            scenario: AbstractScenario = None,
                            use_route_correction: bool = False,
                            use_scenario_for_route_correction: bool = False,
                            ) ->None:
        """
        Initialize the lane map with AbstractMap.
        :param planner_input: PlannerInput object containing ego state and map API
        """
        self._reset()
        self._ego_state = ego_state
        self._map_api = map_api
        self._load_route_dicts(route_roadblock_ids)

        if use_route_correction:
            route_roadblock_ids = route_roadblock_correction(
                ego_state, 
                self._map_api, 
                self._route_roadblock_dict, 
                scenario = scenario if use_scenario_for_route_correction else None
            )
            self._load_route_dicts(route_roadblock_ids)
        self._drivable_area_map, self._drivable_area_map_local = self._update_lane_map_dict(ego_state)

        if self.use_ref_path:
            ego_lane = self.get_starting_lane(ego_state)
            self.ego_lane = ego_lane
            ego_roadblock_ids = ego_lane.get_roadblock_id()
            if ego_roadblock_ids not in self._route_roadblock_dict.keys():
                ego_route = self.get_free_drive_route_plan(ego_lane, ego_state, scenario=scenario)
            else:
                ego_route = self.get_route_plan_by_search(ego_lane, search_depth=20)
            
            self.ego_route = self.refine_route(ego_state, ego_route, scenario=scenario)
            self._update_ref_path(ego_state)
            self.frenet_path_util = FrenetPath(self.ref_path)

        self.initialized = True

    def update(self,
                ego_state: EgoState,
                map_api: AbstractMap,
                route_roadblock_ids: List[str],
                ) ->None:
        
        self._map_api = map_api
        self._load_route_dicts(route_roadblock_ids)
        self._drivable_area_map, self._drivable_area_map_local = self._update_lane_map_dict(ego_state)

        ego_lane = self.get_starting_lane(ego_state)

        # skip route search if still in still same lane form lasted searched lane
        if self.ego_route is None :
            self.ego_lane = ego_lane
            ego_roadblock_ids = ego_lane.get_roadblock_id()
            if ego_roadblock_ids not in self._route_roadblock_dict.keys():
                self.ego_route = self.get_free_drive_route_plan(ego_lane, ego_state)
            else:
                self.ego_route = self.get_route_plan_by_search(ego_lane)
        self.ego_route = self.refine_route(ego_state, self.ego_route)
        self._update_ref_path(ego_state)
        self.frenet_path_util = FrenetPath(self.ref_path)

    def route_correction(self, 
                         ego_state: EgoState,
                         scenario: AbstractScenario = None) -> None:
        
        route_roadblock_ids = route_roadblock_correction(
                ego_state, 
                self._map_api, 
                self._route_roadblock_dict, 
                scenario = scenario,
            )
        
        self._load_route_dicts(route_roadblock_ids)
        
        return 

    def frenet_to_cartesian(self, frenet_points: npt.NDArray, with_yaw: bool=False) -> npt.NDArray:
        """Convert frenet coordinates to cartesian coordinates using precomputed FrenetPath utility.
        
        Args:
            frenet_points (npt.NDArray): Frenet coordinates of shape (..., 2) containing s and d values.

        Returns:
            npt.NDArray: Cartesian coordinates of shape (..., 2) containing x and y values.
        """
        if self.frenet_path_util is None:
            raise ValueError("FrenetPath utility is not initialized. Please call update() first.")
        
        points = self.frenet_path_util.frenet_to_cartesian(frenet_points, with_yaw)
        return points
    
    def cartesian_to_frenet(self, points: npt.NDArray) -> npt.NDArray:
        """Convert cartesian coordinates to frenet coordinates using precomputed FrenetPath utility.
        
        Args:
            points (npt.NDArray): Cartesian coordinates of shape (..., 3) containing x, y, and theta values.
        
        Returns:
            npt.NDArray: Frenet coordinates of shape (..., 2) containing s and d values.
        """
        if self.frenet_path_util is None:
            raise ValueError("FrenetPath utility is not initialized. Please call update() first.")
        
        sd = self.frenet_path_util.cartesian_to_frenet(points)
        return sd

    def _load_route_dicts(self,route_roadblock_ids: List[str]):

        # remove repeated ids while remaining order in list
        self._route_roadblock_dict = {}
        self._route_lane_dict = {}
        max_length = 0
        self.max_route_lane_length = 0.0
        route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))
        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(
                id_, SemanticMapLayer.ROADBLOCK_CONNECTOR
            )

            self._route_roadblock_dict[block.id] = block

            for lane in block.interior_edges:
                self._route_lane_dict[lane.id] = lane
                max_length = max(max_length,lane.baseline_path.length)

        self.max_route_lane_length = max_length

    def _update_lane_map_dict(self, ego_state: EgoState):
        
        # read lane object within map_radius
        DRIVABLE_MAP_LAYERS = [
            SemanticMapLayer.ROADBLOCK,
            SemanticMapLayer.ROADBLOCK_CONNECTOR,
            SemanticMapLayer.CARPARK_AREA,
            SemanticMapLayer.INTERSECTION
        ]

        position = ego_state.rear_axle.point
        
        drivable_area = self._map_api.get_proximal_map_objects(
            position, self.map_radius, DRIVABLE_MAP_LAYERS
        )
        ego_position = ego_state.rear_axle.point.array
        ego_yaw = - ego_state.rear_axle.heading
        rotation_matrix = np.array([[np.cos(ego_yaw), np.sin(ego_yaw)],
                                    [-np.sin(ego_yaw), np.cos(ego_yaw)]])
        
        # collect lane polygons in list, save on-route indices
        drivable_polygons: List[Polygon] = []
        drivable_polygons_local: List[Polygon] = []
        drivable_polygon_ids: List[str] = []
        lane_map_dict : Dict[str,LaneGraphEdgeMapObject] = {}
        lane_map_ids: List[str] = []
        lane_polygon_local: List[Polygon] = []
        lane_speed_limit_hash: Dict[str, float] = {}

        for type in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
            for roadblock in drivable_area[type]:
                for lane in roadblock.interior_edges:
                    drivable_polygons.append(lane.polygon)
                    drivable_polygon_ids.append(lane.id)
                    lane_map_dict.update({lane.id:lane})
                    lane_speed_limit_hash[lane.id] = float(lane.speed_limit_mps or 15.0)
                    # covert polygon to local coordinates
                    polygon_coords = np.array(lane.polygon.exterior.coords)
                    local_polygon_coords = (polygon_coords - ego_position) @ rotation_matrix
                    lane_map_ids.append(lane.id)
                    lane_polygon_local.append(Polygon(local_polygon_coords))
                    drivable_polygons_local.append(Polygon(local_polygon_coords))

        for carpark in drivable_area[SemanticMapLayer.CARPARK_AREA]:
            drivable_polygons.append(carpark.polygon)
            drivable_polygon_ids.append(carpark.id)
            # lane_map_dict.update({carpark.id:carpark})
            # covert polygon to local coordinates
            polygon_coords = np.array(carpark.polygon.exterior.coords)
            local_polygon_coords = (polygon_coords - ego_position) @ rotation_matrix
            drivable_polygons_local.append(Polygon(local_polygon_coords))

        for intersection in drivable_area[SemanticMapLayer.INTERSECTION]:
            drivable_polygons.append(intersection.polygon)
            drivable_polygon_ids.append(intersection.id)
            polygon_coords = np.array(intersection.polygon.exterior.coords)
            local_polygon_coords = (polygon_coords - ego_position) @ rotation_matrix
            drivable_polygons_local.append(Polygon(local_polygon_coords))

        self._lane_map_dict = lane_map_dict
        self._lane_speed_limit_hash = lane_speed_limit_hash
        self._build_lane_connection_hash(lane_map_dict)
        self._lane_map_local = OccupancyMap(lane_map_ids, lane_polygon_local)

        # create occupancy map with lane polygons
        drivable_area_map = OccupancyMap(drivable_polygon_ids, drivable_polygons)
        drivable_area_map_local = OccupancyMap(drivable_polygon_ids, drivable_polygons_local)

        return drivable_area_map, drivable_area_map_local

    def get_free_drive_route_plan(self, 
                                  current_lane: LaneGraphEdgeMapObject,
                                  ego_state: EgoState,
                                  search_depth: int = 10,
                                  scenario: AbstractScenario = None,
                                  ) -> List[LaneGraphEdgeMapObject]:
        """
        Get free-drive route plan from current lane to the end of the lane graph.
        :param current_lane: lane object of starting lane.
        :return: list of lanes for planned route
        """
        # check if out roadblocks within depth in route
        route_keys = list(self._route_roadblock_dict.keys())
        target_set = set(route_keys)
        found_path_rb_ids: Optional[List[str]] = None

        q = deque([(current_lane, 0, [current_lane.get_roadblock_id()])])
        visited_lanes = set([current_lane.id])

        while q:
            lane, d, path_rb_ids = q.popleft()
            if d >= search_depth:
                continue
            for nxt in lane.outgoing_edges:
                if nxt is None or nxt.id in visited_lanes:
                    continue
                visited_lanes.add(nxt.id)
                rb_id = nxt.get_roadblock_id()
                # print(f'At depth {d+1}, exploring lane {nxt.id} with roadblock id {rb_id}')
                # Roadblock sequence along the explored path.
                new_path = path_rb_ids + [rb_id]
                # Stop once the free-drive search reconnects to the existing route.
                if rb_id in target_set:
                    found_path_rb_ids = new_path
                    q.clear()
                    break
                q.append((nxt, d + 1, new_path))
        # print(f'Found free-drive route to route after {len(found_path_rb_ids) if found_path_rb_ids else 0} roadblocks.')
        # print(f'Free-drive roadblock ids: {found_path_rb_ids}')
        if found_path_rb_ids:
            # Prepend the reconnecting roadblocks while preserving discovery order.
            to_insert = []
            for rb_id in found_path_rb_ids:
                if rb_id not in to_insert:
                    to_insert.append(rb_id)

            new_route_roadblock_ids = []
            for rb_id in to_insert:
                if rb_id not in new_route_roadblock_ids:
                    new_route_roadblock_ids.append(rb_id)
            for rb_id in route_keys:
                if rb_id not in new_route_roadblock_ids:
                    new_route_roadblock_ids.append(rb_id)

            self._load_route_dicts(new_route_roadblock_ids)
            # print(f'updated route roadblock dict keys: {list(self._route_roadblock_dict.keys())}')

            # The free-drive path now reconnects to the route, so switch back to route-based search.
            return self.get_route_plan_by_search(current_lane)
        
        # no route found, do free-drive till end of lane graph or depth limit
        route_plan: List[LaneGraphEdgeMapObject] = []

        # use expert history to guide route plan if scenario provided
        if scenario is not None:
            expert_trajectory = scenario.get_ego_future_trajectory(iteration=0, time_horizon=15.0)
            expert_states = [ego_state for ego_state in expert_trajectory]
            expert_traj = np.array([ego_state.rear_axle.array for ego_state in expert_states])
            expert_yaw = np.array([ego_state.rear_axle.heading for ego_state in expert_states])
            expert_half_width = ego_state.car_footprint.half_width + 1.0
            
            # ----- Build expert future trajectory buffer and derive a route along intersecting lanes -----
            if expert_traj.shape[0] >= 2:
                # print("Using expert future trajectory to guide free-drive route planning.")
                traj_linestring = LineString(expert_traj[:, :2])

                # Use flat end caps (cap_style=2) to avoid round extensions, keep topology simple
                traj_buffer_poly: Polygon = traj_linestring.buffer(expert_half_width, cap_style=2, join_style=2)

                intersect_lane_ids: List[LaneGraphEdgeMapObject] = self._drivable_area_map.intersects(traj_buffer_poly)
                current_intersect_lanes = self._drivable_area_map.intersects(ego_state.car_footprint.geometry)
                # expand current intersect lanes to include neighboring lanes
                expanded_intersect_lanes = []
                current_intersect_rb_ids = set()
                for ln_id in current_intersect_lanes:
                    if ln_id not in self._lane_map_dict:
                        continue
                    ln_obj = self._lane_map_dict[ln_id]
                    current_intersect_rb_ids.add(ln_obj.get_roadblock_id())
                for rb_id in current_intersect_rb_ids:
                    rb_obj = self._map_api.get_map_object(rb_id, SemanticMapLayer.ROADBLOCK)
                    if rb_obj is None:
                        rb_obj = self._map_api.get_map_object(rb_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
                    if rb_obj is None:
                        continue
                    for lane in rb_obj.interior_edges:
                        if lane.id not in self._lane_map_dict:
                            continue
                        expanded_intersect_lanes.append(lane.id)
                
                # Nothing intersecting -> fall back to later logic
                if intersect_lane_ids:
                    # Construct the longest connected chain inside buffer.
                    # Each next lane must be an outgoing edge of previous and intersect the buffer.
                    def build_chain(seed: LaneGraphEdgeMapObject,
                                    max_depth: int = 10) -> Tuple[List[LaneGraphEdgeMapObject], float]:
                        def lane_buffer_score(lane_obj: LaneGraphEdgeMapObject) -> float:
                            """Score a lane by expert endpoint progress and heading alignment."""
                            cl = np.array([s.array for s in lane_obj.baseline_path.discrete_path])
                            ls = LineString(cl[:, :2])
                            # inter = ls.intersection(traj_buffer_poly)
                            project_dist = ls.project(Point(expert_traj[-1, :2]))

                            lane_yaw = np.array([s.heading for s in lane_obj.baseline_path.discrete_path])
                            # Match lane start/mid/end to nearby expert samples and average heading error.
                            sample_idx = [0, cl.shape[0] // 2, cl.shape[0] - 1]
                            lane_sample_xy = cl[sample_idx, :2] # [3, 2]
                            lane_sample_yaw = lane_yaw[sample_idx]
                            
                            expert_xy = expert_traj[:, :2] # [N, 2]
                            dists = np.linalg.norm(expert_xy[:, np.newaxis, :] - lane_sample_xy[np.newaxis, :, :], axis=-1) # [N, 3]
                            dists_min_idx = np.argmin(dists, axis=0)  # [3,]
                            expert_sample_yaw = expert_yaw[dists_min_idx]  # [3,]

                            yaw_diffs = np.abs(normalize_angle(lane_sample_yaw - expert_sample_yaw))
                            yaw_diff_avg = float(np.mean(yaw_diffs))

                            # if inter.is_empty:
                            #     overlap_len =  0.0
                            # if inter.geom_type == "LineString":
                            #     overlap_len = inter.length
                            # elif inter.geom_type == "MultiLineString":
                            #     overlap_len = sum(seg.length for seg in inter.geoms)
                            # print(f'Lane {lane_obj.id} overlap length: {overlap_len:.2f}, yaw_diff_avg: {yaw_diff_avg:.3f}')
                            # return overlap_len * np.cos(yaw_diff_avg)  # Penalize misalignment
                            return project_dist * np.cos(yaw_diff_avg)  # Penalize misalignment

                        # Recursively accumulate future lane scores within the expert buffer.
                        memo = {}
                        def lookahead(lane_obj: LaneGraphEdgeMapObject, depth_left: int) -> float:
                            key = (lane_obj.id, depth_left)
                            if key in memo:
                                return memo[key]
                            base = lane_buffer_score(lane_obj)
                            if depth_left == 0:
                                memo[key] = base
                                return base
                            best_child = 0.0
                            for nxt in lane_obj.outgoing_edges:
                                if nxt is None or not nxt.polygon.intersects(traj_buffer_poly):
                                    continue
                                cand = lookahead(nxt, depth_left - 1)
                                if cand > best_child:
                                    best_child = cand
                            memo[key] = base + best_child
                            return memo[key]

                        chain = [seed]
                        current = seed
                        depth = 0
                        total_score = lane_buffer_score(seed)

                        while depth < max_depth:
                            # Candidate next lanes must intersect the expert buffer.
                            next_candidates = [
                                e for e in current.outgoing_edges
                                if (e is not None and e.polygon.intersects(traj_buffer_poly))
                            ]
                            # print(f'At depth {depth}, current lane {current.id}, found {len(next_candidates)} next candidates.')
                            # print(f'  current.outgoing_edges: {[c.id for c in current.outgoing_edges]}')
                            if not next_candidates:
                                break

                            scores = []
                            for cand in next_candidates:
                                s_self = lane_buffer_score(cand)
                                s_future = lookahead(cand, max_depth - depth - 1)
                                # Keep a small local preference so distant overlap does not dominate a poor immediate lane.
                                scores.append((cand, s_self + s_future + 0.1 * s_self))

                            scores.sort(key=lambda x: x[1], reverse=True)
                            best_lane, best_score = scores[0]

                            chain.append(best_lane)
                            current = best_lane
                            total_score += best_score
                            depth += 1

                        return chain, float(total_score)

                    # Evaluate chains from all intersecting lanes to find maximal total length
                    best_chain: List[LaneGraphEdgeMapObject] = []
                    best_len = -1.0
                    # print(f'search intersecting lanes: {current_intersect_lanes}')
                    for ln_id in expanded_intersect_lanes:
                        if ln_id not in self._lane_map_dict:
                            continue
                        ln = self._lane_map_dict[ln_id]
                        chain, clen = build_chain(ln)
                        # print(f'Built chain from lane {ln.id}, length {clen:.2f}, chain: {[l.id for l in chain]}')
                        if clen > best_len:
                            best_len = clen
                            best_chain = chain
                    # print(f'Selected best expert-guided chain of length {best_len:.2f}, lanes: {[l.id for l in best_chain]}')
                    # If starting lane differs from ego lane context, update ego_lane
                    if best_chain and current_lane not in best_chain:
                        self.ego_lane = best_chain[0]

                    # 6. Use best_chain as route_plan and update route dictionaries
                    if best_chain:
                        route_plan = best_chain
                        route_roadblock_ids = [ln.get_roadblock_id() for ln in route_plan]
                        route_roadblock_ids.extend(list(self._route_roadblock_dict.keys()))
                        self._load_route_dicts(route_roadblock_ids)
                        return route_plan  # Early return: expert guided route constructed
            

            if expert_trajectory.shape[0] >=2:
                expert_start_pos = expert_trajectory[0, :2]
                expert_end_pos = expert_trajectory[-1, :2]
                # find nearest lane to expert end pos within lane map dict
                min_dist = float('inf')
                best_lane = None
                for lane in self._lane_map_dict.values():
                    lane_centerline = lane.baseline_path.discrete_path
                    for state in lane_centerline:
                        dist = np.linalg.norm(np.array([state.x, state.y]) - expert_end_pos)
                        if dist < min_dist:
                            min_dist = dist
                            best_lane = lane
                if best_lane is not None:
                    # plan route from current lane to best_lane
                    graph_search = Dijkstra(current_lane, 
                                            list(self._lane_map_dict.keys()),
                                            list(self._route_roadblock_dict.keys()),
                                            )
                    route_plan, path_found = graph_search.search(best_lane)
                    if path_found:
                        # print(f'Free-drive route plan guided by expert history to lane {best_lane.id}')
                        # update route dicts
                        route_roadblock_ids = [route_lane.get_roadblock_id() for route_lane in route_plan]
                        route_roadblock_ids.extend(list(self._route_roadblock_dict.keys()))
                        self._load_route_dicts(route_roadblock_ids)
                        return route_plan
                    else:
                        # print(f'Graph search to expert-guided lane {best_lane.id} failed, fallback to default free-drive.')
                        pass

        lane = current_lane
        dist = 0.0
        # print(f'Starting free-drive route plan from lane {lane.id}')
        while dist < 300.0:
            route_plan.append(lane)
            dist += lane.baseline_path.length
            # print(f'  Added lane {lane.id}, length {lane.baseline_path.length:.2f}, total dist {dist:.2f}')
            # print(f'    Outgoing edges: {[edge.id for edge in lane.outgoing_edges]}')
            if len(lane.outgoing_edges) == 0:
                break
            elif len(lane.outgoing_edges) == 1:
                lane = lane.outgoing_edges[0]
            else:
                # choose lane has most aligned heading with ego lane
                current_dp = lane.baseline_path.discrete_path
                current_hist_idx = int(len(current_dp) * 0.8)
                current_yaw_rate = normalize_angle(current_dp[-1].heading - current_dp[current_hist_idx].heading)
                out_edges = lane.outgoing_edges
                # print(f'    Multiple outgoing edges, current_yaw_rate: {current_yaw_rate:.3f}')
                yaw_rates = []
                for edge in out_edges:
                    edge_dp = edge.baseline_path.discrete_path
                    lane_start_heading = edge_dp[0].heading
                    future_idx = int(len(edge_dp) * 0.2)
                    lane_future_heading = edge_dp[future_idx].heading
                    expected_yaw_rate = normalize_angle(lane_future_heading - lane_start_heading)
                    yaw_rates.append(expected_yaw_rate)
                    # yaw_rate_diff = abs(expected_yaw_rate - current_yaw_rate) if expected_yaw_rate*current_yaw_rate >=0 else 0.3
                    # yaw_rate_diffs.append(yaw_rate_diff)
                    # print(f'    Outgoing edge {edge.id}, expected_yaw_rate {expected_yaw_rate:.3f}, yaw_rate_diff {yaw_rate_diffs[-1]:.3f}')
                idx = int(np.argmax(yaw_rates))
                # idx = -1  # for debug, always choose last outgoing edge
                lane = out_edges[idx]

        # update route dicts
        route_roadblock_ids = [route_lane.get_roadblock_id() for route_lane in route_plan]
        route_roadblock_ids.extend(list(self._route_roadblock_dict.keys()))
        self._load_route_dicts(route_roadblock_ids)
        return route_plan
    
    def get_route_plan_by_search(
        self, current_lane: LaneGraphEdgeMapObject,
        search_neighbor_lane: bool = False,
        search_depth: int = 30,
        search_max_lane_length: float = 200.0,
    ) -> List[LaneGraphEdgeMapObject]:
        """
        Applies a Dijkstra search on the lane-graph to retrieve discrete centerline.
        :param current_lane: lane object of starting lane.
        :param search_depth: depth of search (for runtime), defaults to 30
        :return: list of lanes for planned route
        """
        # check if current lane have been searched
        if current_lane.id in self.route_plans.keys():
            return self.route_plans[current_lane.id]
        
        
        roadblocks = list(self._route_roadblock_dict.values())
        roadblock_ids = list(self._route_roadblock_dict.keys())

        # find current roadblock index
        start_idx = np.argmax(
            np.array(roadblock_ids) == current_lane.get_roadblock_id()
        )
        roadblock_window = []
        total_length = 0.0

        for rb in roadblocks[start_idx:]:

            if roadblock_window == [] :
                # don't count current roadblock length
                rb_ln_length = 0.0
            else:
                rb_ln_length = min([lane.baseline_path.length for lane in rb.interior_edges])
            
            total_length += rb_ln_length
            roadblock_window.append(rb)

            if total_length >= search_max_lane_length or len(roadblock_window) >= search_depth:
                break
        while total_length < search_max_lane_length:
            out_rb = roadblocks[-1].outgoing_edges
            if len(out_rb) ==0:
                break
            next_rb = out_rb[0]
            rb_ln_length = min([lane.baseline_path.length for lane in next_rb.interior_edges])
            total_length += rb_ln_length
            roadblock_window.append(next_rb)
            roadblock_ids.append(next_rb.id)
        
        if len(roadblock_ids) != len(self._route_roadblock_dict.keys()):
            # load new roadblocks into route dict
            self._load_route_dicts(roadblock_ids)

        graph_search = Dijkstra(current_lane, 
                                list(self._route_lane_dict.keys()),
                                list(self._route_roadblock_dict.keys()),
                                )
        route_plan, path_found = graph_search.search(roadblock_window[-1])


        if search_neighbor_lane:
            # search other lanes if current lane cannot reach target
            if not path_found:
                all_lanes = roadblocks[start_idx].interior_edges
                searched_lane_ids = [current_lane.id]
                # search neighbor lane first
                for lane in current_lane.adjacent_edges:
                    
                    if lane is None:
                        continue
                    # check in route block
                    if not lane.is_same_roadblock(current_lane):
                        continue
                    
                    # search ref path
                    graph_search = Dijkstra(lane, 
                                            list(self._route_lane_dict.keys()),
                                            list(self._route_roadblock_dict.keys()),)
                    route_plan, path_found = graph_search.search(roadblock_window[-1])
                    searched_lane_ids.append(lane.id)
                    
                    if path_found :
                        break
            
                # in case of the need to change lane more than twice
                # if not path_found:
                #     for lane in all_lanes:
                #         if lane.id not in searched_lane_ids:
                #             graph_search = Dijkstra(lane, 
                #                                     list(self._route_lane_dict.keys()),
                #                                     list(self._route_roadblock_dict.keys()),)
                #             route_plan, path_found = graph_search.search(roadblock_window[-1])
                #             searched_lane_ids.append(lane.id)
                #         if path_found :
                #             break
        self.route_plans.update({current_lane.id:route_plan})

        route_roadblock_ids = list(self._route_roadblock_dict.keys())
        
        # update route lane dict if new lane found
        has_new_lane = False
        for lane in route_plan:
            if lane.get_roadblock_id() not in route_roadblock_ids:
                route_roadblock_ids.append(lane.get_roadblock_id())
                has_new_lane = True

        if has_new_lane:
            self._load_route_dicts(route_roadblock_ids)

        return route_plan
    def refine_route(self, 
                     ego_state: EgoState, 
                     route_plan: List[LaneGraphEdgeMapObject],
                     scenario: AbstractScenario = None) -> List[LaneGraphEdgeMapObject]:
        """Refine the route plan based on the current ego state.
            Extend backwards to avoid discontinuities in frenet frame.
        
        Args:
            ego_state (EgoState): Current state of the ego vehicle.
            route_plan (List[LaneGraphEdgeMapObject]): Initial route plan.
        
        Returns:
            List[LaneGraphEdgeMapObject]: Refined route plan.
        """
        if not route_plan:
            return route_plan
        
        # backward refinement
        min_dist = float('inf')
        closest_lane_idx = 0
        ego_position = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y])
        refined_route = route_plan.copy()
        first_lane = route_plan[0]
        in_route_backward_lane = [lane for lane in first_lane.incoming_edges if lane.id in self._route_lane_dict.keys()]
        if in_route_backward_lane:
            refined_route.insert(0, in_route_backward_lane[0])
            
        else:
            ego_polygon = ego_state.car_footprint.geometry
            intersect_lanes = [e for e in first_lane.incoming_edges if e.polygon.intersects(ego_polygon)]
            if intersect_lanes:
                refined_route.insert(0, intersect_lanes[0])
                route_roadblock_ids = list(self._route_roadblock_dict.keys())
                rb_id = intersect_lanes[0].get_roadblock_id()
                route_roadblock_ids.insert(0, rb_id)
                self._load_route_dicts(route_roadblock_ids)
            else:
                if first_lane.incoming_edges:
                    refined_route.insert(0, first_lane.incoming_edges[0])
                    route_roadblock_ids = list(self._route_roadblock_dict.keys())
                    rb_id = first_lane.incoming_edges[0].get_roadblock_id()
                    route_roadblock_ids.insert(0, rb_id)
                    self._load_route_dicts(route_roadblock_ids)
        # forward refinement
        lane_centers = []
        for lane in refined_route:
            lane_centers.extend(lane.baseline_path.discrete_path)
        lane_centers = np.array([[state.x, state.y] for state in lane_centers])
        lane_ls = LineString(lane_centers)
        ego_point = Point(ego_position)
        ego_proj_dist = lane_ls.project(ego_point)
        remaining_length = lane_ls.length - ego_proj_dist
        # extend route forward if remaining length less than 80m
        if remaining_length < 80.0:
            last_lane = refined_route[-1]
            route_roadblock_ids = list(self._route_roadblock_dict.keys())
            while remaining_length < 80.0 and last_lane.outgoing_edges:
                out_route_forward_lane = [lane for lane in last_lane.outgoing_edges if lane.id in self._route_lane_dict.keys()]
                if out_route_forward_lane:
                    refined_route.append(out_route_forward_lane[0])
                    remaining_length += out_route_forward_lane[0].baseline_path.length
                    last_lane = out_route_forward_lane[0]
                else:
                    if scenario is not None:
                        expert_traj = scenario.get_ego_future_trajectory(iteration=0, time_horizon=10.0)
                        expert_states = [ego_state for ego_state in expert_traj]
                        expert_traj = np.array([ego_state.rear_axle.array for ego_state in expert_states])
                        expert_traj_ls = LineString(expert_traj[:, :2])
                        # find outgoing lane most aligned with expert traj
                        candidate_lanes = []
                        candidate_roadblocks = []
                        scores = []
                        for out_lane in last_lane.outgoing_edges:
                            rb = out_lane.get_roadblock_id()
                            block = self._map_api.get_map_object(rb, SemanticMapLayer.ROADBLOCK)
                            block = block or self._map_api.get_map_object(
                                rb, SemanticMapLayer.ROADBLOCK_CONNECTOR
                            )
                            if block.polygon.intersects(expert_traj_ls):
                                candidate_lanes.append(out_lane)
                                candidate_roadblocks.append(rb)
                                # compute alignment score
                                lane_dp = out_lane.baseline_path.discrete_path
                                lane_states = np.array([[state.x, state.y] for state in lane_dp])
                                lane_ls = LineString(lane_states)
                                project_dist = lane_ls.project(Point(expert_traj[-1, :2]))
                                scores.append(project_dist)
                        if candidate_lanes:
                            best_idx = int(np.argmax(scores))
                            best_lane = candidate_lanes[best_idx]
                            refined_route.append(best_lane)
                            remaining_length += best_lane.baseline_path.length
                            last_lane = best_lane
                            rb_id = candidate_roadblocks[best_idx]
                            if rb_id not in self._route_roadblock_dict.keys():
                                route_roadblock_ids.extend([rb_id])
                        else:
                            break
                    else:
                        out_lane = last_lane.outgoing_edges[0]
                        out_rb = out_lane.get_roadblock_id()
                        refined_route.append(out_lane)
                        remaining_length += out_lane.baseline_path.length
                        last_lane = out_lane
                        if out_rb not in self._route_roadblock_dict.keys():
                            route_roadblock_ids.extend([out_rb])
            # update route dicts
            self._load_route_dicts(route_roadblock_ids)   
    
        return refined_route
    

    def get_discrete_path(
            self, current_lane, route_plan:List[LaneGraphEdgeMapObject]
            ) -> List[StateSE2] :
        
        centerline_discrete_path: List[StateSE2] = []
        for lane in route_plan:
            centerline_discrete_path.extend(lane.baseline_path.discrete_path)

        # save the search result for reuse
        self.discrete_center_line.update({current_lane.id:centerline_discrete_path})

        return centerline_discrete_path
    
    def get_center_line_from_lane(
            self, current_lane : LaneGraphEdgeMapObject
    ) -> List[StateSE2]:
        """return the longest deterministic centerline from current lane

        Args:
            current_lane (LaneGraphEdgeMapObject): starting lane of current vehicle

        Returns:
            center_line (List[StateSE2]): longest deterministic center line along the lane
        """

        # check if current lane have been searched
        if current_lane.id in self.discrete_center_line.keys():
            return self.discrete_center_line[current_lane.id]
        
        dist = 0
        center_line = current_lane.baseline_path.discrete_path
        dist += current_lane.baseline_path.length
        next_lanes = current_lane.outgoing_edges
        while dist < 200:
            next_lane = next_lanes[0]
            center_line.extend(next_lane.baseline_path.discrete_path)
            dist += next_lane.baseline_path.length
            next_lanes = next_lane.outgoing_edges

        # save the search result for reuse
        self.discrete_center_line.update({current_lane.id:center_line})

        return center_line
    
    def get_starting_lane(self, ego_state: EgoState) -> LaneGraphEdgeMapObject:
        """
        Returns the most suitable starting lane, in target vehicle's vicinity.
        :param agent: Scene agent for infering states
        :return: lane object (on-route)
        """
        agent_state: StateSE2 = ego_state.rear_axle
        starting_lane: LaneGraphEdgeMapObject = None
        # 0. If ego route provided, find nearest route lane
        if self.ego_route is not None:
            closest_distance = np.inf
            for lane in self.ego_route:
                lane_discrete_path: List[
                    StateSE2
                ] = lane.baseline_path.discrete_path
                lane_state_se2_array = np.array(
                    [state.array for state in lane_discrete_path], dtype=np.float64
                )
                # calculate nearest state on baseline
                lane_distances = (
                    np.array([agent_state.x, agent_state.y])[None, ...] - lane_state_se2_array
                ) ** 2
                lane_distances = lane_distances.sum(axis=-1) ** 0.5
                lane_ego_idx = np.argmin(lane_distances)
                min_distance = lane_distances[lane_ego_idx]
                if min_distance < closest_distance:
                    closest_distance = min_distance
                    starting_lane = lane
            if starting_lane is not None:
                return starting_lane
            
        on_lanes, heading_error = self._get_intersecting_lanes(ego_state)
        
        if on_lanes:
            
            on_route_lanes = [lane for lane in on_lanes if lane.id in self._route_lane_dict.keys()]
            on_route_heading_error = [heading_error[i] for i, lane in enumerate(on_lanes) if lane.id in self._route_lane_dict.keys()]
            if on_route_lanes:
                starting_lane = on_route_lanes[np.argmin(on_route_heading_error)]
            else:
                if starting_lane is None:
                    starting_lane = on_lanes[np.argmin(np.abs(heading_error))]
            return starting_lane
        
        # 2. Option: find any intersecting or close lane
        closest_distance = np.inf
        agent_state: StateSE2 = ego_state.rear_axle
        ego_speed = ego_state.dynamic_car_state.speed
        ego_yaw_rate = ego_state.dynamic_car_state.angular_velocity
        agent_future_state = StateSE2(agent_state.x + ego_speed * np.cos(agent_state.heading) * 1.0,
                                     agent_state.y + ego_speed * np.sin(agent_state.heading) * 1.0,
                                     agent_state.heading + ego_yaw_rate * 1.0)
        ego_position_array: npt.NDArray[np.float64] = np.array([agent_state.x, agent_state.y])
        ego_heading: float = agent_state.heading
        for lane in self._route_lane_dict.values():
            lane_discrete_path: List[
                StateSE2
            ] = lane.baseline_path.discrete_path
            lane_state_se2_array = np.array(
                [state.array for state in lane_discrete_path], dtype=np.float64
            )
            # calculate nearest state on baseline
            lane_distances = (
                ego_position_array[None, ...] - lane_state_se2_array
            ) ** 2
            lane_distances = lane_distances.sum(axis=-1) ** 0.5
            lane_ego_idx = np.argmin(lane_distances)
            min_distance = lane_distances[lane_ego_idx]
            
            lane_fut_distances = (
                agent_future_state.array[None, ...] - lane_state_se2_array
            ) ** 2
            lane_fut_distances = lane_fut_distances.sum(axis=-1) ** 0.5
            lane_ego_fut_idx = np.argmin(lane_fut_distances)
            # calculate heading error
            current_heading_error = (
                lane_discrete_path[lane_ego_idx].heading - ego_heading
            )
            future_heading_error = (
                lane_discrete_path[lane_ego_fut_idx].heading - ego_heading
            )
            # average heading error to reduce noise
            heading_error = 0.5 * (current_heading_error + future_heading_error)
            heading_error = np.abs(normalize_angle(heading_error))
            if heading_error < np.pi * 0.6 and min_distance < closest_distance:
                closest_distance = min_distance
                starting_lane = lane
                
        return starting_lane
    

    def _get_intersecting_lanes(
        self, ego_state: EgoState
    ) -> Tuple[List[LaneGraphEdgeMapObject], List[float]]:
        """
        Returns on-route lanes and heading errors where vehicle intersects.
        :param ego_state: state of vehicle
        :return: tuple of lists with lane objects and heading errors [rad].
        """
        if self._drivable_area_map is None:
            raise AssertionError("LaneMap: Drivable area map must be initialized first (call step/update before querying).")
        agent_state: StateSE2 = ego_state.rear_axle
        ego_speed = ego_state.dynamic_car_state.speed
        ego_yaw_rate = ego_state.dynamic_car_state.angular_velocity
        agent_future_state = StateSE2(agent_state.x + ego_speed * np.cos(agent_state.heading) * 0.5,
                                     agent_state.y + ego_speed * np.sin(agent_state.heading) * 0.5,
                                     agent_state.heading + ego_yaw_rate * 0.5)
        ego_position_array: npt.NDArray[np.float64] = np.array([agent_state.x, agent_state.y])
        ego_polygon: Polygon = ego_state.car_footprint.geometry
        ego_heading: float = agent_state.heading

        intersecting_lanes = self._drivable_area_map.intersects(ego_polygon)
        
        on_lanes, on_lane_heading_errors = [], []
        for lane_id in intersecting_lanes:
            if lane_id in self._lane_map_dict.keys():
                
                # collect baseline path as array
                lane_object = self._lane_map_dict[lane_id]
                lane_discrete_path: List[
                    StateSE2
                ] = lane_object.baseline_path.discrete_path
                lane_state_se2_array = np.array(
                    [state.array for state in lane_discrete_path], dtype=np.float64
                )
                # calculate nearest state on baseline
                lane_distances = (
                    ego_position_array[None, ...] - lane_state_se2_array
                ) ** 2
                lane_distances = lane_distances.sum(axis=-1) ** 0.5
                lane_ego_idx = np.argmin(lane_distances)
                lane_fut_distances = (
                    agent_future_state.array[None, ...] - lane_state_se2_array
                ) ** 2
                lane_fut_distances = lane_fut_distances.sum(axis=-1) ** 0.5
                lane_ego_fut_idx = np.argmin(lane_fut_distances)
                # calculate heading error
                current_heading_error = np.abs(normalize_angle(
                    lane_discrete_path[lane_ego_idx].heading - ego_heading
                ))
                future_heading_error = np.abs(normalize_angle(
                    lane_discrete_path[lane_ego_fut_idx].heading - ego_heading
                ))
                # average heading error to reduce noise
                heading_error = 0.5 * (current_heading_error + future_heading_error)
                heading_error = np.abs(normalize_angle(heading_error))
                
                if heading_error > np.pi * 0.5:
                    # discard lanes with large heading error
                    continue
                # add lane to candidates
                on_lanes.append(lane_object)
                on_lane_heading_errors.append(heading_error)

        return on_lanes, on_lane_heading_errors
    
    def _update_ref_path(self, ego_state: EgoState, ref_path_resolution: float = 0.5):
        """
        Build the route reference path with NumPy and store the final arrays.

        Outputs:
          self.ref_path: [N, 5] -> x, y, yaw, s, speed_limit
          self.lane_boundaries: [N, 2] -> left_dist, right_dist
        """
        ref_path_np = None
        lane_boundaries_np = None
        lane_speed_limit_np = None
        cumulative_s = 0.0

        for lane in self.ego_route:
            cumulative_s += lane.baseline_path.length
            path_states = lane.baseline_path.discrete_path
            discrete_lane = np.array([st.array for st in path_states], dtype=np.float64)  # [M,3] (x,y,heading)

            speed_limit = lane.speed_limit_mps or 15.0
            lane_speed_limit = np.full(discrete_lane.shape[0], speed_limit, dtype=np.float32)

            road_block = self._route_roadblock_dict[lane.get_roadblock_id()]
            left_bound_dist, right_bound_dist = self._get_lane_boundaries(road_block, lane)
            lane_boundaries = np.stack((left_bound_dist, -right_bound_dist), axis=1)                # [M,2]

            if ref_path_np is None:
                ref_path_np = discrete_lane[:, :2]
                lane_boundaries_np = lane_boundaries
                lane_speed_limit_np = lane_speed_limit
            else:
                ref_path_np = np.concatenate((ref_path_np, discrete_lane[:, :2]), axis=0)
                lane_boundaries_np = np.concatenate((lane_boundaries_np, lane_boundaries), axis=0)
                lane_speed_limit_np = np.concatenate((lane_speed_limit_np, lane_speed_limit), axis=0)

        if self.use_local_coords and ref_path_np is not None:
            ego_yaw = -ego_state.rear_axle.heading
            c, s = np.cos(ego_yaw), np.sin(ego_yaw)
            R = np.array([[c, s], [-s, c]], dtype=np.float64)
            ego_xy = ego_state.rear_axle.point.array
            ref_path_np = (ref_path_np - ego_xy) @ R

        # Drop near-duplicate points to avoid zero-length segments.
        if ref_path_np is not None and ref_path_np.shape[0] >= 2:
            diffs = np.diff(ref_path_np, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            keep_mask = np.ones(ref_path_np.shape[0], dtype=bool)
            keep_mask[1:] = dists >= 1e-3
            if not np.all(keep_mask):
                ref_path_np = ref_path_np[keep_mask]
                lane_boundaries_np = lane_boundaries_np[keep_mask]
                lane_speed_limit_np = lane_speed_limit_np[keep_mask]

        # Ensure at least two points so yaw and arc length are well-defined.
        if ref_path_np.shape[0] < 2:
            ref_path_np = np.vstack([ref_path_np, ref_path_np[-1] + np.array([1e-3, 0.0])])
            lane_boundaries_np = np.vstack([lane_boundaries_np, lane_boundaries_np[-1]])
            lane_speed_limit_np = np.concatenate([lane_speed_limit_np, lane_speed_limit_np[-1:]])

        diffs = np.diff(ref_path_np, axis=0)
        yaw = np.arctan2(diffs[:, 1], diffs[:, 0])
        if np.any(np.isnan(yaw)) or np.any(np.isinf(yaw)):
            valid = ~np.isnan(yaw) & ~np.isinf(yaw)
            idxs = np.arange(yaw.shape[0])
            if np.any(valid):
                yaw = np.interp(idxs, idxs[valid], yaw[valid])
            else:
                yaw = np.zeros_like(yaw)
        yaw = np.concatenate([yaw, yaw[-1:]], axis=0)

        ds = np.linalg.norm(diffs, axis=1)
        s_arr = np.cumsum(ds)
        s_arr = np.concatenate([[0.0], s_arr], axis=0)

        speed_limit_col = lane_speed_limit_np.reshape(-1, 1)
        ref_path_tensor_np = np.concatenate(
            [ref_path_np,
             yaw.reshape(-1, 1),
             s_arr.reshape(-1, 1),
             speed_limit_col],
            axis=1
        ).astype(np.float32)

        self.ref_path = ref_path_tensor_np
        self.lane_boundaries = lane_boundaries_np

    def _get_lane_boundaries(self, roadblock: RoadBlockGraphEdgeMapObject, lane: LaneGraphEdgeMapObject) -> Tuple[LineString, LineString]:
        """
        Returns the left and right lane boundaries as LineString objects.
        :param lane: lane object
        :return: tuple of left and right lane boundaries
        """
        left_bound_set = {}
        right_bound_set = {}

        discrete_lane = np.array([state.array for state in lane.baseline_path.discrete_path])

        left_lane, right_lane = lane.adjacent_edges

        # determine centerlines and lane boundaries if have adjacent lanes
        if ((left_lane is not None) and 
            (left_lane.id in self._lane_map_dict.keys())
            ) :
            self.left_lane = left_lane
            left_bound = left_lane.left_boundary.discrete_path
            left_bound_set[left_lane.id] = left_bound
        else:
            left_bound = lane.left_boundary.discrete_path
            left_bound_set[lane.id] = left_bound
            
        if ((right_lane is not None) and 
            (right_lane.id in self._lane_map_dict.keys())
            ) :
            right_bound = right_lane.right_boundary.discrete_path
            right_bound_set[right_lane.id] = right_bound
        else:
            right_bound = lane.right_boundary.discrete_path
            right_bound_set[lane.id] = right_bound

        for edge in roadblock.interior_edges:

            # skip if edge if already in set
            if edge.id in left_bound_set.keys() or edge.id in right_bound_set.keys():
                continue

            # Get the centerline of the adjacent lane

            adjacent_centerline = np.array([state.array for state in edge.baseline_path.discrete_path])

            # Calculate the vector from the current lane's centerline to the adjacent lane's centerline
            vector_to_adjacent = adjacent_centerline[adjacent_centerline.shape[0]//2] - discrete_lane[discrete_lane.shape[0]//2]

            # Calculate the direction vector of the current lane
            current_direction = discrete_lane[discrete_lane.shape[0]//2] - discrete_lane[discrete_lane.shape[0]//2 - 1]

            # Use the cross product to determine the relative position
            cross_product = np.cross(current_direction, vector_to_adjacent)

            if cross_product > 0:
                # Adjacent lane is on the left
                left_bound = edge.left_boundary.discrete_path
                left_bound_set[edge.id] = left_bound
            else:
                # Adjacent lane is on the right
                right_bound = edge.right_boundary.discrete_path
                right_bound_set[edge.id] = right_bound 

        left_bound_dist = _calculate_widest_bound(left_bound_set, discrete_lane)
        right_bound_dist = _calculate_widest_bound(right_bound_set, discrete_lane)

        return (left_bound_dist.astype(np.float32),
                right_bound_dist.astype(np.float32),
                )
    
    def get_road_feature(self, ego_state: EgoState, 
                         tl_data: List[TrafficLightStatusData],
                         roi_map: ROIMap = None,
                         max_lane_num: int = 70,
                         max_non_lane_num: int = 10,
                         num_points = 20,  # Target centerline point count.
                         num_edge = 20,     # Target polygon vertex count.
                         lane_preview_dist: float = 10.0
                         ) -> Dict:
        """
        Extract nearby road elements and transform them into the ego-centric frame.
        :param ego_state: Current ego state.
        :param tl_data: Traffic light status records.
        :param roi_map: Optional ROI filter.
        :return: Road feature dictionary in ego coordinates.
        """

        signal_dict = {}
        for data in tl_data:
            lane_id = str(data.lane_connector_id)
            signal_dict[lane_id] = data.status
        
        position = ego_state.rear_axle.point
        elements = self._map_api.get_proximal_map_objects(
            position, self.map_radius, [
                SemanticMapLayer.STOP_LINE,
                SemanticMapLayer.CROSSWALK,
                SemanticMapLayer.ROADBLOCK,
                SemanticMapLayer.ROADBLOCK_CONNECTOR,
                SemanticMapLayer.INTERSECTION,
                SemanticMapLayer.CARPARK_AREA,
            ]
        )

        ego_angle = -ego_state.rear_axle.heading
        rotation_matrix = np.array([
            [np.cos(ego_angle), np.sin(ego_angle)],
            [-np.sin(ego_angle), np.cos(ego_angle)]
        ])
        ego_position = ego_state.rear_axle.point.array

        def empty_road_data() -> Dict[str, List]:
            return {
                'center_line': [],
                'road_geometry': [],
                'road_type': [],
                'road_traffic_light': [],
                'road_speed_limit': [],
                'id': [],
                'distance': [],
            }

        road_data = empty_road_data()
        lane_data = empty_road_data()
        element_data = empty_road_data()

        
        if self.use_ref_path:
            ref_path_np = self.ref_path
            ref_point_preview_point = None
            dist = np.linalg.norm(ref_path_np[:, :2], axis=1)
            ego_current_ref_idx = np.argmin(dist)
            ego_current_s = dist[ego_current_ref_idx]
            if ref_path_np.size and ref_path_np.shape[1] >= 4:
                s_arr = ref_path_np[:, 3] - ego_current_s
                if s_arr[-1] >= lane_preview_dist:
                    idx = np.searchsorted(s_arr, lane_preview_dist)
                    p = ref_path_np[idx, :2]
                else:
                    p = ref_path_np[-1, :2]
                ref_point_preview_point = p
        else:
            ref_point_preview_point = None
        
        for layer in [SemanticMapLayer.ROADBLOCK, 
                      SemanticMapLayer.ROADBLOCK_CONNECTOR]:
            for rbk in elements[layer]:
                for lane in rbk.interior_edges:
                    lane_id = lane.id
                    polygon_obj = lane.polygon
                    polygon_coords = np.array(polygon_obj.exterior.coords)
                    local_polygon_coords = (polygon_coords - ego_position) @ rotation_matrix
                    polygon_obj = Polygon(local_polygon_coords)

                    if roi_map is not None:
                        if not roi_map.polygon_in_roi(polygon_obj):
                            continue

                    lane_id = str(lane.id)
                    if lane_id in signal_dict.keys():
                        tl_signal = signal_dict[lane_id]
                    else:
                        tl_signal = TrafficLightStatusType.GREEN


                    discrete_path = np.array([state.array for state in lane.baseline_path.discrete_path])
                    
                    if len(discrete_path) > num_points:
                        discrete_path = resample_discrete_path(lane.baseline_path.discrete_path, num_points)
                    elif len(discrete_path) < num_points:
                        last_point = discrete_path[-1:]
                        repeat_times = num_points - len(discrete_path)
                        discrete_path = np.concatenate([discrete_path, np.tile(last_point, (repeat_times, 1))])
                    
                    
                    # Simplify polygon until it fits the fixed vertex budget.
                    tolerance = 1.0
                    while True:
                        simplified_polygon = polygon_obj.simplify(tolerance=tolerance, preserve_topology=True)
                        polygon_coords = np.array(simplified_polygon.exterior.coords)
                        if len(polygon_coords) <= num_edge or tolerance > 5.0:
                            break
                        tolerance += 0.5
                    if len(polygon_coords) > num_edge:
                        polygon_coords = polygon_coords[:num_edge]
                    elif len(polygon_coords) < num_edge:
                        last_point = polygon_coords[-1:]
                        repeat_times = num_edge - len(polygon_coords)
                        polygon_coords = np.concatenate([polygon_coords, np.tile(last_point, (repeat_times, 1))])
                    
                    local_path = (discrete_path - ego_position) @ rotation_matrix
                    
                    # Append yaw to the local centerline.
                    diff = np.diff(local_path, axis=0)
                    yaw = np.arctan2(diff[:, 1], diff[:, 0])
                    yaw = np.concatenate([yaw, yaw[-1:]], axis=0)
                    if np.any(np.isnan(yaw)) or np.any(np.isinf(yaw)):
                        valid = ~np.isnan(yaw) & ~np.isinf(yaw)
                        indices = np.arange(len(yaw))
                        if np.any(valid):
                            yaw = np.interp(indices, indices[valid], yaw[valid])
                        else:
                            yaw = np.zeros_like(yaw)
                    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
                    local_path = np.concatenate([local_path, yaw[:, None]], axis=1)

                    if layer == SemanticMapLayer.ROADBLOCK:
                        lane_type = RoadType.LANE
                    elif layer == SemanticMapLayer.ROADBLOCK_CONNECTOR:
                        lane_type = RoadType.CONNECTOR

                    if ref_point_preview_point is not None:
                        lane_center = local_path[:, :2]
                        distance_to_ref = np.linalg.norm(lane_center - ref_point_preview_point, axis=-1).min()
                        distance_to_ego = np.linalg.norm(lane_center, axis=-1).min()
                        distance = np.mean([distance_to_ref, distance_to_ego])
                    else:
                        distance = np.linalg.norm(local_path[:, :2], axis=-1).min()
                    
                    lane_data['id'].append(lane_id)
                    lane_data['distance'].append(distance)
                    lane_data['center_line'].append(local_path)
                    lane_data['road_geometry'].append(polygon_coords)
                    lane_data['road_type'].append(lane_type.value)
                    lane_data['road_traffic_light'].append(tl_signal.value + 1) # +1 to avoid zero index
                    lane_data['road_speed_limit'].append(lane.speed_limit_mps or 15)


        for layer in [# SemanticMapLayer.STOP_LINE,
                        SemanticMapLayer.CROSSWALK,
                        SemanticMapLayer.INTERSECTION,
                        SemanticMapLayer.CARPARK_AREA,
                        # SemanticMapLayer.ROADBLOCK, 
                        # SemanticMapLayer.ROADBLOCK_CONNECTOR,
                        # SemanticMapLayer.WALKWAYS,
                        ]:
            # if layer == SemanticMapLayer.DRIVABLE_AREA:
            #     print(f"Processing layer {layer.name} with {len(elements[layer])} elements (drivable area may be large).")
            for element in elements[layer]:
                center_line_void = np.zeros((num_points, 3))
                element_id = element.id
                polygon_obj = element.polygon
                polygon_coords = np.array(polygon_obj.exterior.coords)
                local_polyon_coords = (polygon_coords - ego_position) @ rotation_matrix
                polygon_obj = Polygon(local_polyon_coords)

                if roi_map is not None:
                    if not roi_map.polygon_in_roi(polygon_obj):
                        continue
                    
                # Simplify polygon until it fits the fixed vertex budget.
                tolerance = 0.5
                while True:
                    simplified_polygon = polygon_obj.simplify(tolerance=tolerance, preserve_topology=True)
                    polygon_coords = np.array(simplified_polygon.exterior.coords)
                    if len(polygon_coords) <= num_edge or tolerance > 5.0:
                        break
                    tolerance += 0.1
                if len(polygon_coords) > num_edge:
                    polygon_coords = polygon_coords[:num_edge]
                elif len(polygon_coords) < num_edge:
                    last_point = polygon_coords[-1:]
                    repeat_times = num_edge - len(polygon_coords)
                    polygon_coords = np.concatenate([polygon_coords, np.tile(last_point, (repeat_times, 1))])

                if layer == SemanticMapLayer.STOP_LINE:
                    lane_type = RoadType.STOP_LINE
                elif layer == SemanticMapLayer.CROSSWALK:
                    lane_type = RoadType.CROSSWALK
                elif layer == SemanticMapLayer.INTERSECTION:
                    lane_type = RoadType.INTERSECTION
                elif layer == SemanticMapLayer.CARPARK_AREA:
                    lane_type = RoadType.CARPARK
                elif layer == SemanticMapLayer.ROADBLOCK:
                    lane_type = RoadType.ROADBLOCK
                elif layer == SemanticMapLayer.ROADBLOCK_CONNECTOR:
                    lane_type = RoadType.ROADBLOCK_CONNECTOR
                elif layer == SemanticMapLayer.WALKWAYS:
                    lane_type = RoadType.WALKWAYS
                # elif lane_type == SemanticMapLayer.DRIVABLE_AREA:
                #     lane_type = RoadType.DRIVABLE_AREA
                else:
                    lane_type = RoadType.INVALID
                    
                element_centroid = polygon_obj.centroid
                element_center = np.array([element_centroid.x, element_centroid.y], dtype=np.float32)

                if ref_point_preview_point is not None:
                    distance_to_ref = np.linalg.norm(element_center - ref_point_preview_point)
                    distance_to_ego = np.linalg.norm(element_center)
                    distance = np.mean([distance_to_ref, distance_to_ego])
                else:
                    distance = np.linalg.norm(element_center)

                element_data['id'].append(element_id)
                element_data['distance'].append(distance)
                element_data['center_line'].append(center_line_void)
                element_data['road_geometry'].append(polygon_coords)
                element_data['road_type'].append(lane_type.value)
                element_data['road_traffic_light'].append(0)
                element_data['road_speed_limit'].append(-1.0)

        # Keep the closest lane and non-lane elements under the feature budget.
        lane_distances = np.array(lane_data['distance'])
        if len(lane_distances) > max_lane_num:
            closest_indices = np.argsort(lane_distances)[:max_lane_num]
            lane_data = {key: [value[i] for i in closest_indices] for key, value in lane_data.items()}
        element_distances = np.array(element_data['distance'])
        if len(element_distances) > max_non_lane_num:
            closest_indices = np.argsort(element_distances)[:max_non_lane_num]
            element_data = {key: [value[i] for i in closest_indices] for key, value in element_data.items()}
        
        for key in road_data.keys():
            road_data[key].extend(lane_data[key])
            road_data[key].extend(element_data[key])

        # Convert lists in road_data to np.ndarray objects
        for key in road_data:
            if key in ['center_line', 'road_geometry', 'road_speed_limit']:
                road_data[key] = np.array(road_data[key], dtype=np.float32)
            elif key in ['road_type', 'road_traffic_light']:
                road_data[key] = np.array(road_data[key], dtype=np.int32)
            else:
                road_data[key] = np.array(road_data[key])

        # Delete id and distance from road_data 
        del road_data['id']
        del road_data['distance']

        # Warn on invalid feature values before returning.
        for key, arr in road_data.items():
            if isinstance(arr, np.ndarray):
                nan_indices = np.argwhere(np.isnan(arr))
                for idx in nan_indices:
                    logging.warning(f"road_data['{key}'] contains NaN at index {tuple(idx)}")
                inf_indices = np.argwhere(np.isinf(arr))
                for idx in inf_indices:
                    logging.warning(f"road_data['{key}'] contains Inf at index {tuple(idx)}")
        
        return road_data

    def get_route_feature(self, 
                          ego_state: EgoState,
                          tl_data: List[TrafficLightStatusData],
                          roi_map: ROIMap = None,
                          max_route_num: int = 10,
                          num_edge = 20     # Target polygon vertex count.
                          ) -> Dict:

        ego_angle = -ego_state.rear_axle.heading
        rotation_matrix = np.array([
            [np.cos(ego_angle), np.sin(ego_angle)],
            [-np.sin(ego_angle), np.cos(ego_angle)]
        ])
        ego_position = ego_state.rear_axle.point.array

        route_data = {
            'route_geometry': [],
            'id': [],
            'distance': [],
        }
        start_idx = 0
        for idx, (rb_id, rbk) in enumerate(self._route_roadblock_dict.items()):
            polygon_obj = rbk.polygon
            polygon_coords = np.array(polygon_obj.exterior.coords)
            local_polyon_coords = (polygon_coords - ego_position) @ rotation_matrix
            polygon_obj = Polygon(local_polyon_coords)
            ego_point = Point(0.0, 0.0)
            distance = polygon_obj.distance(ego_point)

            if start_idx == 0:
                if polygon_obj.contains(ego_point):
                    start_idx = idx

            if roi_map is not None:
                if not roi_map.polygon_in_roi(polygon_obj):
                    continue
                
            # Simplify polygon until it fits the fixed vertex budget.
            tolerance = 0.5
            while True:
                simplified_polygon = polygon_obj.simplify(tolerance=tolerance, preserve_topology=True)
                polygon_coords = np.array(simplified_polygon.exterior.coords)
                if len(polygon_coords) <= num_edge or tolerance > 5.0:
                    break
                tolerance += 0.1
            if len(polygon_coords) > num_edge:
                polygon_coords = polygon_coords[:num_edge]
            elif len(polygon_coords) < num_edge:
                last_point = polygon_coords[-1:]
                repeat_times = num_edge - len(polygon_coords)
                polygon_coords = np.concatenate([polygon_coords, np.tile(last_point, (repeat_times, 1))])

            route_data['route_geometry'].append(polygon_coords)
            route_data['id'].append(rb_id)
            route_data['distance'].append(distance)


        # screen based on distance and max_route_num
        if len(route_data['distance']) > max_route_num:
            clip_start_idx = max(0, start_idx - 1)
            clip_end_idx = min(clip_start_idx + max_route_num, len(route_data['distance']))
            route_data = {key: value[clip_start_idx:clip_end_idx] for key, value in route_data.items()}

        # Convert lists in route_data to np.ndarray objects
        for key in route_data:
            route_data[key] = np.array(route_data[key])
        
        # Delete id and distance from route_data 
        del route_data['id']
        del route_data['distance']

        # save memory by converting to float32
        route_data['route_geometry'] = route_data['route_geometry'].astype(np.float32, copy=False)
        
        return route_data

    def get_ref_path_feature(self, 
                             ego_state: EgoState = None,
                             ref_path_num_points: int = 200,
                             ref_path_lookahead: float = 150.0,
                             ref_path_lookbehind: float = 50.0) -> np.ndarray:
        """
        Returns the reference path feature.
        :return: reference path feature tensor
        """

        if self.ref_path is None:
            AssertionError("LaneMap: Reference path tensor is not initialized!")
        
        ref_path_np = self.ref_path
        ref_path_bounds = self.lane_boundaries
        ref_path_centers = ref_path_np[:, :2]
        
        yaw = ref_path_np[:, 2]
        if np.any(np.isnan(yaw)) or np.any(np.isinf(yaw)):
            valid = ~np.isnan(yaw) & ~np.isinf(yaw)
            indices = np.arange(len(yaw))
            if np.any(valid):
                yaw = np.interp(indices, indices[valid], yaw[valid])
            else:
                yaw = np.zeros_like(yaw)
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        ref_speed_limit = ref_path_np[:, 4].reshape(-1, 1)
        ref_feat = np.concatenate((ref_path_centers, yaw[:, None], ref_path_bounds, ref_speed_limit), axis=1)

        # Clip to the configured lookbehind/lookahead window around the ego projection.
        dist = np.linalg.norm(ref_path_np[:, :2], axis=1)
        ego_current_ref_idx = np.argmin(dist)
        ego_current_s = ref_path_np[ego_current_ref_idx, 3]
        s_arr = ref_path_np[:, 3] - ego_current_s
        if s_arr[-1] > ref_path_lookahead or s_arr[0] < -ref_path_lookbehind:
            idx = np.searchsorted(s_arr, ref_path_lookahead)
            start_idx = np.searchsorted(s_arr, -ref_path_lookbehind)
            ref_feat = ref_feat[start_idx:idx, :]
            s_arr = s_arr[start_idx:idx]

        # Resample to a fixed number of reference path points.
        if ref_feat.shape[0] != ref_path_num_points:
            s_uniform = np.linspace(s_arr[0], s_arr[-1], ref_path_num_points)
            ref_feat_resampled = np.zeros((ref_path_num_points, ref_feat.shape[1]), dtype=ref_feat.dtype)
            for i in range(ref_feat.shape[1]):
                ref_feat_resampled[:, i] = np.interp(s_uniform, s_arr, ref_feat[:, i])
            ref_feat = ref_feat_resampled
        
        nan_indices = np.argwhere(np.isnan(ref_feat))
        assert len(nan_indices) == 0, "ref_feat contains NaN values"
        for idx in nan_indices:
            logging.warning(f"ref_feat contains NaN at index {tuple(idx)}")
        inf_indices = np.argwhere(np.isinf(ref_feat))
        for idx in inf_indices:
            logging.warning(f"ref_feat contains Inf at index {tuple(idx)}")
        
        return ref_feat.astype(np.float32)


def _calculate_widest_bound(bound_set, discrete_lane):
    num_points = discrete_lane.shape[0]
    widest_bound_dist = np.zeros(num_points)
    for bound in bound_set.values():
        upsampled_bound = resample_discrete_path(bound, num_points)
        distances = np.linalg.norm(discrete_lane[:, None, :] - upsampled_bound[None, :, :], axis=-1)
        min_distances = np.min(distances, axis=1)
        widest_bound_dist = np.maximum(widest_bound_dist, min_distances)
    return widest_bound_dist
