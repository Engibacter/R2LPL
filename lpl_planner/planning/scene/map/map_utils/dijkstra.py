from typing import Dict, List, Optional, Tuple

import numpy as np
from nuplan.common.maps.abstract_map_objects import (
    LaneGraphEdgeMapObject,
    RoadBlockGraphEdgeMapObject,
)


class Dijkstra:
    """
    A class that performs dijkstra's shortest path. The class operates on lane level graph search.
    The goal condition is specified to be if the lane can be found at the target roadblock or roadblock connector.
    """

    def __init__(
        self, 
        start_edge: LaneGraphEdgeMapObject, 
        candidate_lane_edge_ids: List[str],
        route_roadblock_ids: List[str],
        curvature_weight: float = 5.0,
        heading_align_weight: float = 2.0,
    ):
        """
        Constructor for the Dijkstra class.
        :param start_edge: The starting edge for the search
        :param candidate_lane_edge_ids: The candidates lane ids that can be included in the search.
        """
        self._queue = list([start_edge])
        self._parent: Dict[str, Optional[LaneGraphEdgeMapObject]] = dict()
        self._candidate_lane_edge_ids = candidate_lane_edge_ids
        self._route_roadblock_ids = route_roadblock_ids

        self._curvature_weight = float(curvature_weight)
        self._heading_align_weight = float(heading_align_weight)
        self._metrics_cache: Dict[str, Tuple[float, float, float]] = {}
        self._heading_cache: Dict[str, Tuple[float, float]] = {}

    def search(
        self, target_roadblock: RoadBlockGraphEdgeMapObject
    ) -> Tuple[List[LaneGraphEdgeMapObject], bool]:
        """
        Performs dijkstra's shortest path to find a route to the target roadblock.
        :param target_roadblock: The target roadblock the path should end at.
        :return:
            - A route starting from the given start edge
            - A bool indicating if the route is successfully found. Successful means that there exists a path
              from the start edge to an edge contained in the end roadblock.
              If unsuccessful the shortest deepest path is returned.
        """
        start_edge = self._queue[0]

        # Initial search states
        path_found: bool = False
        end_edge: LaneGraphEdgeMapObject = start_edge

        self._parent[start_edge.id] = None
        self._frontier = [start_edge.id]
        self._dist = [1]
        self._depth = [1]

        self._expanded = []
        self._expanded_id = []
        self._expanded_dist = []
        self._expanded_depth = []

        while len(self._queue) > 0:
            dist, idx = min((val, idx) for (idx, val) in enumerate(self._dist))
            current_edge = self._queue[idx]
            current_depth = self._depth[idx]

            del self._dist[idx], self._queue[idx], self._frontier[idx], self._depth[idx]

            if self._check_goal_condition(current_edge, target_roadblock):
                end_edge = current_edge
                path_found = True
                break

            self._expanded.append(current_edge)
            self._expanded_id.append(current_edge.id)
            self._expanded_dist.append(dist)
            self._expanded_depth.append(current_depth)

            # Populate queue
            # if len(current_edge.outgoing_edges) > 0:
            #     print(f"Expanding edge {current_edge.id} at depth {current_depth} with {len(current_edge.outgoing_edges)} outgoing edges.")
            #     print(f"Outgoing edge IDs: {[edge.id for edge in current_edge.outgoing_edges]}")
            for next_edge in current_edge.outgoing_edges:
                if next_edge.id not in self._candidate_lane_edge_ids:
                    if any([next_next_edge.id in self._candidate_lane_edge_ids for next_next_edge in next_edge.outgoing_edges]):
                        self._candidate_lane_edge_ids.append(next_edge.id)
                    else:
                        # print(f"Skipping edge {next_edge.id} as it is not in candidate lane edge ids.")
                        continue

                alt = dist + self._edge_cost(next_edge, prev_edge=current_edge)
                if (
                    next_edge.id not in self._expanded_id
                    and next_edge.id not in self._frontier
                ):
                    self._parent[next_edge.id] = current_edge
                    self._queue.append(next_edge)
                    self._frontier.append(next_edge.id)
                    self._dist.append(alt)
                    self._depth.append(current_depth + 1)
                    end_edge = next_edge

                elif next_edge.id in self._frontier:
                    next_edge_idx = self._frontier.index(next_edge.id)
                    current_cost = self._dist[next_edge_idx]
                    if alt < current_cost:
                        self._parent[next_edge.id] = current_edge
                        self._dist[next_edge_idx] = alt
                        self._depth[next_edge_idx] = current_depth + 1

        if not path_found:
            # filter max depth
            max_depth = max(self._expanded_depth)
            idx_max_depth = list(
                np.where(np.array(self._expanded_depth) == max_depth)[0]
            )
            dist_at_max_depth = [self._expanded_dist[i] for i in idx_max_depth]

            dist, _idx = min((val, idx) for (idx, val) in enumerate(dist_at_max_depth))
            end_edge = self._expanded[idx_max_depth[_idx]]

        return self._construct_path(end_edge), path_found


    def _edge_cost(
        self, lane: LaneGraphEdgeMapObject, prev_edge: Optional[LaneGraphEdgeMapObject] = None
    ) -> float:
        """
        Combined edge cost.

        The current route-following cost primarily prefers the next roadblock in
        the route sequence. Geometry caches are kept for optional curvature or
        heading-alignment terms.
        """
        length = lane.baseline_path.length
        start_heading = lane.baseline_path.discrete_path[0].heading
        end_heading = lane.baseline_path.discrete_path[-1].heading
        total_turn = abs(self._wrap_angle(end_heading - start_heading))
        lane_roadblock_in_route_idx = np.argmax(
            np.array(self._route_roadblock_ids) == lane.get_roadblock_id()
        )
        prev_edge_roadblock_in_route_idx = np.argmax(
            np.array(self._route_roadblock_ids) == prev_edge.get_roadblock_id()
        ) if prev_edge is not None else 0
        route_seq_cost = lane_roadblock_in_route_idx - prev_edge_roadblock_in_route_idx
        route_seq_cost = abs(route_seq_cost) * 1000.0
        # cost = total_turn/length + length + route_seq_cost
        cost = length + route_seq_cost

        return float(cost)
    
    def _get_edge_length_and_turn(self, lane: LaneGraphEdgeMapObject) -> Tuple[float, float]:
        """Compute and cache polyline length and total heading change."""
        if lane.id in self._metrics_cache:
            length, total_turn = self._metrics_cache[lane.id]
            return length, total_turn

        xy = np.array([s.array for s in lane.baseline_path.discrete_path], dtype=float)  # shape [N, >=2]
        if xy.ndim != 2 or xy.shape[0] < 2:
            self._metrics_cache[lane.id] = (0.0, 0.0)
            return 0.0, 0.0

        if xy.shape[1] >= 2:
            pts = xy[:, :2]
        else:
            pts = xy

        d = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(d, axis=1)
        length = float(np.sum(seg_len))

        headings = np.arctan2(d[:, 1], d[:, 0])  # size N-1
        dtheta = np.diff(np.unwrap(headings))
        total_turn = float(np.sum(np.abs(dtheta)))

        self._metrics_cache[lane.id] = (length, total_turn)
        return length, total_turn

    def _get_edge_headings(self, lane: LaneGraphEdgeMapObject) -> Tuple[float, float]:
        """Return start and end headings estimated from the first and last segments."""
        if lane.id in self._heading_cache:
            return self._heading_cache[lane.id]

        xy = np.array([s.array for s in lane.baseline_path.discrete_path], dtype=float)
        if xy.shape[1] >= 2:
            pts = xy[:, :2]
        else:
            pts = xy

        if pts.shape[0] >= 2:
            v0 = pts[1] - pts[0]
            v1 = pts[-1] - pts[-2]
            h0 = float(np.arctan2(v0[1], v0[0]))
            h1 = float(np.arctan2(v1[1], v1[0]))
        else:
            h0 = h1 = 0.0

        self._heading_cache[lane.id] = (h0, h1)
        return h0, h1

    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Wrap an angle to [-pi, pi]."""
        return (a + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _check_end_condition(depth: int, target_depth: int) -> bool:
        """
        Check if the search should end regardless if the goal condition is met.
        :param depth: The current depth to check.
        :param target_depth: The target depth to check against.
        :return: True if:
            - The current depth exceeds the target depth.
        """
        return depth > target_depth

    @staticmethod
    def _check_goal_condition(
        current_edge: LaneGraphEdgeMapObject,
        target_roadblock: RoadBlockGraphEdgeMapObject,
    ) -> bool:
        """
        Check if the current edge is at the target roadblock at the given depth.
        :param current_edge: The edge to check.
        :param target_roadblock: The target roadblock the edge should be contained in.
        :return: whether the current edge is in the target roadblock
        """
        return current_edge.get_roadblock_id() == target_roadblock.id

    def _construct_path(
        self, end_edge: LaneGraphEdgeMapObject
    ) -> List[LaneGraphEdgeMapObject]:
        """
        :param end_edge: The end edge to start back propagating back to the start edge.
        :param depth: The depth of the target edge.
        :return: The constructed path as a list of LaneGraphEdgeMapObject
        """
        path = [end_edge]
        while self._parent[end_edge.id] is not None:
            node = self._parent[end_edge.id]
            path.append(node)
            end_edge = node
        path.reverse()

        return path
