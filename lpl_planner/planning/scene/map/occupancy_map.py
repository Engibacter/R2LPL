from typing import List, Dict, Union
import numpy.typing as npt
import numpy as np
import shapely

from shapely.strtree import STRtree
from shapely.geometry import LineString, Polygon, LinearRing, MultiPoint
import shapely.vectorized
Geometry = Union[Polygon, LineString]

class OccupancyMap:
    """Occupancy map class from PDM, based on shapely's str-tree."""

    def __init__(
        self,
        tokens: List[str],
        geometries: npt.NDArray[np.object_],
        node_capacity: int = 10,
    ):
        """
        Constructor of PDMOccupancyMap
        :param tokens: list of tracked tokens
        :param geometries: list/array of polygons
        :param node_capacity: max number of child nodes in str-tree, defaults to 10
        """
        assert len(tokens) == len(
            geometries
        ), f"PDMOccupancyMap: Tokens/Geometries ({len(tokens)}/{len(geometries)}) have unequal length!"

        self._tokens: List[str] = tokens
        self._token_to_idx: Dict[str, int] = {
            token: idx for idx, token in enumerate(tokens)
        }

        self._geometries = geometries
        self._node_capacity = node_capacity
        self._str_tree = STRtree(self._geometries, node_capacity)

    def __getitem__(self, token) -> Geometry:
        """
        Retrieves geometry of token.
        :param token: geometry identifier
        :return: Geometry of token
        """
        return self._geometries[self._token_to_idx[token]]

    def __len__(self) -> int:
        """
        Number of geometries in the occupancy map
        :return: int
        """
        return len(self._tokens)

    @property
    def tokens(self) -> List[str]:
        """
        Getter for track tokens in occupancy map
        :return: list of strings
        """
        return self._tokens

    @property
    def token_to_idx(self) -> Dict[str, int]:
        """
        Getter for track tokens in occupancy map
        :return: dictionary of tokens and indices
        """
        return self._token_to_idx

    def intersects(self, geometry: Geometry) -> List[str]:
        """
        Searches for intersecting geometries in the occupancy map
        :param geometry: geometries to query
        :return: list of tokens for intersecting geometries
        """
        indices = self.query(geometry, predicate="intersects")
        return [self._tokens[idx] for idx in indices]
    
    def contains_polygon(self, polygon: Polygon) -> bool:
        """
        Checks if the occupancy map contains the input polygon
        :param polygon: input polygon
        :return: boolean
        """
        indices = self.query(polygon, predicate="contains")
        return len(indices) > 0

    def query(self, geometry: Geometry, predicate=None):
        """
        Function to directly calls shapely's query function on str-tree
        :param geometry: geometries to query
        :param predicate: see shapely, defaults to None
        :return: query output
        """
        return self._str_tree.query(geometry, predicate=predicate)
    
    def points_in_polygons(
        self,
        points: npt.NDArray[np.float64],
        prefilter_with_point_obb: bool = False,
    ) -> npt.NDArray[np.bool_]:
        """
        Determines wether input-points are in polygons of the occupancy map
        :param points: input-points
        :param prefilter_with_point_obb: if true, first query candidate polygons whose
            geometry intersects the oriented bounding box of all query points, then run
            exact contains checks only on those candidates
        :return: boolean array of shape (polygons, input-points)
        """
        flat_points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        output = np.zeros((len(self._geometries), len(flat_points)), dtype=bool)
        if len(self._geometries) == 0 or len(flat_points) == 0:
            return output

        polygon_indices = np.arange(len(self._geometries), dtype=np.int64)
        if prefilter_with_point_obb and len(self._geometries) > 0:
            point_obb = MultiPoint(flat_points).minimum_rotated_rectangle
            polygon_indices = np.asarray(
                self.query(point_obb, predicate="intersects"),
                dtype=np.int64,
            )
            if polygon_indices.size == 0:
                return output

        for polygon_idx in polygon_indices:
            polygon = self._geometries[int(polygon_idx)]
            output[int(polygon_idx)] = shapely.contains_xy(
                polygon,
                flat_points[:, 0],
                flat_points[:, 1],
            )

        return output
    
    def polygons_for_plot(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        polygons_xy = []
        for polygon in self._geometries:
            x,y = polygon.exterior.xy
            polygon_xy = [x,y]
            polygons_xy.append(polygon_xy)

        return polygons_xy