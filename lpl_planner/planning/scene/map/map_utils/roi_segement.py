import numpy as np
from shapely.geometry import Polygon
from rtree import index


class ROIMap:
    def __init__(self, ref_path: np.ndarray, s_min: float, s_max: float, width: float, extrapolate_back: bool = True):
        self.ref_path = ref_path
        self.polygon = build_ref_path_roi(ref_path, s_min, s_max, width, extrapolate_back)
        self._tree = None


    def polygons_in_roi(self, polygons: list[Polygon]):
        """
        Return indices of polygons that intersect the ROI polygon.
        :param polygons: list of shapely Polygon
        :return: list of indices of polygons that are in the ROI
        """
        idx = index.Index()
        for i, elem in enumerate(polygons):
            idx.insert(i, elem.bounds)
        self._tree = idx
        roi_candidates = list(idx.intersection(self.polygon.bounds))
        in_roi = [i for i in roi_candidates if self.polygon.intersects(polygons[i])]

        return in_roi
    
    def polygon_in_roi(self, poly):
        """
        Check whether a polygon intersects the ROI polygon.
        :param poly: shapely Polygon
        :return: bool
        """
        return self.polygon.intersects(poly)


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
        ref_path = np.vstack([virtual_points, ref_path[:, :2]])
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
