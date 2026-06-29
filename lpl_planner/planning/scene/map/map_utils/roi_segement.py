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
        检查哪些多边形完全或部分处于ROI polygon内部
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
        检查指定多边形是否完全处于ROI polygon内部，或部分处于ROI polygon内部
        :param poly: shapely Polygon
        :return: bool
        """
        return self.polygon.intersects(poly)


def build_ref_path_roi(ref_path: np.ndarray, s_min: float, s_max: float, width: float, extrapolate_back: bool = True):
    """
    构建以ref_path为中心，区间[s_min, s_max]，宽度为width的多边形ROI。
    若s_min<0且ref_path起点不足，则自动向后虚拟延申。
    :param ref_path: [N, 2] or [N, 3] (x, y, ...)
    :param s_min: float, ROI起点相对ref_path起点的距离（可为负）
    :param s_max: float, ROI终点相对ref_path起点的距离
    :param width: float, 区域宽度（左右各width/2）
    :param extrapolate_back: 是否自动向后虚拟延申
    :return: shapely Polygon
    """
    # 计算累积弧长
    diff = np.diff(ref_path[:, :2], axis=0)
    ds = np.linalg.norm(diff, axis=1)
    s = np.concatenate([[0], np.cumsum(ds)])

    # 若s_min<0且ref_path起点不够，自动向后延申
    if extrapolate_back and s_min < 0:
        # 估算采样间距
        step = ds[0] if len(ds) > 0 else 1.0
        n_extrapolate = int(np.ceil(abs(s_min) / (step + 1e-8)))
        # 用起点切线方向反向延申
        tangent = ref_path[1, :2] - ref_path[0, :2]
        tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
        virtual_points = [ref_path[0, :2] - tangent * (i * step) for i in range(n_extrapolate, 0, -1)]
        # 拼接
        ref_path = np.vstack([virtual_points, ref_path[:, :2]])
        # 重新计算弧长
        diff = np.diff(ref_path[:, :2], axis=0)
        ds = np.linalg.norm(diff, axis=1)
        s = np.concatenate([[0], np.cumsum(ds)])
        s_min = 0  # 虚拟段已补齐，ROI起点重置为0
        s_max = s_max + abs(s_min)  # ROI终点相对新起点偏移

    # 选取区间
    idx = np.where((s >= s_min) & (s <= s_max))[0]
    if len(idx) < 2:
        raise ValueError("ref_path 区间太短")
    ref_seg = ref_path[idx, :2]

    # 计算法向量
    tangents = np.diff(ref_seg, axis=0)
    tangents = np.vstack([tangents, tangents[-1]])  # 补最后一个
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / (norms + 1e-8)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)  # 逆时针90度

    # 左右平移
    left = ref_seg + normals * (width / 2)
    right = ref_seg - normals * (width / 2)
    roi_poly = np.vstack([left, right[::-1]])  # 闭合多边形

    return Polygon(roi_poly)