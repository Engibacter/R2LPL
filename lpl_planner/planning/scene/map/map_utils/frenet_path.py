import numpy as np
from scipy.spatial import KDTree

class FrenetPath:
    def __init__(self,
                 ref_path: np.ndarray,
                 lookahead_distance: float = 200,
                 lookback_distance: float = -100,
                 angle_penalty: float = 2.0):
        """初始化FrenetPath对象，预计算参考路径的相关数据以加速转换
        ref_path: shape (N, 4) 包含x,y,theta,s坐标
        """
        start_idx = np.argmin(np.linalg.norm(ref_path[:, :2], axis=1))
        ref_path[:, 3] = ref_path[:, 3] - ref_path[start_idx, 3] # 重新计算s坐标，使起点为0
        if lookahead_distance is not None and ref_path[-1, 3] > lookahead_distance:
            idx = np.searchsorted(ref_path[:, 3], lookahead_distance)
            ref_path = ref_path[:idx + 1]
        if lookback_distance is not None and ref_path[0, 3] < lookback_distance:
            idx = np.searchsorted(ref_path[:, 3], lookback_distance)
            ref_path = ref_path[idx:]
        # 基于s进行等间距插值 ds=0.1，得到均匀的(x,y,theta,s)
        s_orig = ref_path[:, 3]
        if len(s_orig) >= 2:
            ds = 0.1
            s_uniform = np.arange(s_orig[0], s_orig[-1] + ds * 0.5, ds)
            x_uniform = np.interp(s_uniform, s_orig, ref_path[:, 0])
            y_uniform = np.interp(s_uniform, s_orig, ref_path[:, 1])
            theta_unwrapped = np.unwrap(ref_path[:, 2])
            theta_uniform = np.interp(s_uniform, s_orig, theta_unwrapped)
            theta_uniform = (theta_uniform + np.pi) % (2 * np.pi) - np.pi
            speed_limit = np.interp(s_uniform, s_orig, ref_path[:, 4]) 
            ref_path = np.column_stack((x_uniform, y_uniform, theta_uniform, s_uniform, speed_limit))
        self.ref_path = np.array(ref_path[..., :2])  # 仅保留x,y坐标
        self.ref_theta = ref_path[..., 2]  # 切线角度
        self.cumulative_s = ref_path[..., 3]  # s坐标（弧长）
        self.speed_limit = ref_path[..., 4]  # 速度限制
        self.kd_tree = KDTree(np.concatenate((ref_path[..., :2], 
                                              np.cos(ref_path[..., 2:3]) * angle_penalty,
                                              np.sin(ref_path[..., 2:3]) * angle_penalty), axis=-1))  # 用于快速最近点查询
        self.angle_penalty = angle_penalty
    def cartesian_to_frenet(self, points: np.ndarray):
        """使用预计算数据的快速转换
        point: shape (..., 2) 包含x,y坐标
        返回: sd: shape (..., 2) 包含s,d坐标
        """
        # 找到最近点（可以使用KD树加速）
        expanded_points = np.concatenate((points[..., :2], 
                                          np.cos(points[..., 2:3]) * self.angle_penalty,
                                          np.sin(points[..., 2:3]) * self.angle_penalty), axis=-1)
        
        nearest_idx = self.kd_tree.query(expanded_points)[1]
        
        s = self.cumulative_s[nearest_idx]
        theta_ref = self.ref_theta[nearest_idx]

        dx = points[..., 0] - self.ref_path[nearest_idx, 0]
        dy = points[..., 1] - self.ref_path[nearest_idx, 1]
        d = -dx * np.sin(theta_ref) + dy * np.cos(theta_ref)
        
        sd = np.stack((s, d), axis=-1)
        return sd
    
    def frenet_to_cartesian(self, sd: np.ndarray, with_yaw: bool = False):
        """使用预计算数据的快速转换"""

        if sd.shape[-1] < 2:
            raise ValueError(f"sd must have (s,d). Got shape={sd.shape}")
        
        s = sd[..., 0]
        d = sd[..., 1]
        
        s = np.clip(s, self.cumulative_s[0], self.cumulative_s[-1])
        s = np.maximum.accumulate(s)
        d = sd[..., 1]

        # 找到s对应的索引
        right = np.searchsorted(self.cumulative_s, s, side="left")
        right = np.clip(right, 1, len(self.cumulative_s) - 1)
        left = right - 1

        s_left = self.cumulative_s[left]
        s_right = self.cumulative_s[right]
        w = (s - s_left) / np.maximum(s_right - s_left, 1e-6)  # 线性插值权重 (M,)
        
        p_left = self.ref_path[left]
        p_right = self.ref_path[right]
        ref_xy = p_left + w[..., None] * (p_right - p_left)

        theta_left = self.ref_theta[left]
        theta_right = self.ref_theta[right]

        v_left = np.stack([np.cos(theta_left), np.sin(theta_left)], axis=-1)
        v_right = np.stack([np.cos(theta_right), np.sin(theta_right)], axis=-1)
        v_interp = v_left + w[..., None] * (v_right - v_left)
        theta_ref = np.arctan2(v_interp[..., 1], v_interp[..., 0])

        d_safe = np.clip(d, -10.0, 10.0)  
        x = ref_xy[..., 0] - d_safe * np.sin(theta_ref)
        y = ref_xy[..., 1] + d_safe * np.cos(theta_ref)
        
        if with_yaw:
            return np.stack((x, y, theta_ref), axis=-1)
        return np.stack((x, y), axis=-1)
    
    def get_speed_limit(self, s_start: float, s_end: float) -> float:
        """获取s_start到s_end区间的最高速度限制"""
        s_start = np.clip(s_start, self.cumulative_s[0], self.cumulative_s[-1])
        s_end = np.clip(s_end, self.cumulative_s[0], self.cumulative_s[-1])
        if np.abs(s_end - s_start) < 1e-1:
            return self.get_speed_limit_at_s(s_start)
        if s_end < s_start:
            s_start, s_end = s_end, s_start
        idx_start = np.searchsorted(self.cumulative_s, s_start, side="left")
        idx_end = np.searchsorted(self.cumulative_s, s_end, side="right")
        return np.min(self.speed_limit[idx_start:idx_end])
    
    def get_speed_limit_at_s(self, s: float) -> float:
        """获取s位置的速度限制"""
        s = np.clip(s, self.cumulative_s[0], self.cumulative_s[-1])
        idx = np.searchsorted(self.cumulative_s, s, side="left")
        return self.speed_limit[idx]