from typing import List
from lpl_planner.planning.scene.scene_feature.features import SceneFeature, AgentPrediction
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def draw_model_in_out(
                scene_feature: SceneFeature,
                chosen_trajectory = None,
                expert_trajectory = None,
                all_trajectories = None,
                all_trajectory_scores = None,
                trajectory_alpha_by_score: bool = False,
                trajectory_alpha_min: float = 0.15,
                trajectory_alpha_max: float = 0.85,
                agent_prediction: AgentPrediction = None,
                agent_prediction_gt: AgentPrediction = None,
                num_steps: int = None,
                route_polygons: List[np.ndarray] = None,
                all_trajectories_simed: np.ndarray = None,
                use_trajectory_limit: bool = False,
                ref_path: np.ndarray = None
        ):
    plt.ioff()
    fig, axes = plt.subplots(figsize=(10,10))
    axes.set_aspect('equal')
    axes.set(xlim=[-20, 60])
    axes.set(ylim=[-40, 40])
    if use_trajectory_limit:
        trajectory = expert_trajectory if expert_trajectory is not None else chosen_trajectory
        x_lim = [np.min(trajectory[:,0])-5, np.max(trajectory[:,0])+5]
        y_lim = [np.min(trajectory[:,1])-5, np.max(trajectory[:,1])+5]
        max_range = max(x_lim[1]-x_lim[0], y_lim[1]-y_lim[0])
        max_range = max(50, max_range)  # 最小范围50米
        x_center = (x_lim[0] + x_lim[1]) / 2
        y_center = (y_lim[0] + y_lim[1]) / 2
        x_lim = [x_center - max_range / 2, x_center + max_range / 2]
        y_lim = [y_center - max_range / 2, y_center + max_range / 2]
        axes.set(xlim=x_lim)
        axes.set(ylim=y_lim)

    # load features
    road_feature = scene_feature.road_feature
    route_feature = scene_feature.route_feature
    # ref_path_feature = scene_feature.ref_path_feature
    ego_feature = scene_feature.ego_feature
    static_obstacle_feature = scene_feature.static_obstacle_feature
    agent_feature = scene_feature.agent_feature
    
    def rotate_poly(poly, yaw, center):
        """"""
        c, s = np.cos(yaw), np.sin(yaw)
        rot_mat = np.array([[c, -s], [s, c]])
        return (poly - center) @ rot_mat.T + center

    

    # 绘制道路中心线
    for idx, center_line in enumerate(road_feature.center_line):
        center_line = np.array(center_line)
        tl_date = road_feature.road_traffic_light[idx]
        if center_line.shape[0] > 1:
            if tl_date == 3:  # 红灯
                axes.plot(center_line[:, 0], center_line[:, 1], color='red', linestyle='--', linewidth=1, alpha=0.7)
            else:
                axes.plot(center_line[:, 0], center_line[:, 1], color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # 绘制道路多边形
    # print(f'road_feature.road_geometry shape: {road_feature.road_geometry.shape}')
    for idx, polygon in enumerate(road_feature.road_geometry):
        polygon = np.array(polygon)
        tl_date = road_feature.road_traffic_light[idx]
        if polygon.shape[0] > 2:
            if tl_date == 3:  # 红灯
                axes.fill(polygon[:, 0], polygon[:, 1], color='lightcoral', alpha=0.3, edgecolor='red')
            else:
                axes.fill(polygon[:, 0], polygon[:, 1], color='lightgray', alpha=0.3, edgecolor='gray')

    # # 绘制route道路中心线
    # for idx, center_line in enumerate(route_feature.center_line):
    #     center_line = np.array(center_line)
    #     if center_line.shape[0] > 1:
    #         axes.plot(center_line[:, 0], center_line[:, 1], color='green', linestyle='--', linewidth=1, alpha=0.7)
            
    # 绘制route道路多边形
    for idx, polygon in enumerate(route_feature.route_geometry):
        polygon = np.array(polygon)
        if polygon.shape[0] > 2:
            axes.fill(polygon[:, 0], polygon[:, 1], color='green', alpha=0.2)

    if ref_path is not None:
        center_line = ref_path[:, :2]
        if center_line.shape[0] > 1:
            axes.plot(center_line[:, 0], center_line[:, 1], color='green', linestyle='-', linewidth=2, alpha=0.5)

    # 绘制本车多边形
    ego_pose = np.array([0,0,0])
    ego_half_width = ego_feature.ego_geometry[0]
    ego_half_length = ego_feature.ego_geometry[1]
    rear_axle_to_center = ego_feature.ego_geometry[2]
    rear_axle_to_center_translate = np.stack(
        [rear_axle_to_center * np.cos(ego_pose[2]), rear_axle_to_center * np.sin(ego_pose[2])], axis=-1
    )
    ego_center = ego_pose[:2] + rear_axle_to_center_translate
    ego_poly = np.array([
        [ego_center[0] - ego_half_length, ego_center[1] - ego_half_width],
        [ego_center[0] - ego_half_length, ego_center[1] + ego_half_width],
        [ego_center[0] + ego_half_length, ego_center[1] + ego_half_width],
        [ego_center[0] + ego_half_length, ego_center[1] - ego_half_width]
    ])
    axes.fill(ego_poly[:, 0], ego_poly[:, 1], color='blue', alpha=0.5, label='Ego Vehicle', edgecolor='blue')

    # 绘制本车历史轨迹
    ego_history = np.array(ego_feature.ego_history_state)  # [T, (x,y,yaw,vx,vy)]
    valid_history = ego_history
    axes.plot(valid_history[:, 0], valid_history[:, 1], color='blue', linestyle='--', linewidth=1.5, alpha=0.5)

    # 绘制静态障碍物多边形
    if len(static_obstacle_feature.static_obstacle_position) > 0:
        static_obstacle_feature.static_obstacle_position = np.array(static_obstacle_feature.static_obstacle_position) 
        static_obj_pos = static_obstacle_feature.static_obstacle_position # [N, (x,y,yaw)]
        static_obj_dim = static_obstacle_feature.static_object_dimension # [N, (half_length, half_width)]
        static_obj_poly = [
        np.array([
            [static_obj_pos[i][0] - static_obj_dim[i][0], static_obj_pos[i][1] - static_obj_dim[i][1]],
            [static_obj_pos[i][0] - static_obj_dim[i][0], static_obj_pos[i][1] + static_obj_dim[i][1]],
            [static_obj_pos[i][0] + static_obj_dim[i][0], static_obj_pos[i][1] + static_obj_dim[i][1]],
            [static_obj_pos[i][0] + static_obj_dim[i][0], static_obj_pos[i][1] - static_obj_dim[i][1]]
        ])
        for i in range(len(static_obj_pos))
        ]

        static_obj_poly_rotated = []
        for i, poly in enumerate(static_obj_poly):
            yaw = static_obstacle_feature.static_obstacle_position[i][2]  # yaw in radians
            center = static_obstacle_feature.static_obstacle_position[i][:2]
            poly_rot = rotate_poly(poly, yaw, center)
            static_obj_poly_rotated.append(poly_rot)
        static_obj_poly = np.array(static_obj_poly_rotated)
        static_obj_poly = np.array(static_obj_poly)
        static_obj_poly = static_obj_poly.reshape(-1, 4, 2)
        # 绘制静态障碍物多边形
        for static_obj_idx, poly in enumerate(static_obj_poly):
            pose = static_obj_pos[static_obj_idx]
            if poly.shape[0] > 2:
                if static_obj_idx == 0:
                    axes.fill(poly[:, 0], poly[:, 1], color='red', alpha=0.3, label='Static Obstacle', edgecolor='red')
                else:
                    axes.fill(poly[:, 0], poly[:, 1], color='red', alpha=0.3, edgecolor='red')
    if route_polygons is not None:
        for idx, polygon in enumerate(route_polygons):
            polygon = np.array(polygon)
            if polygon.shape[0] > 2:
                axes.fill(polygon[:, 0], polygon[:, 1], color='yellow', alpha=0.2, edgecolor='gold', label='Route Area' if idx==0 else None)

    # 绘制周围车辆多边形与历史轨迹
    if len(agent_feature.agent_current_state) > 0:

        # print(f"Agent vehicles found: {len(agent_feature.agent_current_state)}")
        agent_current = np.array(agent_feature.agent_current_state)
        agent_geo = agent_feature.agent_geometry  # [N, (half_length, half_width)]
        agent_hist = agent_feature.agent_history_state  # [N, T, (x,y,yaw,vx,vy)]
        agent_hist_mask = agent_feature.agent_history_mask  # [N, T]
        agent_poly = [
            np.array([
            [agent_current[i][0] - agent_geo[i][0], agent_current[i][1] - agent_geo[i][1]],
            [agent_current[i][0] - agent_geo[i][0], agent_current[i][1] + agent_geo[i][1]],
            [agent_current[i][0] + agent_geo[i][0], agent_current[i][1] + agent_geo[i][1]],
            [agent_current[i][0] + agent_geo[i][0], agent_current[i][1] - agent_geo[i][1]]
        ])
        for i in range(len(agent_current))
        ]
        agent_poly_rotated = []
        for i, poly in enumerate(agent_poly):
            yaw = agent_current[i][2]  # yaw in radians
            center = agent_current[i][:2]
            poly_rot = rotate_poly(poly, yaw, center)
            agent_poly_rotated.append(poly_rot)
        agent_poly = np.array(agent_poly_rotated)
        agent_poly = np.array(agent_poly)
        agent_poly = agent_poly.reshape(-1, 4, 2)
        # 绘制周围车辆多边形与历史轨迹
        for i, _ in enumerate(agent_current):
            pose = agent_current[i]
            poly = agent_poly[i]
            if poly.shape[0] > 2:
                if i == 0:
                    axes.fill(poly[:, 0], poly[:, 1], color='orange', alpha=0.3, label='Agent Vehicle', edgecolor='orange')
                else:
                    axes.fill(poly[:, 0], poly[:, 1], color='orange', alpha=0.3, edgecolor='orange')
                # 绘制历史轨迹（去除padding，并用不同颜色区分）
                history = agent_hist[i]
                hist_mask = agent_hist_mask[i]
                # 只保留有效的历史轨迹点
                valid_idx = np.where(hist_mask)[0]
                if len(valid_idx) > 1:
                    valid_history = history[valid_idx]
                    axes.plot(valid_history[:, 0], valid_history[:, 1], color='purple', linestyle='--', linewidth=1.5, alpha=0.3)
        if agent_prediction_gt is not None:
            # 绘制预测轨迹
            agent_future = np.array(agent_prediction_gt.agent_future_state)  # [N, T, (x,y,yaw,vx,vy)]
            agent_future_mask = np.array(agent_prediction_gt.agent_future_mask)
            # 去掉padding的future轨迹
            for i, future in enumerate(agent_future):
                mask = agent_future_mask[i]
                valid_idx = np.where(mask)[0]
                if len(valid_idx) > 1:
                    valid_future = future[valid_idx]
                    axes.plot(valid_future[:, 0], valid_future[:, 1], color='purple', linestyle='dotted', linewidth=2, alpha=0.5)
        if agent_prediction is not None:
            # 绘制预测轨迹
            agent_future = np.array(agent_prediction.agent_future_state)  # [N, T, (x,y,yaw,vx,vy)]
            agent_future_mask = np.array(agent_prediction.agent_future_mask)
            # 去掉padding的future轨迹
            for i, future in enumerate(agent_future):
                mask = agent_future_mask[i]
                valid_idx = np.where(mask)[0]
                if len(valid_idx) > 1:
                    valid_future = future[valid_idx]
                    axes.plot(valid_future[:, 0], valid_future[:, 1], color='orange', linestyle='-', linewidth=2, alpha=0.6)

    # 绘制参考路径
    # if scene_feature.route_feature is None:
    #     ref_path = ref_path_feature if type(ref_path_feature) is np.ndarray else np.array(ref_path_feature)  # [T, (x,y,yaw,left_bound,right_bound)]
    #     lane_bound = ref_path[:,3:5]
    #     left_bound = np.expand_dims(lane_bound[:,0],axis=1)*np.vstack((np.cos(ref_path[:,2]+np.pi/2),np.sin(ref_path[:,2]+np.pi/2))).T + ref_path[:,:2]
    #     right_bound = np.expand_dims(lane_bound[:,1],axis=1)*np.vstack((np.cos(ref_path[:,2]+np.pi/2),np.sin(ref_path[:,2]+np.pi/2))).T + ref_path[:,:2]
    #     axes.plot(ref_path[:, 0], ref_path[:, 1], color='green', linestyle='-', linewidth=2, alpha=0.5)
    #     # print(f'left_bound.shape: {left_bound.shape}')
    #     # 绘制参考路径的左右边界多边形
    #     ref_path_poly = np.vstack([left_bound, right_bound[::-1]])
    #     axes.fill(ref_path_poly[:, 0], ref_path_poly[:, 1], color='green', alpha=0.2)
    #     # ax.fill_between(ref_path[:, 0], left_bound, right_bound, color='green', alpha=0.2, label='Reference Path Boundary')
    # else:
    #     ref_path = ref_path_feature if type(ref_path_feature) is np.ndarray else np.array(ref_path_feature)  # [T, (x,y,yaw,left_bound,right_bound)]
    #     axes.plot(ref_path[:, 0], ref_path[:, 1], color='green', linestyle='-', linewidth=2, alpha=0.5)
        
    # 绘制所有候选轨迹
    if all_trajectories is not None and all_trajectory_scores is not None:
        plot_order = np.argsort(np.asarray(all_trajectory_scores), kind='stable')
        ordered_trajectories = np.asarray(all_trajectories)[plot_order]
        ordered_scores = np.asarray(all_trajectory_scores)[plot_order]
        ordered_simed = np.asarray(all_trajectories_simed)[plot_order] if all_trajectories_simed is not None else None
        score_norm = ordered_scores / (np.max(ordered_scores) + 1e-6)
        for idx, traj_score in enumerate(score_norm):
            # 颜色按score从低到高：浅蓝 -> 深蓝
            cmap = plt.get_cmap('Blues')
            color_rgba = cmap(0.2 + 0.8 * traj_score)
            traj_alpha = 0.4
            if trajectory_alpha_by_score:
                traj_alpha = trajectory_alpha_min + (trajectory_alpha_max - trajectory_alpha_min) * float(traj_score)
            axes.plot(
                ordered_trajectories[idx][:, 0],
                ordered_trajectories[idx][:, 1],
                color=color_rgba,
                linewidth=5,
                alpha=traj_alpha,
                label='Candidate Trajectory' if idx==0 else None,
            )
            if ordered_simed is not None:
                sim_alpha = 0.8 if not trajectory_alpha_by_score else min(1.0, traj_alpha + 0.1)
                axes.plot(
                    ordered_simed[idx][:, 0],
                    ordered_simed[idx][:, 1],
                    color=color_rgba,
                    linestyle='--',
                    linewidth=0.8,
                    alpha=sim_alpha,
                    label='Simulated Trajectory' if idx==0 else None,
                )

    # 绘制选中轨迹
    if chosen_trajectory is not None:
        if num_steps is not None:
            # draw segmented chosen_trajectory with color/linestyle/linewidth transitions
            traj = np.asarray(chosen_trajectory)
            n_pts = min(int(num_steps), traj.shape[0]) if isinstance(num_steps, int) else traj.shape[0]
            if n_pts >= 2:
                n_seg = n_pts - 1
                styles = ['-', '--', '-.', ':']
                dark_cyan = np.array([0.0, 0.55, 0.55])
                light_cyan = np.array([0.7, 1.0, 1.0])
                for i in range(n_seg):
                    idx = np.arange(i * traj.shape[0] // n_pts, (i + 1) * traj.shape[0] // n_pts + 1)
                    t = i / max(1, n_seg - 1)
                    color = (1.0 - t) * dark_cyan + t * light_cyan
                    lw = float(6.0 - 4.0 * t)  # thicker to thinner
                    style = styles[min(int(t * (len(styles) - 1)), len(styles) - 1)]
                    axes.plot(
                        traj[idx, 0], traj[idx, 1],
                        color=color, linestyle=style, linewidth=lw, alpha=0.9,
                        label='Chosen Trajectory' if i == 0 else None
                    )
        else:
            axes.plot(chosen_trajectory[:, 0], chosen_trajectory[:, 1], color='cyan', linewidth=3, alpha=0.8, label='Chosen Trajectory')
    
    if expert_trajectory is not None:
        if chosen_trajectory is not None:
            traj_len = chosen_trajectory.shape[0]
            if expert_trajectory.shape[0] > traj_len:
                expert_trajectory = expert_trajectory[:traj_len]

        axes.plot(expert_trajectory[:, 0], expert_trajectory[:, 1], color='red', linewidth=1, alpha=0.8, label='Expert Trajectory')
    
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = rgba[..., :3].copy()  # Convert to RGB by dropping alpha channel
    plt.close(fig)
    return img