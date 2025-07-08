import os
import shutil
import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation as R
import random

# --- 1. 配置参数 (与之前相同) ---
INPUT_ROOT = "/home/senzeyu2/dataset/opv2v/validate"
OUTPUT_ROOT = "/home/senzeyu2/dataset/opv2v/validate_ab"
POINT_CLOUD_RANGE = [-140.8, -40, -3, 140.8, 40, 1]
SPAWN_AREA_EGO = {
    'x_range': [10, 50],
    'y_range': [-10, 10],
    'z_range': [-1, 0.5]
}
ABNORMAL_SIZE_SCALE_RANGE = (0.5, 3.0)
ABNORMAL_TARGET_NUM_POINTS = 500
ABNORMAL_TARGET_ID = 99999
MAX_SPAWN_ATTEMPTS = 50
COLORIZE_ABNORMAL_TARGET = True
ABNORMAL_TARGET_COLOR = [1.0, 0.0, 0.0]


# --- 2. 辅助函数 ---

def generate_box_noise(center, extent, rotation_matrix, num_points):
    points = np.random.rand(num_points, 3) - 0.5
    points *= extent
    points = (rotation_matrix @ points.T).T + center
    return points


def pose_to_transformation_matrix(pose):
    x, y, z, roll, pitch, yaw = pose
    translation = [x, y, z]
    rotation = R.from_euler('zyx', [yaw, pitch, roll], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def get_gt_boxes_in_ego_coords(vehicles_data):
    gt_boxes = []
    if not vehicles_data:
        return gt_boxes
    for _, vehicle in vehicles_data.items():
        center = np.array(vehicle['center'])
        extent = np.array(vehicle['extent'])
        roll, pitch, yaw = vehicle['angle']
        rotation = R.from_euler('zyx', [yaw, pitch, roll], degrees=True).as_matrix()
        gt_boxes.append(o3d.geometry.OrientedBoundingBox(center, rotation, extent))
    return gt_boxes


def crop_pcd_by_range(pcd, pcd_range):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    mask = (
            (points[:, 0] >= pcd_range[0]) & (points[:, 0] <= pcd_range[3]) &
            (points[:, 1] >= pcd_range[1]) & (points[:, 1] <= pcd_range[4]) &
            (points[:, 2] >= pcd_range[2]) & (points[:, 2] <= pcd_range[5])
    )
    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(points[mask])
    if colors is not None:
        cropped_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    return cropped_pcd


# vvvvvvvvvvvvvv  修改部分开始 vvvvvvvvvvvvvv
def check_aabb_intersection(aabb1, aabb2):
    """
    手动检查两个轴对齐包围盒(AABB)是否相交。
    兼容所有Open3D版本。
    """
    min1, max1 = aabb1.get_min_bound(), aabb1.get_max_bound()
    min2, max2 = aabb2.get_min_bound(), aabb2.get_max_bound()

    # 检查在所有三个轴上是否都有重叠
    x_overlap = (max1[0] >= min2[0]) and (min1[0] <= max2[0])
    y_overlap = (max1[1] >= min2[1]) and (min1[1] <= max2[1])
    z_overlap = (max1[2] >= min2[2]) and (min1[2] <= max2[2])

    return x_overlap and y_overlap and z_overlap


def check_collision(new_box, existing_boxes):
    """检查新框是否与现有框重叠 (在同一坐标系下)"""
    new_aabb = new_box.get_axis_aligned_bounding_box()
    for existing_box in existing_boxes:
        existing_aabb = existing_box.get_axis_aligned_bounding_box()

        # 使用我们自己实现的、兼容性好的AABB相交检测
        if check_aabb_intersection(new_aabb, existing_aabb):
            # 如果AABB相交，再进行更精确的顶点检查
            new_box_vertices = new_box.get_box_points()
            existing_box_vertices = existing_box.get_box_points()
            if len(existing_box.get_point_indices_within_bounding_box(new_box_vertices)) > 0:
                return True  # 发生碰撞
            if len(new_box.get_point_indices_within_bounding_box(existing_box_vertices)) > 0:
                return True  # 发生碰撞
    return False  # 未发生碰撞


# ^^^^^^^^^^^^^^  修改部分结束 ^^^^^^^^^^^^^^

# --- 3. 主处理逻辑 (与之前相同) ---
def process_dataset(input_dir, output_dir):
    print(f"开始处理数据集，输入: '{input_dir}', 输出: '{output_dir}'")

    for scene_name in os.listdir(input_dir):
        scene_path = os.path.join(input_dir, scene_name)
        if not os.path.isdir(scene_path): continue

        print(f"\n--- 正在处理场景: {scene_name} ---")

        agents = sorted(
            [d for d in os.listdir(scene_path) if d.isdigit() and os.path.isdir(os.path.join(scene_path, d))])
        if not agents: continue

        ego_agent_id = agents[0]
        ego_agent_path = os.path.join(scene_path, ego_agent_id)
        timestamps = sorted([f.split('.')[0] for f in os.listdir(ego_agent_path) if f.endswith('.pcd')])

        for ts in timestamps:
            print(f"  处理时间戳: {ts}")

            ego_yaml_path = os.path.join(ego_agent_path, f"{ts}.yaml")
            if not os.path.exists(ego_yaml_path):
                print(f"    Ego-Agent '{ego_agent_id}' 缺少 {ts}.yaml，跳过此时间戳。")
                continue

            with open(ego_yaml_path, 'r') as f:
                ego_data = yaml.safe_load(f)

            ego_lidar_pose = ego_data.get('lidar_pose')
            ego_vehicles_data = ego_data.get('vehicles') or {}
            if not ego_lidar_pose:
                print(f"    Ego-Agent {ts}.yaml 缺少 pose 数据，跳过。")
                continue

            gt_boxes_ego = get_gt_boxes_in_ego_coords(ego_vehicles_data)

            if gt_boxes_ego:
                avg_gt_extent = np.mean(np.array([box.extent for box in gt_boxes_ego]), axis=0)
            else:
                avg_gt_extent = np.array([4.0, 1.8, 1.5])

            abnormal_target_info_ego = None
            for attempt in range(MAX_SPAWN_ATTEMPTS):
                scale = random.uniform(*ABNORMAL_SIZE_SCALE_RANGE)
                extent = avg_gt_extent * scale
                center = np.array([
                    random.uniform(*SPAWN_AREA_EGO['x_range']),
                    random.uniform(*SPAWN_AREA_EGO['y_range']),
                    random.uniform(*SPAWN_AREA_EGO['z_range']),
                ])
                yaw = random.uniform(0, 360)
                angle = [0.0, 0.0, yaw]
                rotation = R.from_euler('z', yaw, degrees=True).as_matrix()

                new_box_ego = o3d.geometry.OrientedBoundingBox(center, rotation, extent)

                if not check_collision(new_box_ego, gt_boxes_ego):
                    print(f"    成功在Ego坐标系下生成无碰撞异常目标 at {center.round(2)}")
                    abnormal_target_info_ego = {
                        'center': center,
                        'extent': extent,
                        'angle_deg': angle,
                        'rotation': rotation
                    }
                    break

            if abnormal_target_info_ego is None:
                print(f"    警告: 尝试 {MAX_SPAWN_ATTEMPTS} 次后仍未找到无碰撞位置，跳过此时间戳。")
                continue

            abnormal_target_pcd_ego = generate_box_noise(
                abnormal_target_info_ego['center'],
                abnormal_target_info_ego['extent'],
                abnormal_target_info_ego['rotation'],
                ABNORMAL_TARGET_NUM_POINTS
            )

            T_ego_to_world = pose_to_transformation_matrix(ego_lidar_pose)

            for agent_id in agents:
                agent_path = os.path.join(scene_path, agent_id)
                yaml_path = os.path.join(agent_path, f"{ts}.yaml")
                pcd_path = os.path.join(agent_path, f"{ts}.pcd")

                if not (os.path.exists(yaml_path) and os.path.exists(pcd_path)): continue

                with open(yaml_path, 'r') as f:
                    current_agent_data = yaml.safe_load(f)

                T_current_agent_to_world = pose_to_transformation_matrix(current_agent_data['lidar_pose'])
                T_world_to_current_agent = np.linalg.inv(T_current_agent_to_world)
                T_ego_to_current_agent = T_world_to_current_agent @ T_ego_to_world

                abnormal_target_ego_h = np.hstack(
                    [abnormal_target_pcd_ego, np.ones((abnormal_target_pcd_ego.shape[0], 1))])
                abnormal_target_current_agent_h = (T_ego_to_current_agent @ abnormal_target_ego_h.T).T
                abnormal_target_current_agent = abnormal_target_current_agent_h[:, :3]

                ego_gt_center_h = np.append(abnormal_target_info_ego['center'], 1)
                current_gt_center = (T_ego_to_current_agent @ ego_gt_center_h)[:3]

                R_ego_to_current = T_ego_to_current_agent[:3, :3]
                current_gt_rotation = R_ego_to_current @ abnormal_target_info_ego['rotation']
                current_gt_angle = R.from_matrix(current_gt_rotation).as_euler('zyx', degrees=True)
                current_gt_angle_rpy = [current_gt_angle[2], current_gt_angle[1], current_gt_angle[0]]

                abnormal_gt_entry = {
                    'angle': [float(a) for a in current_gt_angle_rpy],
                    'center': [float(c) for c in current_gt_center],
                    'extent': [float(e) for e in abnormal_target_info_ego['extent']],
                    'location': [0.0, 0.0, 0.0],
                    'speed': 0.0
                }

                if 'vehicles' not in current_agent_data or current_agent_data['vehicles'] is None:
                    current_agent_data['vehicles'] = {}
                current_agent_data['vehicles'][ABNORMAL_TARGET_ID] = abnormal_gt_entry

                original_pcd = o3d.io.read_point_cloud(pcd_path)
                combined_points = np.vstack([np.asarray(original_pcd.points), abnormal_target_current_agent])
                new_pcd = o3d.geometry.PointCloud()
                new_pcd.points = o3d.utility.Vector3dVector(combined_points)

                # (颜色处理逻辑可以在这里添加)

                final_pcd = crop_pcd_by_range(new_pcd, POINT_CLOUD_RANGE)

                output_agent_path = os.path.join(output_dir, scene_name, agent_id)
                os.makedirs(output_agent_path, exist_ok=True)

                output_yaml_path = os.path.join(output_agent_path, f"{ts}.yaml")
                with open(output_yaml_path, 'w') as f:
                    yaml.dump(current_agent_data, f, default_flow_style=None, sort_keys=False)

                output_pcd_path = os.path.join(output_agent_path, f"{ts}.pcd")
                o3d.io.write_point_cloud(output_pcd_path, final_pcd)

    print("\n处理完成！")


if __name__ == '__main__':
    if os.path.exists(OUTPUT_ROOT):
        response = input(f"输出目录 '{OUTPUT_ROOT}' 已存在，继续将覆盖其中文件。是否继续? (y/n): ").lower()
        if response != 'y':
            print("操作已取消。")
            exit()

    process_dataset(INPUT_ROOT, OUTPUT_ROOT)
