import os
import numpy as np
import struct
import open3d as o3d
from tqdm import tqdm
import argparse
import cv2 


class_map = {
    "Car":0,
    "Van":1,
    "Truck":2,
    "Pedestrians":3,
    "Sitters":4,
    "Cyclists":5,
    "Trams":6,
    "Misc":7
}

Tr_cam_to_velo = np.linalg.inv(np.array([
    [ 0.007533745, -0.9997140, -0.000616602, -0.00406977],
    [ 0.01480249,  0.000728073, -0.9998902,  -0.07631618],
    [ 0.9998621,   0.007523790,  0.01480755, -0.2717806],
    [ 0, 0, 0, 1 ]
]))


def convert_kitti_to_custom(input_txt, output_txt):
    with open(input_txt, 'r') as f_in, open(output_txt, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split()
            if (len(parts) < 15):
                continue

            name = parts[0]
            dims = np.array([float(parts[10]), float(parts[9]), float(parts[8])])
            loc_cam = np.array([float(parts[11]), float(parts[12]), float(parts[13]), 1.0])
            ry = float(parts[14])
            score = float(parts[15]) if len(parts) > 15 else 1.0

            cls_id = class_map.get(name, -1)
            if cls_id == -1:
                continue

            loc_lidar = Tr_cam_to_velo @ loc_cam
            center = loc_lidar[:3]
            heading = -ry - np.pi / 2

            f_out.write("%4f %.4f %.4f %4f %.4f %.4f %.4f %d %.4f\n" % (center[0], center[1], center[2], dims[0], dims[1], dims[2], heading, cls_id, score))


def compute_box_corners(center, size, heading):
    dx, dy, dz = size
    x_corners = [dx/2, dx/2, -dx/2, -dx/2, dx/2, dx/2, -dx/2, -dx/2]
    y_corners = [dy/2, -dy/2, -dy/2, dy/2, dy/2, -dy/2, -dy/2, dy/2]
    z_corners = [dz/2, dz/2, dz/2, dz/2, -dz/2, -dz/2, -dz/2, -dz/2]
    corners = np.vstack([x_corners, y_corners, z_corners])
    R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_axis_angle([0, heading, 0])
    rotated = R @ corners 
    translated = rotated + np.array(center).reshape(3, 1)
    return translated.T


def project_to_image(points_3d, P):
    """Project 3D points to 2D using projection matrix"""
    points_3d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # (N, 4)
    pts_2d = (P @ points_3d.T).T  # (N, 3)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, :2]

def draw_projected_box(image, corners_2d, cls_name, probability, color=(0,255,0)):
    """Draw 3D box on image from 8 corners (projected)"""
    corners_2d = corners_2d.astype(int)
    edges = [
        (0,1), (1,2), (2,3), (3,0),  # top square
        (4,5), (5,6), (6,7), (7,4),  # bottom square
        (0,4), (1,5), (2,6), (3,7)   # vertical lines
    ]
    for i, j in edges:
        pt1, pt2 = tuple(corners_2d[i]), tuple(corners_2d[j])
        cv2.line(image, pt1, pt2, color, 2)

    top_center = np.mean(corners_2d[0:4], axis=0).astype(int)
    cv2.putText(image, cls_name, tuple(top_center), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0,0,255), 2, lineType=cv2.LINE_AA)
    cv2.putText(image, f"({round(probability, 2)})", tuple([top_center[0], top_center[1]+20]), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0,0,255), 2, lineType=cv2.LINE_AA)

    return image

def main():
    # 예시 입력
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-number', required=True)
    parser.add_argument('--pcdet', action='store_true')
    args = parser.parse_args()
    data_num = args.data_number


    image_path = "/root/shared/kitti/kitti/training/image_2/" + data_num + ".png"

    if (args.pcdet): 
        print(f"pcdet True")
        original_bbox_path = "/workspace/OpenPCDet/output/kitti_models/pointpillar/default/eval/eval_all_default/pcdet/" + data_num + ".txt"
        bbox_path = "/workspace/OpenPCDet/output/kitti_models/pointpillar/default/eval/eval_all_default/pcdet/" + data_num + "_new.txt"
        convert_kitti_to_custom(original_bbox_path, bbox_path)
    else:
        print(f"pcdet False")
        bbox_path = "/root/shared/kitti/kitti/training/velodyne/" + data_num + ".txt"

    # KITTI Calibration
    # Projection matrix P2 (Camera 2)
    P = np.array([
        [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
        [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
        [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]
    ])

    # Transformation: LiDAR to Camera (Tr_velo_to_cam)
    Tr_velo_to_cam = np.array([
        [ 0.007533745, -0.9997140, -0.000616602, -0.00406977],
        [ 0.01480249,  0.000728073, -0.9998902,  -0.07631618],
        [ 0.9998621,   0.007523790,  0.01480755, -0.2717806 ]
    ])
    
    # Load image
    img = cv2.imread(image_path)


    # Read bounding box txt file
    with open(bbox_path, 'r') as f:
        print(f"bbox_path : {bbox_path}")
        for line in f:
            parts = list(map(float, line.strip().split()))
            center = parts[0:3]
            size = parts[3:6]
            heading = parts[6]
            obj_cls = int(parts[7])
            probability = float(parts[8])
            if probability < 0:
                continue

            # Step 1: 3D bounding box corners in LiDAR frame
            corners_lidar = compute_box_corners(center, size, heading)

            # Step 2: Transform to camera frame
            corners_cam = (Tr_velo_to_cam @ np.hstack((corners_lidar, np.ones((8,1)))).T).T  # (8, 3)

            if np.any(corners_cam[:, 2] <= 0):
                print(f"Skipping box : behind camera")
                continue

            # Step 3: Project to 2D
            corners_2d = project_to_image(corners_cam, P)

            # Step 4: Draw on image
            cls_name = next(k for k, v in class_map.items() if v == obj_cls)
            img = draw_projected_box(img, corners_2d, cls_name, probability)

    cv2.imwrite("projected.png", img)

if __name__=="__main__":
    main()

