import os
import numpy as np
import struct
import open3d as o3d
from tqdm import tqdm
import argparse

move_step = 0.1 
rotation_step = np.pi / 18

def get_rotation_callback(pcd, bboxes, axis, angle):
    def callback(vis):
        R = pcd.get_rotation_matrix_from_axis_angle(np.array(axis) * angle)
        pcd.rotate(R, center = [0, 0, 0])
        vis.update_geometry(pcd)
        for box in bboxes:
            box.rotate(R, center = [0, 0, 0])
            vis.update_geometry(pcd)
        return False
    return callback
        

def get_translation_callback(pcd, bboxes, delta):
    def callback(vis):
        pcd.translate(delta)
        vis.update_geometry(pcd)
        for box in bboxes:
            box.translate(delta)
            vis.update_geometry(box)
        return False
    return callback

def create_bounding_box(center, size, heading):
    dx, dy, dz = size
    cx, cy, cz = center

    box = o3d.geometry.OrientedBoundingBox()
    box.center = [cx, cy, cz]
    box.extent = [dx, dy, dz]

    R = box.get_rotation_matrix_from_axis_angle([0, heading, 0])
    box.rotate(R, center=box.center)
    box.color = (1, 0, 0)
    return box

def read_bin_velodyne(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('ffff',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],point[2]])
    return np.asarray(pc_list,dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-number', required=True)
    args = parser.parse_args()

    data_num = args.data_number
    data_pcd_path = data_num + ".bin"
    data_txt_path = data_num + ".txt"

    pcd=o3d.open3d.geometry.PointCloud()
    path=data_pcd_path
    example=read_bin_velodyne(path)
    pcd.points= o3d.open3d.utility.Vector3dVector(example)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Move With Arrow Keys")

    box_list = []
    with open(data_txt_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            center = parts[0:3]
            size = parts[3:6]
            heading = parts[6]
            bbox = create_bounding_box(center, size, heading)
            box_list.append(bbox)
            vis.add_geometry(bbox)

    # R1 : Bird-Eye View
    # R2 : Side View
    # R2@R1 : Horizontal View
    R1 = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi / 2))
    R2 = pcd.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
    pcd.rotate(R2 @ R1, center = [0, 0, 0])
    vis.add_geometry(pcd)

    for box in box_list:
        box.rotate(R1, center = [0, 0, 0])
        box.rotate(R2, center = [0, 0, 0])
        vis.add_geometry(box)


    # Control Forward & Backward w/ Mouse Control
    vis.register_key_callback(262, get_translation_callback(pcd, box_list, [move_step, 0, 0]))
    vis.register_key_callback(263, get_translation_callback(pcd, box_list, [-move_step, 0, 0]))
    vis.register_key_callback(264, get_translation_callback(pcd, box_list, [0, -move_step, 0]))
    vis.register_key_callback(265, get_translation_callback(pcd, box_list, [0, move_step, 0]))

    vis.register_key_callback(328, get_rotation_callback(pcd, box_list, [1, 0, 0], rotation_step))
    vis.register_key_callback(322, get_rotation_callback(pcd, box_list, [1, 0, 0], -rotation_step))
    vis.register_key_callback(324, get_rotation_callback(pcd, box_list, [0, 1, 0], rotation_step))
    vis.register_key_callback(326, get_rotation_callback(pcd, box_list, [0, 1, 0], -rotation_step))
    vis.register_key_callback(327, get_rotation_callback(pcd, box_list, [0, 0, 1], rotation_step))
    vis.register_key_callback(329, get_rotation_callback(pcd, box_list, [0, 0, 1], -rotation_step))

    vis.run()
    vis.destroy_window()

if __name__=="__main__":
    main()

