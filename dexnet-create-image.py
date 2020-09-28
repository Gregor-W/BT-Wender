import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
import tempfile
import shutil
from pathlib2 import Path

from autolab_core import YamlConfig
import dexnet
import dexnet.database.mesh_processor as mp
from dexnet.grasping.gripper import RobotGripper
from dexnet.grasping.grasp_sampler import AntipodalGraspSampler
from dexnet.grasping.graspable_object import GraspableObject3D
from dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory
from dexnet.grasping.grasp_quality_function import GraspQualityFunctionFactory
from dexnet.grasping import GraspCollisionChecker

from autolab_core import NormalCloud, PointCloud, RigidTransform
from meshpy import MaterialProperties, LightingProperties, ObjFile, VirtualCamera, ViewsphereDiscretizer, SceneObject

import numpy as np
from perception import CameraIntrinsics, RenderMode
from meshpy.mesh_renderer import ViewsphereDiscretizer, VirtualCamera
from visualization import Visualizer2D as vis

import pickle

output = "/home/ubuntu/egad/grasp-data/test-object0"
temp_dir = "/tmp/obj-files"

# Use local config file
#config_path = '/home/ubuntu/egad/egad/scripts/cfg/dexnet_api_settings.yaml'
grasp_config = YamlConfig(str('/home/ubuntu/test-dexnet/test/config.yaml'))
config = YamlConfig("/home/ubuntu/test-dexnet/cfg/tools/generate_gqcnn_dataset.yaml")
render_config = YamlConfig("/home/ubuntu/egad/config/dexnet-settings.yaml")


stable_pose_min_p = 0

# Requires data from the dexnet project.
os.chdir(str(Path(dexnet.__file__).resolve().parent.parent.parent))

# setup grasp params
table_alignment_params = config['table_alignment']
min_grasp_approach_offset = -np.deg2rad(table_alignment_params['max_approach_offset'])
max_grasp_approach_offset = np.deg2rad(table_alignment_params['max_approach_offset'])
max_grasp_approach_table_angle = np.deg2rad(table_alignment_params['max_approach_table_angle'])
num_grasp_approach_samples = table_alignment_params['num_approach_offset_samples']

phi_offsets = []
if max_grasp_approach_offset == min_grasp_approach_offset:
    phi_inc = 1
elif num_grasp_approach_samples == 1:
    phi_inc = max_grasp_approach_offset - min_grasp_approach_offset + 1
else:
    phi_inc = (max_grasp_approach_offset - min_grasp_approach_offset) / (num_grasp_approach_samples - 1)
                                                        
phi = min_grasp_approach_offset
while phi <= max_grasp_approach_offset:
    phi_offsets.append(phi)
    phi += phi_inc

coll_check_params = config['collision_checking']
approach_dist = coll_check_params['approach_dist']
delta_approach = coll_check_params['delta_approach']
table_offset = coll_check_params['table_offset']
table_mesh_filename = coll_check_params['table_mesh_filename']



# Write aligned grasps to pickle 
def grasp_depth_images(dir_path, mesh_file):
    #mp_cache = tempfile.mkdtemp()
    
    mp_cache = '/home/ubuntu/egad/grasp-data/object-files'
    
    # prepare mesh and +generate Grasps
    mesh_processor = mp.MeshProcessor(os.path.join(dir_path, mesh_file), mp_cache)
    mesh_processor.generate_graspable(grasp_config)

    gripper = RobotGripper.load('yumi_metal_spline', gripper_dir=grasp_config['gripper_dir'])
    sampler = AntipodalGraspSampler(gripper, grasp_config)
    obj = GraspableObject3D(mesh_processor.sdf, mesh_processor.mesh)
    grasps = sampler.generate_grasps(obj, max_iter=grasp_config['max_grasp_sampling_iters'])
    
    # setup collision checker
    collision_checker = GraspCollisionChecker(gripper)
    collision_checker.set_graspable_object(obj)
    
    grasps_trans = list()

    # read in the stable poses of the mesh
    stable_poses = mesh_processor.stable_poses
    for i, stable_pose in enumerate(stable_poses):
        # render images if stable pose is valid
        if stable_pose.p > stable_pose_min_p:

            # setup table in collision checker
            T_obj_stp = stable_pose.T_obj_table.as_frames('obj', 'stp')
            T_obj_table = obj.mesh.get_T_surface_obj(T_obj_stp, delta=table_offset).as_frames('obj', 'table')
            T_table_obj = T_obj_table.inverse()
            collision_checker.set_table(table_mesh_filename, T_table_obj)

            # align grasps with the table
            aligned_grasps = [grasp.perpendicular_table(stable_pose) for grasp in grasps]

            # check grasp validity
            for aligned_grasp in aligned_grasps:
                # check angle with table plane and skip unaligned grasps
                _, grasp_approach_table_angle, _ = aligned_grasp.grasp_angles_from_stp_z(stable_pose)
                perpendicular_table = (np.abs(grasp_approach_table_angle) < max_grasp_approach_table_angle)
                if not perpendicular_table: 
                    continue

                # check whether any valid approach directions are collision free
                collision_free = False
                for phi_offset in phi_offsets:
                    rotated_grasp = aligned_grasp.grasp_y_axis_offset(phi_offset)
                    collides = collision_checker.collides_along_approach(rotated_grasp, approach_dist, delta_approach)
                    if not collides:
                        collision_free = True
                        break

                # visualize
                if collision_free:
                    grasps_trans.append({"grasp_T": aligned_grasp.T_grasp_obj,
                                         "axis": aligned_grasp.axis,
                                         "close_width": aligned_grasp.close_width,
                                         "approach_angle": aligned_grasp.approach_angle,
                                         "T_obj_table": stable_pose.T_obj_table,
                                         "T_table_obj": stable_pose.T_obj_table.inverse(),
                                         "T_obj_world": stable_pose.T_obj_world,
                                         "table_pose": T_table_obj})
    
    print("found %d grasps, writing to file" % len(grasps_trans))
    
    with open(os.path.join(output, 'aligned_grasps_list.pkl'), 'wb') as out:
            pickle.dump(grasps_trans, out)


mesh_files_dir = "/home/ubuntu/egad/output/1589817197/pool"

# get list of all mesh files
all_files = os.listdir(mesh_files_dir)
obj_files = [f for f in all_files if f.endswith('.obj')]

# filter mesh files
filter_mesh_files = '0024'
mesh_files = [f for f in obj_files if f.split('_')[0] == filter_mesh_files]
mesh_files = [mesh_files[1]]

print(mesh_files)

for m in mesh_files:
    grasp_depth_images(mesh_files_dir, m)

