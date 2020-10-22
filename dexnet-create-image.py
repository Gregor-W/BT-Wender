import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
import tempfile
import shutil
from pathlib2 import Path
import subprocess

from autolab_core import YamlConfig
import dexnet
import dexnet.database.mesh_processor as mp
from dexnet.grasping.gripper import RobotGripper
from dexnet.grasping.grasp_sampler import AntipodalGraspSampler
from dexnet.grasping.graspable_object import GraspableObject3D
from dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory
from dexnet.grasping.grasp_quality_function import GraspQualityFunctionFactory, QuasiStaticQualityFunction
from dexnet.grasping import GraspCollisionChecker

from autolab_core import NormalCloud, PointCloud, RigidTransform

import numpy as np
from perception import CameraIntrinsics, RenderMode
from visualization import Visualizer2D as vis

import pickle
from Render import get_grasp

# output directories
pickle_output = "/home/ubuntu/egad/grasp-data/pickle-files"
mesh_output = '/home/ubuntu/egad/grasp-data/object-files'

# input egad folder with meshes
mesh_files_dir = "/home/ubuntu/egad/output/1603106069/pool"

# filter mesh files for egad generation
filter_mesh_files = '0032'

# Use local config file
egad_path = YamlConfig('/home/ubuntu/egad/egad/scripts/cfg/dexnet_api_settings.yaml')
grasp_config = YamlConfig('/home/ubuntu/test-dexnet/test/config.yaml') # DEXNET test config
coll_vis_config = YamlConfig("/home/ubuntu/test-dexnet/cfg/tools/generate_gqcnn_dataset.yaml")

# min stable pose probability
stable_pose_min_p = 0

# Requires data from the dexnet project.
os.chdir(str(Path(dexnet.__file__).resolve().parent.parent.parent))

# setup grasp params
table_alignment_params = coll_vis_config['table_alignment']
min_grasp_approach_offset = -np.deg2rad(table_alignment_params['max_approach_offset'])
max_grasp_approach_offset = np.deg2rad(table_alignment_params['max_approach_offset'])
max_grasp_approach_table_angle = np.deg2rad(table_alignment_params['max_approach_table_angle'])
num_grasp_approach_samples = table_alignment_params['num_approach_offset_samples']

# paramet for alignement of grasps
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

# setup collision params for alignment
coll_check_params = coll_vis_config['collision_checking']
approach_dist = coll_check_params['approach_dist']
delta_approach = coll_check_params['delta_approach']
table_offset = coll_check_params['table_offset']
table_mesh_filename = coll_check_params['table_mesh_filename']

# convert quality value to percent, copied from egad
scale_min = 0.0005
scale_max = 0.004
def convert_quality(q):
    max(min(q, scale_max), scale_min)
    q_percent = (q - scale_min)/(scale_max - scale_min) * 100
    return q_percent


# Write aligned grasps to pickle 
def grasp_depth_images(dir_path, mesh_file):
    #mesh_output = tempfile.mkdtemp()-
    # prepare mesh and +generate Grasps
    mesh_processor = mp.MeshProcessor(os.path.join(dir_path, mesh_file), mesh_output)
    mesh_processor.generate_graspable(grasp_config)
    
    gripper = RobotGripper.load('yumi_metal_spline', gripper_dir=grasp_config['gripper_dir'])
    sampler = AntipodalGraspSampler(gripper, grasp_config)
    obj = GraspableObject3D(mesh_processor.sdf, mesh_processor.mesh)
    grasps = sampler.generate_grasps(obj, max_iter=grasp_config['max_grasp_sampling_iters'])
    
    # setup collision checker
    collision_checker = GraspCollisionChecker(gripper)
    collision_checker.set_graspable_object(obj)
    
    # setup Grasp Quality Function
    metric_config = GraspQualityConfigFactory().create_config(egad_path['metrics']['ferrari_canny'])
    quality_fn = GraspQualityFunctionFactory.create_quality_function(obj, metric_config)
    # https://github.com/BerkeleyAutomation/dex-net/blob/0f63f706668815e96bdeea16e14d2f2b266f9262/src/dexnet/grasping/quality.py
    
    grasps_trans = list()
    
    # read in the stable poses of the mesh
    stable_poses = mesh_processor.stable_poses
    # find aligned grasps for each stable pose
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
            
            grasp_data = list()
            
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
                if (collision_free and aligned_grasp.close_fingers(obj)[0]):
                    #print(aligned_grasp.close_fingers(obj))
                    #print(aligned_grasp.close_fingers(obj)[1][1].point)
                    quality = convert_quality(quality_fn(aligned_grasp).quality)
                    #print(quality)
                    
                    # add to grasp list for single grasp and stable pose
                    grasp_data.append({"grasp_T": aligned_grasp.T_grasp_obj.matrix,
                                       "contact0": aligned_grasp.close_fingers(obj)[1][0].point,
                                       "contact1": aligned_grasp.close_fingers(obj)[1][1].point,
                                       "quality": quality})
                                       
            # add to total mesh+stableposes list        
            grasps_trans.append({"mesh": mesh_file.replace("uf.obj", "uf_proc.obj"),
                                 "table_pose": T_table_obj.matrix,
                                 "grasps": grasp_data
                                })
                                            
    
    # write pickle file
    print("found %d grasps, writing to file" % len(grasps_trans))
    with open(os.path.join(pickle_output, mesh_file.replace(".uf.obj", "") + '_aligned_grasps_list.pkl'), 'wb') as out:
            pickle.dump(grasps_trans, out)
    #get_grasp.generate_depth_images(grasps_trans, pickle_output, os.path.join(dir_path, mesh_file))

# make dirs
if not os.path.exists(pickle_output):
    os.makedirs(pickle_output)

if not os.path.exists(mesh_output):
    os.makedirs(mesh_output)

# get list of all mesh files
all_files = os.listdir(mesh_files_dir)
obj_files = [f for f in all_files if f.endswith('.obj')]

# Check and apply filter
if filter_mesh_files is None or filter_mesh_files == "":
    mesh_files = [mesh_files[1]]
else:
    mesh_files = [f for f in obj_files if f.split('_')[0] == filter_mesh_files]


print("%d mesh files found" % len(mesh_files))

for e, m in enumerate(mesh_files):
    print("%d out of %d mesh files" % (e, len(mesh_files)))
    grasp_depth_images(mesh_files_dir, m)
    subprocess.call(["rm" , os.path.join(mesh_output, "*.sdf")])
