#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

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

import random
import pickle
import pathlib
import datetime
import argparse
# config file base paths
DEXNET_PATH = "/home/co/dexnet/"
EGAD_PATH = "/home/co/egad/"
FILTER_MESH_FILES = ""
STABLE_POSE_MIN_P = 0

# convert grasp quality value to percent, copied from egad
scale_min = 0.0005
scale_max = 0.004
def convert_quality(q):
    max(min(q, scale_max), scale_min)
    q_percent = (q - scale_min)/(scale_max - scale_min) * 100
    return q_percent

# Helper class for StablePoseData to save grasp data to pickle
# grasp_T: numpy_array 4x3      Transformation Matrix grasp frame to obj frame
# contact0/1: numpy_array 3x1   Contact Points for Gripper
# quality: float                Graspquality
class GraspData:
    def __init__(self, grasp_T, contact0, contact1, quality):
        self.grasp_T = grasp_T
        self.contact0 = contact0
        self.contact1 = contact1
        self.quality = quality
    
    # return dictionary to write to pickle
    def get_dict(self):
        return {"grasp_T":  self.grasp_T,
                "contact0": self.contact0,
                "contact1": self.contact1,
                "quality":  self.quality}
                
                
# Class to save grasp data to pickle
# mesh_file: string                 mesh file name
# T_table_obj: numpy_array 4x3      stable pose, Transformation Matrix table frame to obj frame
# grasps: list of GraspData         list of grasps for stable pose
class StablePoseData:
    @classmethod
    # save list of StablePoseData as pickle file
    def save_grasps(cls, spd_list, pickle_out):
        if len(spd_list):
            # convert to dict
            dict_data = []
            for spd in spd_list:
                dict_data.append(spd.get_dict())
            
            # get meshname
            meshname = spd_list[0].mesh_file.replace(".uf", "").replace("_proc", "").replace(".obj", "")
            with open(os.path.join(pickle_out, meshname + '_aligned_grasps_list.pkl'), 'wb') as out:
                    pickle.dump(dict_data, out)
        else:
            print("List of StablePoseData is empty")

    def __init__(self, mesh_file, T_table_obj, grasps):
        self.mesh_file = mesh_file
        self.T_table_obj = T_table_obj
        self.grasps = grasps
    
    # return dictionary to write to pickle
    def get_dict(self):
        grasps_dict = []
        for g in self.grasps:
            grasps_dict.append(g.get_dict())
    
        return {"mesh":  self.mesh_file,
                "stable_pose": self.T_table_obj,
                "grasps":  grasps_dict}
    
    

# Get StablePoseData and processed 3D Object using DEXNET
# mesh_out: string           mesh output dictionary
# sp_min_p: float            stable pose probability
# egad_path: string          path to egad config
# dexnet_path: string        path to dexnet config
class DexnetGetGraspdata:
    def __init__(self, mesh_out, sp_min_p, egad_path, dexnet_path):
        ## VARIABLES
        # min stable pose probability
        self.stable_pose_min_p = sp_min_p
        
        # mesh output dir
        self.mesh_out = mesh_out
        
        ## SETUP CONFIG
        self.egad_config = None
        self.table_offset = None
        self.table_mesh_filename = None
        self.approach_dist = None
        self.delta_approach = None 
        self.max_grasp_approach_table_angle = None
        self.phi_offsets = None
        
        self.setup_config(egad_path, dexnet_path)
        
        ## DEXNET SETUP
        # setup collision checker and sampler
        os.chdir(str(Path(dexnet.__file__).resolve().parent.parent.parent))
        self.gripper = RobotGripper.load('yumi_metal_spline', gripper_dir=self.egad_config['gripper_dir'])
        self.sampler = AntipodalGraspSampler(self.gripper, self.egad_config)
        # setup Grasp Quality Function
        self.metric_config = GraspQualityConfigFactory().create_config(self.egad_config['metrics']['ferrari_canny'])
        # has to be reinitiated  to avoid errors
        self.collision_checker  = None
        
        self.grasp_data_list = None
        
    # setup config
    def setup_config(self, egad_path, dexnet_path):
        # Use local config file
        self.egad_config = YamlConfig(os.path.join(egad_path, "scripts/cfg/dexnet_api_settings.yaml"))
        
        # setup collision params for alignment
        coll_vis_config = YamlConfig(os.path.join(dexnet_path, "cfg/tools/generate_gqcnn_dataset.yaml"))
        coll_check_params = coll_vis_config['collision_checking']
        
        self.table_offset = coll_check_params['table_offset']
        self.table_mesh_filename = coll_check_params['table_mesh_filename']
        self.approach_dist = coll_check_params['approach_dist']
        self.delta_approach = coll_check_params['delta_approach']
        
        # paramet for alignement of grasps
        table_alignment_params = coll_vis_config['table_alignment']
        min_grasp_approach_offset = -np.deg2rad(table_alignment_params['max_approach_offset'])
        max_grasp_approach_offset = np.deg2rad(table_alignment_params['max_approach_offset'])
        num_grasp_approach_samples = table_alignment_params['num_approach_offset_samples']
        
        self.max_grasp_approach_table_angle = np.deg2rad(table_alignment_params['max_approach_table_angle'])
        self.phi_offsets = []
        
        if max_grasp_approach_offset == min_grasp_approach_offset:
            phi_inc = 1
        elif num_grasp_approach_samples == 1:
            phi_inc = max_grasp_approach_offset - min_grasp_approach_offset + 1
        else:
            phi_inc = (max_grasp_approach_offset - min_grasp_approach_offset) / (num_grasp_approach_samples - 1)
                                                                
        phi = min_grasp_approach_offset
        while phi <= max_grasp_approach_offset:
            self.phi_offsets.append(phi)
            phi += phi_inc
    
    # get list of grasps for given stable_pose and obj
    def get_grasps(self, stable_pose, obj):
        # align grasps with the table
        grasps = self.sampler.generate_grasps(obj, max_iter=self.egad_config['max_grasp_sampling_iters'])
        quality_fn = GraspQualityFunctionFactory.create_quality_function(obj, self.metric_config)
        # https://github.com/BerkeleyAutomation/dex-net/blob/0f63f706668815e96bdeea16e14d2f2b266f9262/src/dexnet/grasping/quality.py
        
        aligned_grasps = [grasp.perpendicular_table(stable_pose) for grasp in grasps]
        
        self.grasp_data_list = list()
        
        # check grasp validity
        for aligned_grasp in aligned_grasps:
            # check angle with table plane and skip unaligned grasps
            _, grasp_approach_table_angle, _ = aligned_grasp.grasp_angles_from_stp_z(stable_pose)
            perpendicular_table = (np.abs(grasp_approach_table_angle) < self.max_grasp_approach_table_angle)
            if not perpendicular_table: 
                continue

            # check whether any valid approach directions are collision free
            collision_free = False
            for phi_offset in self.phi_offsets:
                rotated_grasp = aligned_grasp.grasp_y_axis_offset(phi_offset)
                collides = self.collision_checker.collides_along_approach(rotated_grasp, self.approach_dist, self.delta_approach)
                if not collides:
                    collision_free = True
                    break

            if (collision_free and aligned_grasp.close_fingers(obj)[0]):
                quality = convert_quality(quality_fn(aligned_grasp).quality)
                self.grasp_data_list.append(GraspData(aligned_grasp.T_grasp_obj.matrix,
                                                 aligned_grasp.close_fingers(obj)[1][0].point,
                                                 aligned_grasp.close_fingers(obj)[1][1].point,
                                                 quality))
        return self.grasp_data_list
    
    # setup table data and in collision checker, return table to obj trans matrix
    def setup_Table(self, stable_pose, obj):
        # get table trans matrix
        T_obj_stp = stable_pose.T_obj_table.as_frames('obj', 'stp')
        T_obj_table = obj.mesh.get_T_surface_obj(T_obj_stp, delta=self.table_offset).as_frames('obj', 'table')
        T_table_obj = T_obj_table.inverse()
        # setup table in collision checker
        self.collision_checker.set_table(self.table_mesh_filename, T_table_obj)
        return T_table_obj
    
    # Generate Stable Poses, Grasps and write to pickle 
    def generate_write(self, dir_path, mesh_file):
        # prepare mesh and +generate Grasps
        mesh_processor = mp.MeshProcessor(os.path.join(dir_path, mesh_file), self.mesh_out)
        mesh_processor.generate_graspable(self.egad_config)
        
        obj = GraspableObject3D(mesh_processor.sdf, mesh_processor.mesh) 
        self.collision_checker = GraspCollisionChecker(self.gripper)        
        self.collision_checker.set_graspable_object(obj)

        spd_list = list()
        
        # read in the stable poses of the mesh
        stable_poses = mesh_processor.stable_poses
        # find aligned grasps for each stable pose
        for i, stable_pose in enumerate(stable_poses):
            # render images if stable pose is valid
            if stable_pose.p > self.stable_pose_min_p:
                # add to total mesh+stableposes list
                
                T_table_obj = self.setup_Table(stable_pose, obj)
                grasp_data = self.get_grasps(stable_pose, obj)
                spd_list.append(StablePoseData(mesh_file.replace(".obj", "_proc.obj"),
                                                   T_table_obj.matrix,
                                                   grasp_data))

        
        # write pickle file
        print("found %d grasps, writing to file" % len(spd_list))
        return spd_list

# python main
def run():
    # commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, nargs=1, help="Base folder, including EGAD output folder")
    parser.add_argument("--limit", help="limit amount of used EGAD meshes", type=int, nargs='?')
    
    args = parser.parse_args()
    limit = args.limit
    
    # get egad
    base_dir = args.directory[0]
    egad_input = os.path.join(base_dir, "egad-output")
    folder = sorted(os.listdir(egad_input))[-1]
    
    downloaded = False
    # find mesh folder
    if os.path.exists(os.path.join(egad_input, folder, "pool")):
        mesh_files_dir = os.path.join(egad_input, folder, "pool")
    else:
        # for downloaded dataset
        mesh_files_dir = os.path.join(egad_input, folder)#
        downloaded = True

    # get list of all mesh files
    all_files = os.listdir(mesh_files_dir)
    obj_files = [f for f in all_files if f.endswith('.obj')]
    
    # output folders
    #output = os.path.join(base_output, datetime.datetime.now().strftime('%y%m%d_%H%M'))
    output = os.path.join(base_dir, "grasp-data")
    pickle_output = os.path.join(output, "pickle-files")
    mesh_output = os.path.join(output, "object-files")
    
    print("using meshes from EGAD output %s, writing to: %s" %(mesh_files_dir, output))
    
    # setup output folders
    # check if resume or new
    if not os.path.exists(output):
        os.makedirs(output)

        # make dirs
        if not os.path.exists(pickle_output):
            os.makedirs(pickle_output)

        if not os.path.exists(mesh_output):
            os.makedirs(mesh_output)
  
    # Check and apply filter
    if FILTER_MESH_FILES is None or FILTER_MESH_FILES == "":
        mesh_files = obj_files
    else:
        mesh_files = [f for f in obj_files if f.split('_')[0] == filter_mesh_files]

    # Limit mesh files
    if limit is not None and limit < len(mesh_files):
        if downloaded:
            random.shuffle(mesh_files)
            mesh_files = mesh_files[0:limit]
        else:
            mesh_files = sorted(mesh_files)[0:limit]
    
    print("total %d mesh files found" % len(mesh_files))
    
    # check existing filenames for egad object names
    already_done = [f[0:8] for f in os.listdir(pickle_output)]
    mesh_files = [f for f in mesh_files if f[0:8] not in already_done]

    # start grasp creation
    Grasp_generator = DexnetGetGraspdata(mesh_output, STABLE_POSE_MIN_P, EGAD_PATH, DEXNET_PATH)
    
    for e, m in enumerate(mesh_files):
        print("%d out of %d mesh files" % (e + 1, len(mesh_files)))
        grasp_data = Grasp_generator.generate_write(mesh_files_dir, m)
        StablePoseData.save_grasps(grasp_data, pickle_output)
if __name__ == "__main__":
    run()