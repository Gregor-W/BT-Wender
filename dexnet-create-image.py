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

from autolab_core import NormalCloud, PointCloud, RigidTransform
from meshpy import MaterialProperties, LightingProperties, ObjFile, VirtualCamera, ViewsphereDiscretizer, SceneObject

import numpy as np
from perception import CameraIntrinsics, RenderMode
from meshpy.mesh_renderer import ViewsphereDiscretizer, VirtualCamera
from visualization import Visualizer2D as vis

output = "/home/ubuntu/egad/depth-images/"
temp_dir = "/tmp/obj-files"

# Use local config file
config_path = '/home/ubuntu/egad/egad/scripts/cfg/dexnet_api_settings.yaml'
config = YamlConfig(str(config_path))
render_config = YamlConfig("/home/ubuntu/egad/config/dexnet-settings.yaml")

# Requires data from the dexnet project.
os.chdir(str(Path(dexnet.__file__).resolve().parent.parent.parent))

def grasp_depth_images(dir_path, mesh_file):
    # mp_cache = tempfile.mkdtemp()
    try:
        #mesh_processor = mp.MeshProcessor(os.path.join(dir_path, mesh_file), temp_dir)
        #mesh_processor.generate_graspable(config)
        # shutil.rmtree(mp_cache)
        #obj = GraspableObject3D(mesh_processor.sdf, mesh_processor.mesh)
        # grasps = sampler.generate_grasps(obj, max_iter=config['max_grasp_sampling_iters'])
        save_file_path = os.path.join(output, mesh_file.replace('.uf.obj', ''))
        #renderImages(save_file_path, obj.mesh, mesh_processor.stable_poses)
        wirtePointCloud(save_file_path, os.path.join(dir_path, mesh_file))
    except ValueError:
        print("failed for: " + mesh_file)
        # shutil.rmtree(mp_cache)
    
    
def renderImages(file_path, mesh, stable_poses):
    # setup virtual camera
    width = render_config['width']
    height = render_config['height']
    f = render_config['focal']
    cx = float(width) / 2
    cy = float(height) / 2
    ci = CameraIntrinsics('camera', fx=f, fy=f, cx=cx, cy=cy, height=height, width=width)
    vp = ViewsphereDiscretizer(min_radius=render_config['min_radius'],
                                       max_radius=render_config['max_radius'],
                                       num_radii=render_config['num_radii'],
                                       min_elev=render_config['min_elev']*np.pi,
                                       max_elev=render_config['max_elev']*np.pi,
                                       num_elev=render_config['num_elev'],
                                       num_az=render_config['num_az'],
                                       num_roll=render_config['num_roll'])
    vc = VirtualCamera(ci)
    
    image_number = 0

    for tf in [vp.object_to_camera_poses()[0]]:
        rendered_images = vc.wrapped_images_viewsphere(mesh, vp, RenderMode.COLOR, stable_poses[0])
        
        # COLOR, DEPTH, SCALED_DEPTH
        
        # rendered_images = vc.wrapped_images(mesh, [vp.object_to_camera_poses()[0]], RenderMode.COLOR, stable_poses[0])
        # rendered_images = vc.wrapped_images(mesh, vp.object_to_camera_poses(), RenderMode.COLOR, stable_poses[0])
        for r_image in rendered_images:
            image_number += 1
            save_name = file_path + '_RI1_' + str(image_number) + '.png'
            print("Writing image: " + save_name)
            r_image.image.save(save_name)


def renderImages2(file_path, mesh_filename):
    # setup virtual camera
    width = render_config['width']
    height = render_config['height']
    f = render_config['focal']
    cx = float(width) / 2
    cy = float(height) / 2
    ci = CameraIntrinsics('camera', fx=f, fy=f, cx=cx, cy=cy, height=height, width=width)
    
    orig_mesh = ObjFile(mesh_filename).read()
    mesh = orig_mesh.subdivide(min_tri_length=0.01)
    mesh.compute_vertex_normals()
    stable_poses = mesh.stable_poses()
    
    image_number = 0
    for stable_pose in stable_poses:
        T_obj_world = mesh.get_T_surface_obj(stable_pose.T_obj_table).as_frames('obj', 'world')
        virtual_camera  = VirtualCamera(ci)
        T_light_camera = RigidTransform(translation=[0,0,0],
                                        from_frame='light',
                                        to_frame=ci.frame)
        light_props = LightingProperties(ambient=-0.25,
                                         diffuse=1,
                                         specular=0.25,
                                         T_light_camera=T_light_camera,
                                         cutoff=180)
        mat_props = MaterialProperties(color=(249,241,21),
                                       ambient=0.5,
                                       diffuse=1.0,
                                       specular=1,
                                       shininess=0)
        cam_dist = 0.6
        T_camera_world = RigidTransform(rotation=np.array([[0, 1, 0],
                                                           [1, 0, 0],
                                                           [0, 0, -1]]),
                                        translation=[0,0,cam_dist],
                                        from_frame=ci.frame,
                                        to_frame='world')
        T_obj_camera = T_camera_world.inverse() * T_obj_world
        renders = virtual_camera.wrapped_images(mesh,
                                                [T_obj_camera],
                                                RenderMode.COLOR,
                                                mat_props=mat_props,
                                                light_props=light_props,
                                                debug=False)
        for r_image in renders:
            image_number += 1
            save_name = file_path + '_RI2_' + str(image_number) + '.png'
            print("Writing image: " + save_name)
            r_image.image.save(save_name)
        

def renderImages3(file_path, mesh_filename):
    orig_mesh = ObjFile(mesh_filename).read()
    # orig_mesh = ObjFile("/home/ubuntu/egad/obj-files/test-mesh.obj").read()
    mesh = orig_mesh.subdivide(min_tri_length=0.01)
    mesh.compute_vertex_normals()
    stable_poses = mesh.stable_poses()

    # setup virtual camera
    width = render_config['width']
    height = render_config['height']
    f = render_config['focal']
    cx = float(width) / 2
    cy = float(height) / 2
    ci = CameraIntrinsics('camera', fx=f, fy=f, cx=cx, cy=cy, height=height, width=width)
    vp = ViewsphereDiscretizer(min_radius=render_config['min_radius'],
                                       max_radius=render_config['max_radius'],
                                       num_radii=render_config['num_radii'],
                                       min_elev=render_config['min_elev']*np.pi,
                                       max_elev=render_config['max_elev']*np.pi,
                                       num_elev=render_config['num_elev'],
                                       num_az=render_config['num_az'],
                                       num_roll=render_config['num_roll'])
    vc = VirtualCamera(ci)

    image_number = 0

    for stable_pose in stable_poses:
        # rendered_images = vc.wrapped_images_viewsphere(mesh, vp, RenderMode.COLOR, stable_poses[0])
        
        # COLOR, DEPTH, SCALED_DEPTH
        
        rendered_images = vc.wrapped_images(mesh, [vp.object_to_camera_poses()[0]], RenderMode.DEPTH, stable_pose)
        # rendered_images = vc.wrapped_images(mesh, vp.object_to_camera_poses(), RenderMode.COLOR, stable_poses[0])
        for r_image in rendered_images:
            image_number += 1
            save_name = file_path + '_RI3_' + str(image_number) + '.png'
            print("Writing image: " + save_name)
            r_image.image.to_color(normalize=True).save(save_name)

def wirtePointCloud(file_path, mesh_filename):
    orig_mesh = ObjFile(mesh_filename).read()
    # orig_mesh = ObjFile("/home/ubuntu/egad/obj-files/test-mesh.obj").read()
    mesh = orig_mesh.subdivide(min_tri_length=0.01)
    mesh.compute_vertex_normals()
    stable_poses = mesh.stable_poses()

    # setup virtual camera
    width = render_config['width']
    height = render_config['height']
    f = render_config['focal']
    cx = float(width) / 2
    cy = float(height) / 2
    ci = CameraIntrinsics('camera', fx=f, fy=f, cx=cx, cy=cy, height=height, width=width)
    vp = ViewsphereDiscretizer(min_radius=render_config['min_radius'],
                                       max_radius=render_config['max_radius'],
                                       num_radii=render_config['num_radii'],
                                       min_elev=render_config['min_elev']*np.pi,
                                       max_elev=render_config['max_elev']*np.pi,
                                       num_elev=render_config['num_elev'],
                                       num_az=render_config['num_az'],
                                       num_roll=render_config['num_roll'])
    vc = VirtualCamera(ci)

    image_number = 0

    for stable_pose in stable_poses:
        rendered_images = vc.wrapped_images(mesh, [vp.object_to_camera_poses()[0]], RenderMode.DEPTH, stable_pose)
        for r_image in rendered_images:
            image_number += 1
            save_name = file_path + '_PC_' + str(image_number) + '.pcd'
            
            # create point cloud
            pc = r_image.image.point_normal_cloud(ci)
            pc.remove_zero_points()
            points = pc.points
            with open(save_name, "w") as f:
                write_header(f)
                f.write("POINTS " + str() + "\n")
                for i, (x, y, z) in enumerate(zip(points.x_coords, points.y_coords, points.z_coords)):
                    f.write("%f %f %f %f %i\n" % (x * 10000, y * 10000, z * 10000, 0, i))

       
def write_header(file):
    file.write("# .PCD v.7 - Point Cloud Data file format\n")
    file.write("FIELDS x y z rgb index\n")
    file.write("SIZE 4 4 4 4 4\n")
    file.write("TYPE F F F F U\n")
    file.write("COUNT 1 1 1 1 1\n")
    file.write("DATA ascii\n")
   

mesh_files_dir = "/home/ubuntu/egad/output/1589817197/pool"

# Cache directory
#mp_cache = tempfile.mkdtemp()
#mesh_processor = mp.MeshProcessor(mesh_file, mp_cache)
#mesh_processor.generate_graspable(config)
#shutil.rmtree(mp_cache)

#gripper = RobotGripper.load('yumi_metal_spline', gripper_dir=config['gripper_dir'])
#sampler = AntipodalGraspSampler(gripper, config)
#obj = GraspableObject3D(mesh_processor.sdf, mesh_processor.mesh)

#grasps = sampler.generate_grasps(obj, max_iter=config['max_grasp_sampling_iters'])

#database = dexnet.database.Hdf5Database('/home/ubuntu/egad/database/database-di.hdf5')
#dataset = dexnet.database.Hdf5Dataset("DepthImages", None, cache_dir="~/egad/cache")

#db = dexnet.database.Hdf5Database("/home/ubuntu/egad/database/example.hdf5", "READ_WRITE")
#dataset = db.create_dataset("Test1234", [])
#dataset.create_graspable("obj0", mesh_processor.mesh, mesh_processor.sdf, mesh_processor.stable_poses)

# dataset.grasps("obj0")
# dataset.rendered_images("obj0")
# dataset.store_rendered_images("obj0")

# path for obj data /egad/output/1589817197/pool


all_files = os.listdir(mesh_files_dir)
obj_files = [f for f in all_files if f.endswith('.obj')]

filter_mesh_files = '0024'
mesh_files = [f for f in obj_files if f.split('_')[0] == filter_mesh_files]
mesh_files = [mesh_files[0]]

for m in mesh_files:
    grasp_depth_images(mesh_files_dir, m)

#images = vc.images(obj.mesh, [vp.object_to_camera_poses()[0]], None, None, False)
#w_images = vc.wrapped_images(obj.mesh, [vp.object_to_camera_poses()[0]], RenderMode.DEPTH, mesh_processor.stable_poses[0])
#w_images[0].image.save('/home/ubuntu/tmp/test0.png')
