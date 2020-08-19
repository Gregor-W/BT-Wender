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

import pickle

output = "/home/ubuntu/egad/grasp-data/test-object0"
temp_dir = "/tmp/obj-files"

# Use local config file
#config_path = '/home/ubuntu/egad/egad/scripts/cfg/dexnet_api_settings.yaml'
config_path = '/home/ubuntu/test-dexnet/test/config.yaml'
config = YamlConfig(str(config_path))
render_config = YamlConfig("/home/ubuntu/egad/config/dexnet-settings.yaml")

# Requires data from the dexnet project.
os.chdir(str(Path(dexnet.__file__).resolve().parent.parent.parent))

# Create different depth files
def grasp_depth_images(dir_path, mesh_file):
    try:
        #mp_cache = tempfile.mkdtemp()
        
        mp_cache = '/home/ubuntu/egad/grasp-data/object-files'
        
        # generate Grasps
        mesh_processor = mp.MeshProcessor(os.path.join(dir_path, mesh_file), mp_cache)
        mesh_processor.generate_graspable(config)

        gripper = RobotGripper.load('yumi_metal_spline', gripper_dir=config['gripper_dir'])
        sampler = AntipodalGraspSampler(gripper, config)
        obj = GraspableObject3D(mesh_processor.sdf, mesh_processor.mesh)

        grasps = sampler.generate_grasps(obj, max_iter=config['max_grasp_sampling_iters'])
        
        grasps_trans = list()
        for e, g in enumerate(grasps):
            grasps_trans.append((g.T_grasp_obj, g.axis, g.close_width))
            
        with open(os.path.join(output, 'grasp_list' '.pkl'), 'wb') as out:
            pickle.dump(grasps_trans, out)
        
        stable_poses = list()
        for sp in mesh_processor.stable_poses:
            stable_poses.append((sp.T_obj_table, sp.p))
        
        with open(os.path.join(output, 'stable_poses_list' '.pkl'), 'wb') as out:
            pickle.dump(stable_poses, out)
        
        # stable_grasps = sampler.generate_grasps_stable_poses(obj, mesh_processor.stable_poses, max_iter=config['max_grasp_sampling_iters'])

        
        #save_file_path = os.path.join(output, mesh_file.replace('.uf.obj', ''))
        #renderImages(save_file_path, os.path.join(dir_path, mesh_file), mp_cache)
        #renderImages2(save_file_path, os.path.join(dir_path, mesh_file))
        #renderImages3(save_file_path, os.path.join(dir_path, mesh_file))
        #renderImages4(save_file_path, os.path.join(dir_path, mesh_file))
        #wirtePointCloud(save_file_path, os.path.join(dir_path, mesh_file))
        #shutil.rmtree(mp_cache)
    except ValueError:
        print("failed for: " + mesh_file)
        #shutil.rmtree(mp_cache)

    

# create mesh object from file_path
def createMesh1(mesh_filename):
    orig_mesh = ObjFile(mesh_filename).read()
    #mesh = orig_mesh.subdivide(min_tri_length=0.01)
    mesh = orig_mesh.subdivide(min_tri_length=0.5)
    mesh.compute_vertex_normals()
    stable_poses = mesh.stable_poses()
    return mesh, stable_poses
    

# create mesh object from file_path using dexnet meshprocessor
def createMesh2(mesh_filename, mp_cache):
    mesh_processor = mp.MeshProcessor(mesh_filename, mp_cache)
    mesh, sdf, stable_poses = mesh_processor.generate_graspable(config)
    obj = GraspableObject3D(mesh_processor.sdf, mesh_processor.mesh)
    #grasps = sampler.generate_grasps(obj, max_iter=config['max_grasp_sampling_iters'])
    # return obj.mesh, mesh_processor.stable_poses
    return mesh, stable_poses


# setup virtual camera and viewsphere
def setupCamera():
    # setup virtual camera
    width = render_config['width'] * 2
    height = render_config['height'] * 2
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
    return vp, vc, ci


# render in color using default setup
def renderImages(file_path, mesh_filename, cache):
    mesh, stable_poses = createMesh2(mesh_filename, cache)
    vp, vc, ci = setupCamera()
    
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


# render color with light and material properties
def renderImages2(file_path, mesh_filename):
    mesh_filename = '/home/ubuntu/test-dexnet/data/meshes/table.obj'
    mesh, stable_poses = createMesh1(mesh_filename)
    vp, virtual_camera, ci = setupCamera()
    
    image_number = 0
    
    """
    table_mesh = ObjFile('/home/ubuntu/test-dexnet/data/meshes/table.obj').read()
    table_mesh = table_mesh.subdivide()
    table_mesh.compute_vertex_normals()
    table_mat_props = MaterialProperties(color=(0,255,0),
                                         ambient=0.5,
                                         diffuse=1.0,
                                         specular=1,
                                         shininess=0)
    """
    
    for stable_pose in stable_poses:
        T_obj_world = mesh.get_T_surface_obj(stable_pose.T_obj_table).as_frames('obj', 'world')
        
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
        """
        scene_objs = {'table': SceneObject(table_mesh, T_obj_world.inverse(),
                                           mat_props=table_mat_props)}
        for name, scene_obj in scene_objs.iteritems():
            virtual_camera.add_to_scene(name, scene_obj)                               
        """
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
                                                RenderMode.SCALED_DEPTH,
                                                mat_props=mat_props,
                                                light_props=light_props,
                                                debug=False)
        for r_image in renders:
            image_number += 1
            save_name = file_path + '_RI2_' + str(image_number) + '.png'
            print("Writing image: " + save_name)
            r_image.image.save(save_name)
        

# render as depth using default setup
def renderImages3(file_path, mesh_filename):
    mesh, stable_poses = createMesh1(mesh_filename)
    vp, vc, ci = setupCamera()

    image_number = 0
    for stable_pose in stable_poses:
        # COLOR, DEPTH, SCALED_DEPTH
        
        rendered_images = vc.wrapped_images(mesh, [vp.object_to_camera_poses()[0]], RenderMode.DEPTH, stable_pose)
        for r_image in rendered_images:
            image_number += 1
            save_name = file_path + '_RI3_' + str(image_number) + '.png'
            print("Writing image: " + save_name)
            r_image.image.inpaint().to_color(normalize=True).save(save_name)


# render as scaled depth using default setup
def renderImages4(file_path, mesh_filename):
    mesh, stable_poses = createMesh1(mesh_filename)
    vp, vc, ci = setupCamera()

    image_number = 0
    for stable_pose in stable_poses:
        # COLOR, DEPTH, SCALED_DEPTH
        
        rendered_images = vc.wrapped_images(mesh, [vp.object_to_camera_poses()[0]], RenderMode.SCALED_DEPTH, stable_pose)
        for r_image in rendered_images:
            image_number += 1
            save_name = file_path + '_RI4_' + str(image_number) + '.png'
            print("Writing image: " + save_name)
            r_image.image.save(save_name)


# write point cloud file (.pcd)
def wirtePointCloud(file_path, mesh_filename):
    mesh, stable_poses = createMesh1(mesh_filename)
    vp, virtual_camera, ci = setupCamera()
    
    image_number = 0
    for stable_pose in stable_poses:
        T_obj_world = mesh.get_T_surface_obj(stable_pose.T_obj_table).as_frames('obj', 'world')
        
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
                                                RenderMode.DEPTH,
                                                mat_props=mat_props,
                                                light_props=light_props,
                                                debug=False)
        for r_image in renders:
            image_number += 1
            save_name = file_path + '_PC_' + str(image_number) + '.pcd'
            
            # create point cloud
            pc = r_image.image.point_normal_cloud(ci)
            pc.remove_zero_points()
            points = pc.points
            with open(save_name, "w") as f:
                write_header(f)
                f.write("POINTS " + str(pc.num_points) + "\n")
                #for i, (x, y, z) in enumerate(zip(points.x_coords, points.y_coords, points.z_coords)):
                #    f.write("%f %f %f %f %i\n" % (x, y, z, 0, i))
                for i in range(0, pc.num_points):
                    #print(pc.__getitem__(i))
                    x = pc.__getitem__(i)[0].x
                    y = pc.__getitem__(i)[0].y
                    z = pc.__getitem__(i)[0].z
                    f.write("%f %f %f %f %i\n" % (x, y, z, 0, i))

# write header for point cloud file
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

#images = vc.images(obj.mesh, [vp.object_to_camera_poses()[0]], None, None, False)
#w_images = vc.wrapped_images(obj.mesh, [vp.object_to_camera_poses()[0]], RenderMode.DEPTH, mesh_processor.stable_poses[0])
#w_images[0].image.save('/home/ubuntu/tmp/test0.png')
