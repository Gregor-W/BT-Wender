import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import argparse
import os
import sys
import pickle
import cv2

IMG_W = 640
IMG_H = 480
DIST = 0.15
GRASP_WIDTH = 0.01
MIN_QUALITY = 70
DEBUG = True

# sort points couterclockwise
def sort(ps, center):
    sorted_ps = [("A", ps[0], angle(ps[0], center)),
                 ("A", ps[1], angle(ps[1], center)),
                 ("B", ps[2], angle(ps[2], center)),
                 ("B", ps[3], angle(ps[3], center))]
    
    sorted_ps = sorted(sorted_ps, key=lambda a: a[2]) 
    
    # make sure first two points are the same gripper finger
    if sorted_ps[0][0] == sorted_ps[1][0]:
        return [p[1] for p in sorted_ps]
    else:
        sorted_ps = [sorted_ps[-1]] + sorted_ps[:-1]
    
    if sorted_ps[0][0] == sorted_ps[1][0]:
        return [p[1] for p in sorted_ps]
    else:
        print("ERROR points couldn't be sorted")
        return None
        
# get angle to x axis with C as center
def angle(P, C):
    x = P[0] - C[0]
    y = P[1] - C[1]
    return np.rad2deg(np.arctan2(y, x))


class GraspRender:
    def __init__(self, mesh_path, output_path, img_w=IMG_W, img_h=IMG_H, cam_dist=DIST,
                 grasp_width=GRASP_WIDTH, min_grasp_quality=MIN_QUALITY):
        # create single figure here to be reused later to avoid GC issues
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # set instance variables
        self.mesh_path = mesh_path
        self.output_path = output_path
        self.img_w = img_w
        self.img_h = img_h
        self.dist = cam_dist
        self.rectwidth = grasp_width
        self.min_quality = min_grasp_quality
        
        # create unset instance variables
        self.number = 0
        self.proj_matrix = None
        self.camera_pose = None
        self.obj_pose = None
        self.scene = None
        self.mesh = None
        self.grasp_points = list()
        self.table_mesh_path = os.path.join(sys.path[0], 'table.obj')
        
        
    # convert point in obj coords to point on image
    def convert_object_point_to_img(self, s_point):
        # point to world coords
        point = self.obj_pose.dot(s_point)
        # point to camera coords
        cam_point = np.linalg.inv(self.camera_pose).dot(point)
        # point to projection matrix
        twoD_point = self.proj_matrix.dot(cam_point)

        x = (twoD_point[0] / twoD_point[3] + 1) * 0.5 * self.img_w
        y = (-twoD_point[1] / twoD_point[3] + 1) * 0.5 * self.img_h

        return x, y
    
    
    # create pyrender scene with current mesh
    def create_scene(self):
        self.scene = pyrender.Scene()

        # load table
        table_trimesh = trimesh.load(self.table_mesh_path)
        table_mesh =  pyrender.Mesh.from_trimesh(table_trimesh)
        self.scene.add(table_mesh)
        
        # setup camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        self.camera_pose = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, self.dist],
            [0, 0, 0, 1]
        ])
        nc = pyrender.Node(camera=camera, matrix=self.camera_pose)
        self.scene.add_node(nc)
        # get camera projection matrix
        self.proj_matrix = camera.get_projection_matrix(self.img_w, self.img_h)
        
        # add light
        light = pyrender.SpotLight(color=np.ones(3), intensity=1.0,
                                   innerConeAngle=np.pi/16.0,
                                   outerConeAngle=np.pi/6.0)
        light_pose = np.array([
            [1, 0, 0, 0.2],
            [0, 1, 0, 0.2],
            [0, 0, 1, 1],
            [0, 0, 0, 1]
        ])
        self.scene.add(light, pose=light_pose)
        
        # load mesh
        mesh_file = os.path.join(self.mesh_path, self.mesh)
        fuze_trimesh = trimesh.load(mesh_file)
        mesh_file = pyrender.Mesh.from_trimesh(fuze_trimesh)
        self.scene.add(mesh_file, pose=self.obj_pose)
    
    
    # get 2D points
    def get_points(self, T_grasp_obj, contact_p0, contact_p1):
        # grasps
        s_point = np.array([row[3] for row in T_grasp_obj])

        width = np.linalg.norm(np.array(contact_p0) - np.array(contact_p1))

        points = list()
        # create grasping rectangle
        points.append(T_grasp_obj.dot([0, 0.75 * width, -self.rectwidth, 0]) + s_point)
        points.append(T_grasp_obj.dot([0, 0.75 * width, self.rectwidth, 0]) + s_point)
        points.append(T_grasp_obj.dot([0, -0.75 * width, self.rectwidth, 0]) + s_point)
        points.append(T_grasp_obj.dot([0, -0.75 * width, -self.rectwidth, 0]) + s_point)

        # convert to img coords
        points_conv = [self.convert_object_point_to_img(p) for p in points]
        # center point for sorting
        s_point_conv = self.convert_object_point_to_img(s_point)
        
        # sort points
        points_conv = sort(points_conv, s_point_conv)
       
        return points_conv
        
    # render images
    def render_write(self, show_points=False):
        output_file = self.get_output_filename()
    
        r = pyrender.OffscreenRenderer(self.img_w, self.img_h)
        color, depth = r.render(self.scene)
        # render scene
        color, depth = r.render(self.scene)
        
        # write depth
        sc = list()        
        cv2.imwrite(output_file + "d.tiff", depth)
        
        # write color
        plt.imshow(color) 
        
        # add grasp points
        if show_points:
            for p in self.grasp_points[0:4]:
               sc.append(plt.scatter(p[0], p[1]))           
        plt.savefig(output_file + "r.png")
        plt.clf()
        
        # write positive grasp txt file
        with open(output_file + "cpos.txt", "w") as f: 
            for p in self.grasp_points:
                f.write("%.3f" % p[0])
                f.write(" ")
                f.write("%.3f" % p[1])
                f.write("\n")
    
    
    # get filename for output
    def get_output_filename(self):
        self.number += 1
        # create name for output file, everything after "pcd" has to be a number or underscore for GGCNN
        if self.mesh[0].isnumeric():
            output_img = "pcd" + self.mesh.replace(".uf_proc.obj", "_")
        else:
            output_img = "pcd" + self.mesh[1:].replace("_proc.obj", "_")
        # add unique number for stable pose and mesh
        output_img += format(self.number, '04d')
        return os.path.join(self.output_path, output_img)     
    
    
    # write grasp points and render
    def run(self, stable_pose_grasps, show_points=False):
        self.obj_pose = np.linalg.inv(stable_pose_grasps['stable_pose'])
        self.mesh = stable_pose_grasps['mesh']
        
        self.create_scene()
            
        # get grasp points
        grasps = stable_pose_grasps['grasps']
        render = False
        self.grasp_points = list()
        for grasp in grasps:
            if grasp["quality"] >= self.min_quality:
                # only render if good grasps are found
                render = True
                self.grasp_points.extend(self.get_points(grasp["grasp_T"], grasp["contact0"], grasp["contact1"]))
        
        if render:
            self.render_write(show_points)
            return True
        else:
            return False

            
# python main
def run():
    # commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, nargs=1, help="path to base directory with grasp-data folder and egad-output")
    
    args = parser.parse_args()
    
    # get grasp data folder
    base_path = os.path.join(args.directory[0], "grasp-data")
    
    pickle_path = os.path.join(base_path, "pickle-files")
    mesh_path = os.path.join(base_path, "object-files")
    output_path = os.path.join(base_path, "images")
    
    files = os.listdir(pickle_path)
    total_images = 1
    
    # create output folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    grasp_renderer = GraspRender(mesh_path, output_path)

    print("total pickle files: %d" % len(files))
    for e, file in enumerate(files):
        print("%d of %d" % (e,len(files)))
        print("total images: %d" % total_images)
        with open(os.path.join(pickle_path, file), 'rb') as f:
            grasp_data = pickle.load(f, encoding="latin1")


        for n, sp in enumerate(grasp_data):
            if grasp_renderer.run(sp, DEBUG):
                total_images += 1

if __name__ == "__main__":
    run()