import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import argparse
import os
import pickle
import cv2

base_path = "/data/grasp-data/"
folder = sorted(os.listdir(base_path))[-1]
input_path = os.path.join(base_path, folder)

img_w = 640
img_h = 480
dist = 0.15
min_quality = 70
debug = True


# convert point in obj coords to point on image
def convert_object_point_to_img(s_point, obj_pose, camera_pose, proj_matrix):
    # point to world coords
    point = obj_pose.dot(s_point)
    # point to camera coords
    cam_point = np.linalg.inv(camera_pose).dot(point)
    # point to projection matrix
    twoD_point = proj_matrix.dot(cam_point)

    x = (twoD_point[0] / twoD_point[3] + 1) * 0.5 * img_w
    y = (-twoD_point[1] / twoD_point[3] + 1) * 0.5 * img_h
    
    return x, y


# sort points couterclockwise (points are never random, either clockwise or couterclockwise)
def sort(points, center):
    if check_counterclockwise(points, center):
        return points
    
    points.reverse()
    if check_counterclockwise(points, center):
        return points
    else:
        print("ERROR")



# check if points are couterclockwise
def check_counterclockwise(points, center):
    for n, p in enumerate(points):
        if n + 1 == len(points):
            n = -1
        # top right
        ccw = True
        if p[0] >= center[0] and p[1] < center[1]:
            ccw = ccw and points[n + 1][0] < p[0]
        # top left
        if p[0] < center[0] and p[1] <= center[1]:
            ccw = ccw and points[n + 1][1] > p[1]
        # bottom right
        if p[0] <= center[0] and p[1] > center[1]:
            ccw = ccw and points[n + 1][0] > p[0]
        # bottom left
        if p[0] > center[0] and p[1] >= center[1]:
            ccw = ccw and points[n + 1][1] < p[1]
    return ccw

       


if __name__ == "__main__":
    # commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('work_dir', type=str, nargs=1, help="Output directory for converted meshes and pickle files")
    
    args = parser.parse_args()
    base_path = args.work_dir[0]
    folder = sorted(os.listdir(base_path))[-1]
    input_path = os.path.join(base_path, folder)
    
    pickle_path = os.path.join(input_path, "pickle-files")
    mesh_path = os.path.join(input_path, "object-files")
    output_path = os.path.join(input_path, "images")
    
    
    files = os.listdir(pickle_path)
    #files = [f for f in files if os.path.basename(f).startswith("0033")]
    total_images = 0

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # create single figure here to be reused later to avoid GC issues
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    print("total pickle files: %d" % len(files))
    for e, file in enumerate(files):
        print("%d of %d" % (e,len(files)))
        print("total images: %d" % total_images)
        with open(os.path.join(pickle_path, file), 'rb') as f:
            grasp_data = pickle.load(f, encoding="latin1")


        for n, sp in enumerate(grasp_data):
            mesh = os.path.join(mesh_path, sp['mesh'])
            
            output_img = "pcd" + sp['mesh'].replace(".uf_proc.obj", "_") + format(n, '04d')        
            output_file = os.path.join(output_path, output_img)        

            obj_pose = sp['table_pose']
            obj_pose = np.linalg.inv(obj_pose)        
            
            grasps = sp['grasps']

            # load mesh
            fuze_trimesh = trimesh.load(mesh)
            mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
            scene = pyrender.Scene()

            # load table
            table_trimesh = trimesh.load('~/table.obj')
            table_mesh =  pyrender.Mesh.from_trimesh(table_trimesh)
            scene.add(table_mesh)

            

            scene.add(mesh, pose=obj_pose)
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

            camera_pose = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, dist],
                [0, 0, 0, 1]
            ])
            

            # render scene
            nc = pyrender.Node(camera=camera, matrix=camera_pose)
            scene.add_node(nc)
            light = pyrender.SpotLight(color=np.ones(3), intensity=1.0,
                                       innerConeAngle=np.pi/16.0,
                                       outerConeAngle=np.pi/6.0)
            scene.add(light, pose=camera_pose)
            r = pyrender.OffscreenRenderer(img_w, img_h)
            color, depth = r.render(scene)

            # draw grasp points
            pro_matrix = camera.get_projection_matrix(img_w, img_h)
            render = False
            grasp_points = list()
            for grasp in grasps:
                if grasp["quality"] >= min_quality:
                    render = True
                    T_grasp_obj = grasp["grasp_T"]
                    # grasps
                    s_point = np.array([row[3] for row in T_grasp_obj])



                    contact_p0 = np.array(grasp["contact0"])
                    contact_p1 = np.array(grasp["contact1"])
                    width = np.linalg.norm(contact_p0 - contact_p1)

                    points = list()
                    points.append(T_grasp_obj.dot([0, 0.75 * width, -0.01, 0]) + s_point)
                    points.append(T_grasp_obj.dot([0, 0.75 * width, 0.01, 0]) + s_point)
                    points.append(T_grasp_obj.dot([0, -0.75 * width, 0.01, 0]) + s_point)
                    points.append(T_grasp_obj.dot([0, -0.75 * width, -0.01, 0]) + s_point)


                    points_conv = [convert_object_point_to_img(p, obj_pose, camera_pose, pro_matrix) for p in points]
                    s_point_conv = convert_object_point_to_img(s_point, obj_pose, camera_pose, pro_matrix)

                    points_conv = sort(points_conv, s_point_conv)
                    grasp_points.extend(points_conv)
            
            if render:
                sc = list()        
                cv2.imwrite(output_file + "d.tiff", depth)
                
                
                plt.imshow(color) 
                if debug:
                    for p in points_conv:
                       sc.append(plt.scatter(p[0], p[1]))           
                plt.savefig(output_file + "r.png")
                plt.clf()
                total_images += 1

                with open(output_file + "cpos.txt", "w") as f: 
                    for p in grasp_points:
                        f.write("%.3f" % p[0])
                        f.write(" ")
                        f.write("%.3f" % p[1])
                        f.write("\n")
