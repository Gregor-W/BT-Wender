import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import argparse

# command arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mesh')
parser.add_argument('--pose', nargs='*', type=float)
parser.add_argument('--grasp_center', nargs='*', type=float)
parser.add_argument('--contact_p0', nargs='*', type=float)
parser.add_argument('--contact_p1', nargs='*', type=float)
parser.add_argument('--path')
args = parser.parse_args()

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

def sort(points, center):
    if check_counterclockwise(points, center):
        return points
    
    points.reverse()
    if check_counterclockwise(points, center):
        return points
    else:
        print("ERROR")



# sort points couterclockwise to write in file
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
        
            

# load mesh
fuze_trimesh = trimesh.load(args.mesh)
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
scene = pyrender.Scene()

# load table
table_trimesh = trimesh.load('~/Share/Festo/table.obj')
table_mesh =  pyrender.Mesh.from_trimesh(table_trimesh)
scene.add(table_mesh)


# stable pose
obj_pose = np.array([
    args.pose[0: 4],
    args.pose[4: 8],
    args.pose[8: 12],    
    args.pose[12: 16]
])

obj_pose = np.linalg.inv(obj_pose)

scene.add(mesh, pose=obj_pose)
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

img_w = 400
img_h = 400
dist = 0.15

camera_pose = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, dist],
    [0, 0, 0, 1]
])


# render scene
nc = pyrender.Node(camera=camera, matrix=camera_pose)
scene.add_node(nc)
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                           innerConeAngle=np.pi/16.0,
                           outerConeAngle=np.pi/6.0)

r = pyrender.OffscreenRenderer(img_w, img_h)
color, depth = r.render(scene)

# draw grasp points
pro_matrix = camera.get_projection_matrix(img_w, img_h)

s_point = np.array([args.grasp_center[3], args.grasp_center[7],
                    args.grasp_center[11], args.grasp_center[15]])


T_grasp_obj = np.array([
    args.grasp_center[0: 4],
    args.grasp_center[4: 8],
    args.grasp_center[8: 12],    
    args.grasp_center[12: 16]
])

contact_p0 = np.array(args.contact_p0)
contact_p1 = np.array(args.contact_p1)
width = np.linalg.norm(contact_p0 - contact_p1)

points = []
points.append(T_grasp_obj.dot([0, 0.75 * width, -0.01, 0]) + s_point)
points.append(T_grasp_obj.dot([0, 0.75 * width, 0.01, 0]) + s_point)
points.append(T_grasp_obj.dot([0, -0.75 * width, 0.01, 0]) + s_point)
points.append(T_grasp_obj.dot([0, -0.75 * width, -0.01, 0]) + s_point)

points_conv = [convert_object_point_to_img(p, obj_pose, camera_pose, pro_matrix) for p in points]
s_point_conv = convert_object_point_to_img(s_point, obj_pose, camera_pose, pro_matrix)

points_conv = sort(points_conv, s_point_conv)



sc = list()

# First plot
fig = plt.figure()
ax1 = plt.subplot(1, 1, 1)
ax1.axis('off')
ax1.imshow(depth, cmap=plt.cm.gray_r)
for p in points_conv:
   sc.append(ax1.scatter(p[0], p[1]))
#print(args.path)
plt.savefig(args.path)

with open(args.path.replace(".tiff", "cpos.txt"), "w") as f: 
    for p in points_conv:
        f.write("%.3f" % p[0])
        f.write(" ")
        f.write("%.3f" % p[1])
        f.write("\n")
print(pro_matrix)

