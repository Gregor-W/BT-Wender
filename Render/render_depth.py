import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
matplotlib.use('agg')
import argparse

# command arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mesh')
parser.add_argument('--pose', nargs='*', type=float)
parser.add_argument('--grasp_center', nargs='*', type=float)
parser.add_argument('--grasp_vector', nargs='*', type=float)
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

scene = pyrender.Scene()

# generate table
#plane = trimesh.creation.box(extends=[0.02, 0.02, 0.02])
#plane.visual.face_colors = [1, 0, 0.5, 0.5]
#plane.fix_normals()
#plane.export("table_mesh.obj")
#table_obj = trimesh.load("table_mesh.obj")
#table_mesh = pyrender.Mesh.from_trimesh(plane.smoothed(), smooth=False)
#table_mesh = pyrender.Mesh.from_trimesh(plane)
#table_mesh = pyrender.Mesh.from_trimesh(table_obj)
#scene.add(table_mesh)

#table_pose = np.array([
#    [1,0,0,0],
#    [0,0,1,-0.03],
#    [0,-1,0,0],
#    [0,0,0,1]
#])

#cyl = trimesh.creation.cylinder(1, height=0.02)
#cyl_mesh = pyrender.Mesh.from_trimesh(cyl)
#scene.add(cyl_mesh, pose=table_pose)


table_trimesh = trimesh.load('~/Share/Festo/table.obj')
table_mesh =  pyrender.Mesh.from_trimesh(table_trimesh)
scene.add(table_mesh)


# load mesh
fuze_trimesh = trimesh.load(args.mesh)
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)

# stable pose
T_obj_table = np.array([
    args.pose[0: 4],
    args.pose[4: 8],
    args.pose[8: 12],    
    args.pose[12: 16]
])

#obj_pose = T_obj_table
obj_pose = np.linalg.inv(T_obj_table)

scene.add(mesh, pose=obj_pose)
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

img_w = 400
img_h = 400
alpha = 0
height = 0.1 #0.1
dist = 0

# rotate around y
camera_pose = np.array([
    [np.cos(alpha), 0, np.sin(alpha), np.sin(alpha) * height],
    [0, 1, 0, 0],
    [-np.sin(alpha), 0, np.cos(alpha), np.cos(alpha) * height],
    [0, 0, 0, 1]
])


#rotate around z
#camera_pose = np.array([
#    [np.cos(alpha), np.sin(alpha), 0, np.sin(alpha) * dist],
#    [-np.sin(alpha), np.cos(alpha), 0, np.cos(alpha) * dist],
#    [0, 0, 1, height],
#    [0, 0, 0, 1]
#])




light_pose = np.array([
    [1,0,0,0.2],
    [0,1,0,0],
    [0,0,1,1],
    [0,0,0,1]
])

# render scene
nc = pyrender.Node(camera=camera, matrix=camera_pose)
scene.add_node(nc)
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                           innerConeAngle=np.pi/16.0,
                           outerConeAngle=np.pi/6.0)
scene.add(light, pose=light_pose)
r = pyrender.OffscreenRenderer(img_w, img_h)
color, depth = r.render(scene)

# draw grasp points
pro_matrix = camera.get_projection_matrix(img_w, img_h)


s_point = np.array(args.grasp_center)

grasp_vector = np.array(args.grasp_vector)
s_outer_point0 = grasp_vector * 0.05 + s_point
s_outer_point1 = grasp_vector * -0.05 + s_point

(x0, y0) = convert_object_point_to_img(s_point, obj_pose, camera_pose, pro_matrix)
(x1, y1) = convert_object_point_to_img(s_outer_point0, obj_pose, camera_pose, pro_matrix)
(x2, y2) = convert_object_point_to_img(s_outer_point1, obj_pose, camera_pose, pro_matrix)

print(x0)
print(y0)

# rotate view and update view
def press(event):
    if event.key == 'a':
        global alpha
        global scene
        global ax0
        global sc
        alpha += np.radians(1)
        camera_pose = np.array([
            [np.cos(alpha), 0, np.sin(alpha), np.sin(alpha) * height],
            [0, 1, 0, 0],
            [-np.sin(alpha), 0, np.cos(alpha), np.cos(alpha) * height],
            [0, 0, 0, 1]
        ])
        scene.set_pose(nc, pose=camera_pose)
        color, depth = r.render(scene)
        (x0, y0) = convert_object_point_to_img(s_point, obj_pose, camera_pose, pro_matrix)
        (x1, y1) = convert_object_point_to_img(s_outer_point0, obj_pose, camera_pose, pro_matrix)
        (x2, y2) = convert_object_point_to_img(s_outer_point1, obj_pose, camera_pose, pro_matrix)
        
        for s in sc:
            s.remove()
        
        ax0.axis('off')
        ax0.imshow(color)        
        sc[0] = ax0.scatter(x0,y0)
        sc[1] = ax0.scatter(x1,y1)
        sc[2] = ax0.scatter(x2,y2)
        
        ax1.axis('off')
        ax1.imshow(depth, cmap=plt.cm.gray_r)
        sc[3] = ax1.scatter(x0,y0)
        sc[4] = ax1.scatter(x1,y1)
        sc[5] = ax1.scatter(x2,y2)
        
        plt.draw()

sc = [None] * 6

# First plot
fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', press)
ax0 = plt.subplot(1,2,1)
ax0.axis('off')
ax0.imshow(color)
sc[0] = ax0.scatter(x0,y0)
sc[1] = ax0.scatter(x1,y1)
sc[2] = ax0.scatter(x2,y2)
ax1 = plt.subplot(1,2,2)
ax1.axis('off')
ax1.imshow(depth, cmap=plt.cm.gray_r)
sc[3] = ax1.scatter(x0,y0)
sc[4] = ax1.scatter(x1,y1)
sc[5] = ax1.scatter(x2,y2)
plt.show()