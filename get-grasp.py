import numpy as np
from autolab_core import RigidTransform
import pickle
import subprocess


# read grasps
with open('depth_testdata/grasp_list.pkl', 'rb') as f:
    grasps = pickle.load(f)
g_i = 10
sp_i = 10
print("### Grasp ###")
print(grasps[g_i])

with open('depth_testdata/stable_poses_list.pkl', 'rb') as f:
    stable_poses = pickle.load(f)
print("### Stable Pose ###")
print(stable_poses[1])

# find grasps in plane
valid_grasps = list()
for grasp in grasps:
    grasp_vector = grasp[1]
    for sp in stable_poses:
        # create trans matirx
        trans = np.empty((4, 4))
        trans[:3, :3] = sp[0].rotation
        trans[:3, 3] = sp[0].translation
        trans[3, :] = [0,0,0,1]
        
        # find angle to world y axis
        grasp_vector = trans.dot(np.append(grasp_vector, [1]))
        grasp_vector = grasp_vector[0:3]
        unit_gv = grasp_vector / np.linalg.norm(grasp_vector)
        angle = np.arccos(np.dot(unit_gv, [0, 1, 0]))
        angle = np.rad2deg(angle)
        print(angle)
        if angle > 45 and angle > 135:
            valid_grasps.append((grasp, sp)) 

# create command
grasp_string = list()
for i in grasps[g_i][0].translation:
    grasp_string.append(format(i, 'f'))
grasp_string.append('1')

pose_string = list()
for e, i in enumerate(stable_poses[sp_i][0].translation):
    for j in stable_poses[sp_i][0].rotation[e]:
        pose_string.append(format(j, 'f'))
    pose_string.append(format(i, 'f'))

pose_string = pose_string + ['0', '0', '0', '1']

# Build process call
call_list = ["python3", "render-depth.py", "--mesh", "depth_testdata/0024_2439.uf_proc.obj", "--pose"]
call_list += pose_string
call_list.append("--grasp_center")
call_list += grasp_string

print(call_list)

subprocess.call(call_list)


