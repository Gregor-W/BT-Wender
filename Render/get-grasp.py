import numpy as np
from autolab_core import RigidTransform
import pickle
import subprocess


# read grasps
with open('/home/gregor/Share/Festo/grasp_list.pkl', 'rb') as f:
    grasps = pickle.load(f)
g_i = 10
sp_i = 10
print("### Grasp ###")
print(grasps[g_i])

with open('/home/gregor/Share/Festo/stable_poses_list.pkl', 'rb') as f:
    stable_poses = pickle.load(f)
print("### Stable Pose ###")
print(stable_poses[1])

# find grasps in plane
valid_grasps = list()
for grasp in grasps:
    for sp in stable_poses:
        grasp_vector = grasp[1]
        # create trans matirx
        trans = np.empty((4, 4))
        trans[:3, :3] = sp[0].rotation
        trans[:3, 3] = sp[0].translation
        trans[3, :] = [0,0,0,1]
        
        # find angle to world y axis
        grasp_vector = trans.dot(np.append(grasp_vector, [0]))
        grasp_vector = grasp_vector[0:3]
        unit_gv = grasp_vector / np.linalg.norm(grasp_vector)
        angle = np.arccos(np.dot(unit_gv, [0, 1, 0]))
        angle = np.rad2deg(angle)
        if angle > 85 and angle < 95:
            valid_grasps.append((grasp, sp, angle)) 


for n, vg in enumerate(valid_grasps):
    
    grasp = vg[0]
    stable_pose = vg[1]
    
    print("Angle: %f" % vg[2])

    grasp_string = list()
    for i in grasp[0].translation:
        grasp_string.append(format(i, 'f'))
    grasp_string.append('1')


    vector_string = list()
    for i in grasp[1]:
        vector_string.append(format(i, 'f'))
    vector_string.append('1')


    pose_string = list()
    for e, i in enumerate(stable_pose[0].translation):
        for j in stable_pose[0].rotation[e]:
            pose_string.append(format(j, 'f'))
        pose_string.append(format(i, 'f'))

    pose_string = pose_string + ['0', '0', '0', '1']

    # Build process call
    call_list = ["python3", "render-depth-save.py", "--mesh", "/home/gregor/Share/Festo/0024_2439.uf_proc.obj", "--pose"]
    call_list += pose_string
    call_list.append("--grasp_center")
    call_list += grasp_string
    call_list.append("--grasp_vector")
    call_list += vector_string
    
    call_list.append("--number")
    call_list.append(str(n))

    print(call_list)

    subprocess.call(call_list)


