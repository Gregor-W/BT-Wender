import numpy as np
from autolab_core import RigidTransform
import pickle
import subprocess
import os


path = "/home/gregor/Festo/render-depth/images"

# read grasps
with open('/home/gregor/Share/Festo/aligned_grasps_list.pkl', 'rb') as f:
    grasps = pickle.load(f)

for n, grasp in enumerate(grasps):
    
    stable_pose = grasp['table_pose']

    grasp_string = list()
    for i in grasp["grasp_T"].translation:
        grasp_string.append(format(i, 'f'))
    grasp_string.append('1')


    vector_string = list()
    for i in grasp["axis"]:
        vector_string.append(format(i, 'f'))
    vector_string.append('1')


    pose_string = list()
    for e, i in enumerate(stable_pose.translation):
        for j in stable_pose.rotation[e]:
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
    
    call_list.append("--path")
    call_list.append(os.path.join(path, "depth" + str(n) + ".png"))

    print(call_list)

    subprocess.call(call_list)


