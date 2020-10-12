import numpy as np
from autolab_core import RigidTransform
import pickle
import subprocess
import argparse
import os

# command arguments
parser = argparse.ArgumentParser()
parser.add_argument('grasp', type=int)
args = parser.parse_args()

# convert Transformation matrix into argument for render-depth
def matrix_arg(T_matrix):
    arg_string = list()
    for e, i in enumerate(T_matrix.translation):
        for j in T_matrix.rotation[e]:
            arg_string.append(format(j, 'f'))
        arg_string.append(format(i, 'f'))
    arg_string += ['0', '0', '0', '1']
    return arg_string
    
# convert vector into argument for render-depth
def vector_arg(v):
    arg_string = list()
    for i in v:
        arg_string.append(format(i, 'f'))
    arg_string.append('1')
    return arg_string


mesh_path = "/home/gregor/Festo/grasp-data/object-files"

# read grasps
with open('/home/gregor/Festo/grasp-data/test-object0/0024_2439_aligned_grasps_list.pkl', 'rb') as f:
    grasp_data = pickle.load(f), encoding="bytes")

vg_i = args.grasp


sp = grasp_data[0]
mesh = sp['mesh']
stable_pose = sp['table_pose']
grasps = sp['grasps']
pose_string = matrix_arg(stable_pose)

grasp = grasps[vg_i]
grasp_T = grasp["grasp_T"]
            
grasp_string = matrix_arg(grasp_T)


p0_string = vector_arg(grasp["contact0"])
p1_string = vector_arg(grasp["contact1"])

# Build process call
call_list = ["python3", "render-depth.py", "--mesh", os.path.join(mesh_path, mesh), "--pose"]
call_list += pose_string
call_list.append("--grasp_center")
call_list += grasp_string
call_list.append("--contact_p0")
call_list += p0_string
call_list.append("--contact_p1")
call_list += p1_string

print(call_list)

subprocess.call(call_list)


