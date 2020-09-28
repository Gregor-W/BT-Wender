import numpy as np
from autolab_core import RigidTransform
import pickle
import subprocess
import argparse

# command arguments
parser = argparse.ArgumentParser()
parser.add_argument('grasp', type=int)
args = parser.parse_args()

# read grasps
with open('/home/gregor/Share/Festo/aligned_grasps_list.pkl', 'rb') as f:
    grasps = pickle.load(f)

vg_i = args.grasp

# create command
grasp = grasps[vg_i]

print("axis: ")
print(grasp['axis'])

print("T_obj_table")
print(grasp['T_obj_table'])

print("T_table_obj")
print(grasp['T_table_obj'])

print('T_obj_world')
print(grasp['T_obj_world'])

print("table_pose:")
print(grasp['table_pose'])



print("close_width: %f" % grasp["close_width"])
print("approach_angle: %f" % grasp["approach_angle"])

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
call_list = ["python2", "render-depth.py", "--mesh", "/home/gregor/Share/Festo/0024_2439.uf_proc.obj", "--pose"]
call_list += pose_string
call_list.append("--grasp_center")
call_list += grasp_string
call_list.append("--grasp_vector")
call_list += vector_string


print(call_list)

subprocess.call(call_list)


