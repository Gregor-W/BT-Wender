import numpy as np
from autolab_core import RigidTransform
import pickle
import subprocess
import os

input_path = "/home/gregor/Festo/grasp-data/test-object0"
mesh_path = "/home/gregor/Festo/grasp-data/object-files"
path = "/home/gregor/Festo/render-depth/images"

# scp -i /home/gregor/Festo/Wender-keypair.pem -r ubuntu@18.191.177.49:/home/ubuntu/egad/grasp-data /home/gregor/Festo/

dist = 0.15
camera_pose = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, dist],
    [0, 0, 0, 1]
])

pro_matrix = np.array([[ 1.73205081, 0, 0, 0],
    [0, 1.73205081, 0, 0],
    [0, 0, -1, -0.1],
    [0, 0, -1, 0]])




# read grasps

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

for file in os.listdir(input_path):
    with open(os.path.join(input_path, file), 'rb') as f:
        grasp_data = pickle.load(f)


    for n, sp in enumerate(grasp_data):
        mesh = sp['mesh']

        stable_pose = sp['table_pose']
        print(stable_pose)

        grasps = sp['grasps']
        
        pose_string = matrix_arg(stable_pose)

        for grasp in grasps:
            grasp_T = grasp["grasp_T"]
            
            grasp_string = matrix_arg(grasp_T)


            p0_string = vector_arg(grasp["contact0"])
            p1_string = vector_arg(grasp["contact1"])

            # Build process call
            call_list = ["python3", "render-depth-save.py", "--mesh", os.path.join(mesh_path, mesh), "--pose"]
            call_list += pose_string
            call_list.append("--grasp_center")
            call_list += grasp_string
            call_list.append("--contact_p0")
            call_list += p0_string
            call_list.append("--contact_p1")
            call_list += p1_string
            
            call_list.append("--path")
            call_list.append(os.path.join(path,  mesh.replace(".uf_proc.obj", "_") + str(int(grasp["quality"])) + "_depth" + str(n) + ".tiff"))

            print(call_list)
            
            pro_matrix = subprocess.check_output(call_list)


