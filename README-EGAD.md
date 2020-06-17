# Egad - Evolved Grasping Analysis Dataset
https://dougsm.github.io/egad/

Main Code:
https://github.com/dougsm/egad
Paper:
https://arxiv.org/abs/2003.01314

Install:
Singularity 3.5
https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps
Go 1.14.2
https://golang.org/doc/install

Main Ideas:
    • need for large and diverse datasets
    • evolutionary algorithms to generate a dataset

3D Objects are generated and encoded using Compositional Pattern Producing Networks

Two-dimensional search space for objects in terms of shape complexity and grasp difficulty
A maximum number of objects is allowed at each cell.  If the number of objects in a cell exceeds the maximum, the objects are compared to all other objects in the search space, and the least geometrically diverse object is removed

Shape Complexity:
measure of morphological complexity

Grasp Difficulty:
Dex-Net analytical grasp planner (only use of dexnet and grasping in this application)
https://arxiv.org/pdf/1703.09312.pdf

Geometric Diversity:
Topology Matching metric based on Multi resolutional Reeb Graphs

Dex-Net project provides extra functionality,  including the ability to label objects with other grasp sampling strategies or quality metrics, add custom grippers, and generate large image datasets for training visual grasp detection algorithms.

ROBOTIC EXPERIMENTS:
GG-CNN was trained on the Cornell Grasping Dataset, not using any new datasets.
→ Generation of depth image datasets not used here, implemented somewhere in dexnet


# EGAD Installation Problems:
Wrong Version Fix:
https://github.com/BerkeleyAutomation/autolab_core/blob/master/setup.py

Installing correct package in singularity.def:
for autolab
“sed -i 's/joblib/joblib==0.14.1/g' setup.py”

X server Bug:
https://github.com/BerkeleyAutomation/dex-net/blob/cccf93319095374b0eefc24b8b6cd40bc23966d2/src/dexnet/database/mesh_processor.py#L192
function “_load_mesh” calls command “meshlabserver -i \"%s\" -o \"%s\”
can’t connect to X server in singularity
fix: 
https://github.com/cnr-isti-vclab/meshlab/issues/78
sed -i 's/meshlabserver -i/xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i/g' setup.py

# Dex-Net
Research project including code, datasets, and algorithms for generating datasets of synthetic point clouds, robot parallel-jaw grasps and metrics of grasp robustness based on physics for thousands of 3D object models to train machine learning-based methods to plan robot grasps
Python API for HDF5 databases of 3D object models, parallel-jaw grasps, and grasp robustness metrics
Dex-Net:
https://github.com/BerkeleyAutomation/dex-net
Documentation:
https://berkeleyautomation.github.io/dex-net/code.html
Fix:
https://docs.google.com/document/d/1YImq1cBTy9E1n1On6-00gueDT4hfmYJK4uOcxZIzPoY/edit

Paper:
https://arxiv.org/pdf/1703.09312.pdf

Dex-Net
    • python api
    • command line tool:
      https://docs.google.com/document/d/1a9aoDuo-iYG-UyCJPq-ubnyW2Hgnf0YIdYvkp5DlBf0/edit
    • generates grasp points from 3D model
    • Fig. 3 in paper → can create depth images

Generating 3D models HDF5 from obj:
egad code for Dexnet:
https://github.com/dougsm/egad/blob/151f898afd86c41750e159a4100a7761d6dac841/scripts/dexnet_grasp_rank_interface.py

https://berkeleyautomation.github.io/dex-net/api/database.html?highlight=generate_graspable#dexnet.database.MeshProcessor.generate_graspable

3D objects are generated using trimesh:
https://github.com/mikedh/trimesh
and exported 
https://github.com/mikedh/trimesh/blob/b63a092ea2836ae810c41021f3371e2cf00c2866/trimesh/exchange/export.py#L18
as obj files

default database:
https://berkeley.app.box.com/s/eaq37px77jxktr8ggti016pr3gpudp9l

# Creating depth images:
Only dexnet fuctions regarding images:
https://berkeleyautomation.github.io/dex-net/api/database.html
“store_rendered_images”
“rendered_images”

using perception render mode (perception.RenderMode.DEPTH)
https://berkeleyautomation.github.io/perception/api/image.html#rendermode

Small example database
Depth images:
https://github.com/BerkeleyAutomation/dex-net/blob/0f63f706668815e96bdeea16e14d2f2b266f9262/test/database_test.py

Depth image renderer:
https://github.com/BerkeleyAutomation/meshrender

-> how are images saved?

Other options using meshpy:

Render meshpy:
https://github.com/BerkeleyAutomation/meshpy/blob/22ad5d88351170f07301873083bec5a8651be893/examples/render_images.py
Render meshrender:
https://github.com/BerkeleyAutomation/meshrender/blob/master/examples/test_viewer.py

can create normal images, depth images do not work.



# Problems Meshpy Installation:
Fix boost:
https://stackoverflow.com/questions/24173330/cmake-is-not-able-to-find-boost-libraries

Fix meshpy
https://github.com/rdkit/rdkit/issues/1581
https://github.com/BerkeleyAutomation/meshpy/issues/7
https://stackoverflow.com/questions/8430332/uninstall-boost-and-install-another-version
https://stackoverflow.com/questions/4123618/how-to-add-compiler-include-paths-and-linker-library-paths-for-newly-installed-b
https://stackoverflow.com/questions/51037886/trouble-with-linking-boostpythonnumpy

Cmake – setup:
https://iainhaslam.com/scraps/hello-boost-python/

libboost_numpy.so:
https://docs.google.com/document/d/1YImq1cBTy9E1n1On6-00gueDT4hfmYJK4uOcxZIzPoY/edit
https://github.com/ndarray/Boost.NumPy/issues/43
https://stackoverflow.com/questions/46934760/importerror-libboost-python-so-1-65-1-cannot-open-shared-object-file-no-such

Visualizer:
https://github.com/BerkeleyAutomation/dex-net/issues/20
needs shapely
