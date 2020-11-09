#!/bin/bash

# Render depth images
# Pyrender requirements
mkdir -p ~/pyrender
cd ~/pyrender
wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb
sudo dpkg -i ./mesa_18.3.3-0.deb || true
sudo apt install -y -f


git clone https://github.com/mmatl/pyopengl.git
pip install ./pyopengl

pip install pyrender
pip install trimesh
#pip install opencv-python

export PYOPENGL_PLATFORM=osmesa
echo 'export PYOPENGL_PLATFORM=osmesa' >> ~/.bashrc