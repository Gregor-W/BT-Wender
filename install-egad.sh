#!/bin/bash

#fix python pip
sudo python3 -m pip install -U --force-reinstall pip

cd ~

#### EGAD ####
# Basic
sudo apt install -y libspatialindex-dev python-rtree
git clone https://github.com/dougsm/egad.git 
cd egad
pip3 install -e .
cd ~


sudo apt-get update && sudo apt-get install -y \
    build-essential \
    uuid-dev \
    libgpgme-dev \
    squashfs-tools \
    libseccomp-dev \
    wget \
    pkg-config \
    git \
    cryptsetup-bin \
	llvm-6.0

# Go
export VERSION=1.14.2 OS=linux ARCH=amd64 && \
    wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz && \
    sudo tar -C /usr/local -xzvf go$VERSION.$OS-$ARCH.tar.gz && \
    rm go$VERSION.$OS-$ARCH.tar.gz
echo 'export GOPATH=${HOME}/go' >> ~/.bashrc && \
    echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> ~/.bashrc && \
    source ~/.bashrc

# Singularity
export VERSION=3.5.3 && # adjust this as necessary \
    wget https://github.com/sylabs/singularity/releases/download/v${VERSION}/singularity-${VERSION}.tar.gz && \
    tar -xzf singularity-${VERSION}.tar.gz && \
    cd singularity

git clone https://github.com/sylabs/singularity.git && \
    cd singularity && \
    git checkout v${VERSION}
./mconfig && \
    make -C ./builddir && \
    sudo make -C ./builddir install

# EGAD
cp ~/BT-Wender/singularity.def ~/egad/singularity/
cd ~/egad/singularity
sudo singularity build --tmpdir ~/egad/build-tmp egad.sif singularity.def
# sudo singularity build egad.sif singularity.def &> install-log.txt



# Pointcloud
#pip install pypcd

# Render depth images
# Pyrender requirements
mkdir -p ~/pyrender
cd ~/pyrender
wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb
sudo dpkg -i ./mesa_18.3.3-0.deb || true
sudo apt install -y -f

#apt-get install -y llvm-6.0 freeglut3 freeglut3-dev
#wget ftp://ftp.freedesktop.org/pub/mesa/mesa-18.3.3.tar.gz
#tar xfv mesa-18.3.3.tar.gz
#cd mesa-18.3.3
#./configure --prefix=/usr/local                           \
		--enable-opengl --disable-gles1 --disable-gles2   \
		--disable-va --disable-xvmc --disable-vdpau       \
		--enable-shared-glapi                             \
		--disable-texture-float                           \
		--enable-gallium-llvm --enable-llvm-shared-libs   \
		--with-gallium-drivers=swrast,swr                 \
		--disable-dri --with-dri-drivers=                 \
		--disable-egl --with-egl-platforms= --disable-gbm \
		--disable-glx                                     \
		--disable-osmesa --enable-gallium-osmesa          \
		ac_cv_path_LLVM_CONFIG=llvm-config-6.0
#make -j8
#make install		
#cd ..

git clone https://github.com/mmatl/pyopengl.git
pip install ./pyopengl
#pip3 install PyOpenGL PyOpenGL_accelerate

#pip3 uninstall -y PyOpenGL PyOpenGL_accelerate
#git clone https://github.com/mmatl/pyopengl.git
#cd pyopengl
#python3 setup.py develop
#cd accelerate
#python3 setup.py develop
#cd ..
#cd ..

pip install pyrender
pip install trimesh
#pip install opencv-python

export PYOPENGL_PLATFORM=osmesa
echo 'export PYOPENGL_PLATFORM=osmesa' >> ~/.bashrc

# ggcnn
cd ~
git clone https://github.com/Gregor-W/ggcnn_development_features.git
pip install keras
pip install tensorflow

pip install --upgrade --force-reinstall tensorflow-gpu==1.9.0


#https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
#tar -xzvf cudnn-10.0-linux-x64-v7.6.4.38.tgz
#sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
#sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
#sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

