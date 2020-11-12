#!/bin/bash

#current directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

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
# copy changed singularity.def
cp $DIR/../singularity.def ~/egad/singularity/
cd ~/egad/singularity
mkdir ~/egad/build-tmp
sudo singularity build --tmpdir ~/egad/build-tmp egad.sif singularity.def &> install-log.txt


