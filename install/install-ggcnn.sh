#!/bin/bash

# ggcnn
cd ~
git clone https://github.com/Gregor-W/ggcnn_development_features.git
pip install keras
pip install tensorflow
#pip install --upgrade --force-reinstall tensorflow-gpu==1.9.0


#https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
#tar -xzvf cudnn-10.0-linux-x64-v7.6.4.38.tgz
#sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
#sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
#sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
