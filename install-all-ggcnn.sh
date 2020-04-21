#!/bin/bash

# vnc
sudo apt-get install ubuntu-desktop gnome-panel gnome-settings-daemon metacity nautilus gnome-terminal xfce4 vnc4server
mkdir -p .vnc
cp vnc-startup ~/.vnc/xstartup

# ggcnn
mkdir ~/ggcnn
cd ~/ggcnn
git clone https://github.com/MrDio/ggcnn_development_features.git
git clone https://github.com/MrDio/ggcnn_tutorial_dataset.git
git clone https://github.com/dougsm/ggcnn
# create python env
conda create --name ggcnn-env
source activate ggcnn-env
cd ~/ggcnn/ggcnn
# pip installs
pip3 install -r requirements.txt
pip3 install -U scikit-image

# flatpak
sudo apt-get install flatpak
sudo flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
sudo flatpak install flathub org.octave.Octave
