#!/bin/bash

# vnc
sudo apt-get install ubuntu-desktop gnome-panel gnome-settings-daemon metacity nautilus gnome-terminal xfce4 vnc4server
mkdir -p .vnc
cp vnc-startup ~/.vnc/xstartup

# flatpak
sudo apt-get install flatpak
sudo flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
sudo flatpak install flathub org.octave.Octave

