# Generating Evolved Datasets to Train Neural Networks for Robotic Grasping
This is the Repo includes installation and execution code to generate an EGAD-dataset (https://github.com/dougsm/egad) and to convert it for training of GGCNN (https://github.com/dougsm/ggcnn).
Author: [Gregor Wender](https://github.com/Gregor-W)
This project was created as part of a collaboration with [Dionysios Satikidis](https://github.com/MrDio/) (University of Applied Sciences Esslingen) in collaboration with [Prof. Weiland](https://www.tec.reutlingen-university.de/fakultaet/personen/professoren/#jens-weiland) from [Reutingen University](https://www.reutlingen-university.de/home/)

# Installation:
The included installation scripts are setup for an AWS machine with a GPU running on Ubuntu 18.04.5 LTS.
The install scripts the file permissions have to be adjusted:
```bash
chmod +x install/*
```
Run install scrpits:
```bash
./install/install-egad
./install/install-ggcnn
./install/install-pyrender
```

# Running process automatically:
With the bash script "run-all.sh" the process can be run automatically from EGAD object creation to finished model.
Adjusting file permissions:
```bash
chmod +x run-all.sh
```
The script will create a new folder in .../output
```bash
./run-all .../output [OPTIONS]
Options:
-d, --download 	Download official EGAD training dataset instead of creating unique models
-l, --limit 	Limit the amount of converted mesh files for dataset generation (default=120)
-r, --resume 	Resume EGAD from previous run
-s, --skip 		Skip EGAD object creation if old EGAD models already exist
```

# Running manually:
Create an output dir with extra folder for egad
```bash
mkdir .../output/
mkdir .../output/egad-output
```

## 1
Run egad data generation
```bash
cd ~/egad/singularity
singularity run -B .../output/egad-output:/output --app datasetgen egad.sif
```
Resume latest egad data generation
```bash
singularity run -B .../output/egad-output:/output --app datasetgen egad.sif --resume
```
Instead of generating new EGAD dataset, official EGAD training dataset can be downloaded in egad-output folder
```bash
cd .../output/egad-output
wget https://data.researchdatafinder.qut.edu.au/dataset/c5a0ccba-fa28-4cb7-a9f8-4a7f93670344/resource/2b581c49-17f0-4941-8f8f-ffd4871c1117/download/egadtrainset.zip
unzip egadtrainset.zip
rm egadtrainset.zip
```

## 2
Run dexnet grasp and mesh preparation
```bash
cd ~/egad/singularity
singularity run --app dexnetgraspdata egad.sif .../output --limit 120
```
If the singularity container is unable to access the output directory use this command instead:
```bash
singularity run -B .../output:/output --app dexnetgraspdata egad.sif /output --limit 120
```

## 3
Run depthimage renderer
```bash
DISPLAY variable has to be unset for OSmesa offscreen rendering
export DISPLAY=""
python3 render/render-depth-all.py .../output
```

## 4
Run ggcnn dataset generation
```bash
cd ~/ggcnn_development_features/ggcnn
python3 generate_dataset.py .../output
```

## 5
Run ggcnn training
```bash
python3 train_ggcnn.py .../output
```
Continue training on existing model
```bash
python3 train_ggcnn.py .../output --contine_train .../old_model.hdf5
```

# References and Acknowledgements
This project wouldn't have been possible without the following projects:
* EGAD: https://github.com/dougsm/egad
* GGCNN: https://github.com/dougsm/ggcnn
* Dex-Net: https://github.com/BerkeleyAutomation/dex-net
* Pyrender: https://github.com/mmatl/pyrender
O* penrave: https://github.com/crigroup/openrave-installation


