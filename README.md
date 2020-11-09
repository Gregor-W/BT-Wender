# BT-Wender
This is the Repo for the Thesis of G. Wender

# Installation:

Set permissions for install scripts
```bash
chmod +x install/*
```
Run install scrpits
```bash
./install-egad
./install-ggcnn
./install-pyrender
```

# Running process automatically:
```bash
chmod +x create-egad-model.sh
```

# Running manually:
Create an output dir with extra folder for egad
```bash
mkdir .../output/
mkdir .../output/egad-output
```
Run egad data generation
```bash
cd ~/egad/singularity
singularity run -B $output/egad-output:/output --app datasetgen egad.sif
```
Resume latest egad data generation
```bash
singularity run -B $output/egad-output:/output --app datasetgen egad.sif --resume
```
Run dexnet grasp and mesh preparation
```bash
cd ~/egad/singularity
singularity run --app dexnetgendataset egad.sif .../output/egad-output/ --output_dir  .../output/grasp-data
```
Run depthimage renderer
```bash
DISPLAY variable has to be unset for OSmesa offscreen rendering
export DISPLAY=""
python3 Render/render-depth-save.py .../output/grasp-data
```
Run ggcnn dataset generation
```bash
cd ~/ggcnn_development_features/ggcnn
python3 generate_dataset.py .../output/datasets
```
Run ggcnn training
```bash
python3 train_ggcnn.py .../output/networks
```
