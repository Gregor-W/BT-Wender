# BT-Wender
This is the Repo for the Thesis of G. Wender

Installation:

Set permissions for install scripts
chmod +x install/*

Run install scrpits
./install-egad
./install-ggcnn
./install-pyrender


Running process automatically:
chmod +x create-egad-model.sh


Running manually:
Create an output dir with extra folder for egad
mkdir .../output/
mkdir .../output/egad-output

Run egad data generation
cd ~/egad/singularity
singularity run -B $output/egad-output:/output --app datasetgen egad.sif

Resume latest egad data generation
singularity run -B $output/egad-output:/output --app datasetgen egad.sif --resume

Run dexnet grasp and mesh preparation
cd ~/egad/singularity
singularity run --app dexnetgendataset egad.sif .../output/egad-output/ --output_dir  .../output/grasp-data

Run depthimage renderer
DISPLAY variable has to be unset for OSmesa offscreen rendering
export DISPLAY=""
python3 Render/render-depth-save.py .../output/grasp-data

Run ggcnn dataset generation
cd ~/ggcnn_development_features/ggcnn
python3 generate_dataset.py .../output/datasets

Run ggcnn training
python3 train_ggcnn.py .../output/networks