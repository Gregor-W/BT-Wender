#!/bin/bash


while [ "$1" != "" ]; do
    case $1 in
        -o | --output_dir )           shift
                                output="$1"
                                ;;
    esac
    shift
done


mkdir $output/
mkdir $output/egad-output

cd ~/egad/singularity
singularity run -B $output/egad-output:/output --app datasetgen egad.sif

singularity run --app dexnetgendataset egad.sif ~/data/egad-output/ --output_dir  ~/data/grasp-data


OLDDISPLAY=$DISPLAY
export DISPLAY=""
python3 Render/render-depth-all.py $output/grasp-data
export DISPLAY=$OLDDISPLAY

cd ~/ggcnn_development_features/ggcnn

python3 generate_dataset.py $output/datasets

python3 train_ggcnn.py $output/networks