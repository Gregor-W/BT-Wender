#!/bin/bash
# get 
while [ "$1" != "" ]; do
    case $1 in
        -o | --output_dir )           shift
                                output="$1"
                                ;;
    esac
    shift
done

if [$output != ""]
then
	mkdir $output/
	mkdir $output/egad-output

	cd ~/egad/singularity
	# run egad
	singularity run -B $output/egad-output:/output --app datasetgen egad.sif
	# create graspdata
	singularity run -B $output:/output --app dexnetgendataset egad.sif /output/egad-output/  --limit 120

	# render depth images
	OLDDISPLAY=$DISPLAY
	export DISPLAY=""
	python3 render/render-depth-all.py $output
	export DISPLAY=$OLDDISPLAY

	cd ~/ggcnn_development_features/ggcnn
	# generate ggcnn datasets
	python3 generate_dataset.py $output
	# run ggcnn training
	python3 train_ggcnn.py $output
else
	echo no output folder specified
fi
