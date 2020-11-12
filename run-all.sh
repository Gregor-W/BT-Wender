#!/bin/bash
if [ ! -z "$1" ];
then
	OUTPUT=$1
else
	echo Error: no output folder specified
	exit 1
fi
shift

LIMIT=120

while (( "$#" )); do
  case "$1" in
    -r|--resume)
      RESUME=true
      shift
      ;;
	-l|--limit)
	  LIMIT="$2"
      shift
	  shift
      ;;
	-s|--skip)
	  SKIP=true
      shift
      ;;
	-d|--download)
	  DOWNLOAD=true
      shift
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
	*)
      shift
  esac
done
	
if [ "$RESUME" == true ] || [ "$SKIP" == true ];
then
	if [-a $OUTPUT];
	then
		WORKDIR=$OUTPUT
	else
		echo Resume directory does not exist
		exit 1
	fi
else
	DATESTR=$(date +%Y%m%d_%H%M%S)
	WORKDIR=$OUTPUT/egad-ggcnn_$DATESTR
	
	mkdir -p $OUTPUT
	mkdir $WORKDIR
	mkdir $WORKDIR/egad-output
	echo Creating new output folder: $WORKDIR
fi

SINGULARITYPATH=~/egad/singularity

# run EGAD
if [ ! "$SKIP" == true ];
then
	if [ "$DOWNLOAD" == true ];
	then
		pushd $WORKDIR/egad-output
		wget https://data.researchdatafinder.qut.edu.au/dataset/c5a0ccba-fa28-4cb7-a9f8-4a7f93670344/resource/2b581c49-17f0-4941-8f8f-ffd4871c1117/download/egadtrainset.zip
		unzip egadtrainset.zip
		rm egadtrainset.zip
		popd
	elif [ "$RESUME" == true ];
	then
		singularity run -B $WORKDIR/egad-output:/output --app datasetgen $SINGULARITYPATH/egad.sif --resume
	else
		singularity run -B $WORKDIR/egad-output:/output --app datasetgen $SINGULARITYPATH/egad.sif
	fi
fi

# create graspdata
singularity run -B $WORKDIR:/output --app dexnetgraspdata $SINGULARITYPATH/egad.sif /output --limit $LIMIT

# render depth images
OLDDISPLAY=$DISPLAY
export DISPLAY=""
python3 render/render-depth-all.py $WORKDIR
export DISPLAY=$OLDDISPLAY

# GGCNN
cd ~/ggcnn_development_features/ggcnn
# generate ggcnn datasets
python3 generate_dataset.py $WORKDIR
# run ggcnn training
python3 train_ggcnn.py $WORKDIR

