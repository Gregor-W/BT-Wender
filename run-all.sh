#!/bin/bash
if [ ! -z "$1" ];
then
	OUTPUT = $1
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
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
	*)
      shift
  esac
done
	
if [ "$RESUME" == true ];
then
	WORKDIR = $OUTPUT
else
	DATESTR = $(date +%Y%m%d_%H%M%S)
	WORKDIR = $OUTPUT/egad-ggcnn_$DATESTR
	
	mkdir $OUTPUT
	mkdir $WORKDIR
	mkdir $WORKDIR/egad-output
fi

cd ~/egad/singularity
# run EGAD
if [ "$RESUME" == true ];
then
	singularity run -B $WORKDIR/egad-output:/output --app datasetgen egad.sif --resume
else
	singularity run -B $WORKDIR/egad-output:/output --app datasetgen egad.sif
fi
# create graspdata
singularity run -B $WORKDIR:/output --app dexnetgendataset egad.sif /output --LIMIT $LIMIT

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

