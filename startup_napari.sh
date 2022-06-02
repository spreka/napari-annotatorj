#!/bin/bash

# Run this script from the terminal 
# to install napari-annotatorj on your Linux PC
# after you have successfully installed Python 3.7
# or later

# to verify Python install, run the next command
python -V
if [ $? -ne 0 ]
then
    echo ERROR: "Error checking Python version. Do you have Python installed on your PC?"
    exit 1
fi
echo "Verified Python version"

# current folder
ROOT_DIR=$0

# parent of current folder
#PARENT_FOLDER="$(dirname "$(dirname "$ROOT_DIR")")"
PARENT_FOLDER=$(dirname $(dirname $(realpath $0)))
# name of the virtual environment
VENVNAME=$PARENT_FOLDER/napariAnnotatorjEnv
echo $VENVNAME

# check if the default virtenv already exists in the
# expected location
if [ -d "$VENVNAME" ]; then
  if [ -d "$VENVNAME/lib/site-packages/pip" ]; then
    echo "Virtenv $VENVNAME already exists"
  else
    echo "Virtenv $VENVNAME does not exist, please run install.sh first"
    exit 1
  fi
else
  echo "Virtenv $VENVNAME does not exist, please run install.sh first"
  exit 1
fi

# activate virtenv and run napari
source $VENVNAME/bin/activate && echo "Activated virtual environment successfully: $VENVNAME"  && napari