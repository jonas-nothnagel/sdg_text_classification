#!/bin/bash

# Run this script: "py_venv.sh COMMAND" to run the given command within a venv. 
# You can either set the ENVNAME below or pass the venv name to the script:
# py_venv.sh COMMAND VENV_NAME
COMMAND=$1

# Note: make sure to use "" for commands/directories which include spaces

# Folder in which you want to maintain all virtual environments
# NOTE: no trailing / !
VENVS_FOLDER=/data/nothnagel/venvs

# You can manually change the venv name here
ENVNAME=env1

# Checks if more than one argument where passed to the script
if [ $# -gt 1 ]
  then
    ENVNAME=$2
fi

# Full path to desired venv folder
TARGET_DIR=${VENVS_FOLDER}/${ENVNAME}

if [ -d $TARGET_DIR ]
    then
        echo "Using existing venv: $TARGET_DIR "
        # Activate the environment
        source ${TARGET_DIR}/bin/activate
 
    else
        echo "Creating new venv: $TARGET_DIR ..."
        # NOTE: using --system-site-packages here. This allows the python environment
        # To use the system-wide installed packages!
        python3 -m venv $TARGET_DIR --system-site-packages

        # Activate the environment
        source ${TARGET_DIR}/bin/activate

        # Manually update pip to the newest version
        python3 -m pip install --upgrade pip
        echo "Done!"
fi

eval $COMMAND

# In the end you might want to disable the python environment to reset everything to it's normal state
deactivate



