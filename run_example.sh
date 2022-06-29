#!/bin/bash

#check environement
#creat in doubt
#pip install -r requirements.txt
srun \
  --container-image=/data/enroot/nvcr.io_nvidia_pytorch_22.05-py3.sqsh \
  --container-workdir="`pwd`" \
  --container-mounts=/data/nothnagel:/data/nothnagel/test_folder \
  python ./src/test.py
