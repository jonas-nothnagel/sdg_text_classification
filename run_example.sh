#!/bin/bash

# install requirements 
pip install -r requirements.txt

srun \
  --container-image=/data/enroot/nvcr.io_nvidia_pytorch_22.05-py3.sqsh \
  --container-workdir="`pwd`" \
  --container-mounts=/data/nothnagel/sdg_text_classification:/data/nothnagel/sdg_text_classification \
  python ./src/test.py

