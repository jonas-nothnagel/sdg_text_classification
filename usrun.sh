#!/bin/sh

set -xe

IMAGE=/data/enroot/nvcr.io_nvidia_pytorch_23.06-py3.sqsh
#IMAGE=/data/enroot/nvcr.io_nvidia_tensorflow_22.05-tf2-py3.sqsh

srun -K \
  --container-mounts=/data:/data,$HOME:$HOME \
  --container-workdir=$PWD \
  --container-image=$IMAGE \
  $*

