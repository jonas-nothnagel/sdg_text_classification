#!/bin/bash

# show output in terminal
set -xe
# install python packages
pip install -r requirements.txt
# install git lfs
apt-get update
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs
git lfs install
# run training script
python ./src/train.py