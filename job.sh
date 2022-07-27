#!/bin/bash

set -xe
wandb login 
pip install -r requirements.txt
python ./src/train.py
