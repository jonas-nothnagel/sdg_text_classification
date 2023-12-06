#!/bin/bash

set -xe
pip install -r requirements.txt
python ./src/train_hyperparameter_tuning.py