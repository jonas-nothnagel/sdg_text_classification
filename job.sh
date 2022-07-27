#!/bin/bash

set -xe
WANDB_API_KEY=$771c73c3100b69dffd05bcbb7b8a4a4f02c73f4a

pip install -r requirements.txt
wandb login $WAND_API_KEY
python ./src/train.py
