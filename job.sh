#!/bin/bash

set -xe

pip install transformers datasets accelerate
python ./src/train.py
