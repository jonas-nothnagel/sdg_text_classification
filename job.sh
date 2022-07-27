#!/bin/bash

set -xe

pip install -r requirements.txt
python ./src/train.py
