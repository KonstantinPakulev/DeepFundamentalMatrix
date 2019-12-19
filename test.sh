#!/bin/bash

test_path="/home/konstantin/personal/DeepFundamentalMatrix/south-building"
model_path="/home/konstantin/personal/DeepFundamentalMatrix/checkpoints_sh/model_epoch2.pt"

python3 -u ~/personal/DeepFundamentalMatrix/test.py --test_path="${test_path}" --model_path="${model_path}"