#!/bin/bash

dataset_path="/home/konstantin/personal/DeepFundamentalMatrix/Family"

rm "nohup.out"

nohup python3 -u ~/personal/DeepFundamentalMatrix/train.py --dataset_path="${dataset_path}" &