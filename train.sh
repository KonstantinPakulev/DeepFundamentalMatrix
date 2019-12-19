#!/bin/bash

train_path="/home/konstantin/personal/DeepFundamentalMatrix/Family"
val_path="/home/konstantin/personal/DeepFundamentalMatrix/south-building"

#python3 -u ~/personal/DeepFundamentalMatrix/train.py --train_path="${train_path}" --val_path="${val_path}"

rm "nohup.out"
nohup python3 -u ~/personal/DeepFundamentalMatrix/train.py --train_path="${train_path}" --val_path="${val_path}" &