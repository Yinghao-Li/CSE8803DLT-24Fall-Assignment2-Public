#!/bin/bash

# Quit if there are any errors
set -e

name=""
gtid=""

data_dir="./data"

lr=0.00005
batch_size=16
n_epochs=20
warmup_ratio=0.1

seed=42


CUDA_VISIBLE_DEVICES=$1 python run.py \
  --name "$name" \
  --gtid "$gtid" \
  --data_dir $data_dir \
  --lr $lr \
  --batch_size $batch_size \
  --n_epochs $n_epochs \
  --warmup_ratio $warmup_ratio \
  --seed $seed
