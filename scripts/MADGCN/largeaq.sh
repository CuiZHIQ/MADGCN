#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/MADGCN" ]; then
    mkdir ./logs/MADGCN
fi

model_name="MADGCN"
dataset_name="LargeAQ"
config_file="model/MADGCN/LargeAQ.py"

gpus="0"

random_seed=2024

echo "=== Starting MADGCN Training on LargeAQ Dataset ==="
echo "Model: $model_name"
echo "Dataset: $dataset_name"
echo "Config: $config_file"
echo "GPU: $gpus"
echo "Random Seed: $random_seed"
echo "================================================="

python -u scripts/run_madgcn.py \
    --config_file $config_file \
    --gpus $gpus \
    --seed $random_seed \
    2>&1 | tee logs/MADGCN/${model_name}_${dataset_name}_seed${random_seed}_$(date +%Y%m%d_%H%M%S).log

echo "=== Training completed! Check logs/MADGCN/ for results ===" 