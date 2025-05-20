#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# Define the unique parameter combinations
params=(
    "10 40 1050 45 17"
    "10 40 1350 35 16"
    "10 40 1450 35 18"
    "10 44 1000 35 14"
    "10 44 1000 35 20"
    "10 44 1200 30 16"
    "10 44 1350 35 14"
    "15 16 1150 35 20"
    "15 22 1100 40 19"
)

# Loop through each parameter combination and run the simulation 100 times
for param in "${params[@]}"; do
    read temperature max_act lambda_act adh lambda_p <<< "$param"
    for i in $(seq 1 100); do
        python3 example3d.py $temperature $max_act $lambda_act $adh $lambda_p
    done
done