#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# Define the unique parameter combinations
params=(
    "33 959 79 50 14 9"
)

# Loop through each parameter combination and run the simulation 100 times
for param in "${params[@]}"; do
    read temperature lambda_act max_act adh lambda_p lambda_a <<< "$param"
    for i in $(seq 1 100); do
        python3 example3d.py $temperature $lambda_act $max_act $adh $lambda_p $lambda_a
    done
done