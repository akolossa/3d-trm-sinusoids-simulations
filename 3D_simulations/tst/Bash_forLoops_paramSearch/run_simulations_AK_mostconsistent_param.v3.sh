#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

# Define the unique parameter combinations
params=(
    "15 22 1750 20 18"
    "15 24 1050 40 13"
    "15 24 1100 45 20"
    "15 24 1150 40 19"
    "15 26 1000 25 16"
    "15 32 1100 20 19"
    "15 32 1550 35 11"
    "15 34 1200 40 19"
    "15 36 1550 30 18"
    "15 36 1950 35 19"
    "15 38 1150 40 11"
    "15 38 1200 25 17"
    "15 38 1550 40 18"
    "15 40 1000 35 18"
    "15 40 1400 35 11"
    "15 40 1550 15 17"
    "15 40 2000 40 19"
    "15 42 1100 20 13"
)


# Loop through each parameter combination and run the simulation 100 times
for param in "${params[@]}"; do
    read temperature max_act lambda_act adh lambda_p <<< "$param"
    for i in $(seq 1 100); do
        python3 example3d.py $temperature $max_act $lambda_act $adh $lambda_p
    done
done