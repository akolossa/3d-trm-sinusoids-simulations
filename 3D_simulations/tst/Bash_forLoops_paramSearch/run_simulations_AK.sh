#!/bin/bash

export CUDA_VISIBLE_DEVICES=3



temperature=4
max_act=12 
lamda_act=1000
adh=20
lambda_p=20 #this is devided by 100 
for temperature in $(seq 5 5 40) ; do
    for max_act in $(seq 12 2 45) ; do 
        for lambda_act in $(seq 1000 50 2000) ; do
            for adh in $(seq 5 5 45) ; do
                for lambda_p in $(seq 10 20) ; do #this is devided by 100, so it ranges from 0.1 to 0.2
                    python3 example3d.py $temperature $max_act $lambda_act $adh $lambda_p
                    python3 example3d.py $temperature $max_act $lambda_act $adh $lambda_p
                    python3 example3d.py $temperature $max_act $lambda_act $adh $lambda_p
                done
            done
        done
    done
done
