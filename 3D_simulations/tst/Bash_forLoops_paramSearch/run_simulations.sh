#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

lambda_p=2
temperature=20
adh=20
max_act=10
for max_act in $(seq 5 12) ; do
for lambda_act in $(seq 0 50 1500) ; do
	for adh in $(seq 20 4 40) ; do
		python3 example3d.py $temperature $max_act $lambda_act $adh $lambda_p
		python3 example3d.py $temperature $max_act $lambda_act $adh $lambda_p
	done
done
done
