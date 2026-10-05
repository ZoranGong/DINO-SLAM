#!/bin/bash

for scene in 0 1 2 3 4  
do
    for seed in 0 1 2
    do
        SEED=${seed}
        export SEED
        SCENE_NUM=${scene}
        export SCENE_NUM
        echo "Running scene number ${SCENE_NUM} with seed 0"
        python3 -u scripts/gaus_mp.py configs/scannetpp/config.py
    done 
done