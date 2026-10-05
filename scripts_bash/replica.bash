#!/bin/bash
EXP=0
export EXP
for seed in 0 1 2
do
    SEED=${seed}
    export SEED
    for scene in 2 0 1 3 4 5 6 7
    do
        SCENE_NUM=${scene}
        export SCENE_NUM
        echo "Running scene number ${SCENE_NUM} with seed ${SEED}"
        python3 -u  -W ignore scripts/gaus_mp.py configs/replica/config.py
    done
done
