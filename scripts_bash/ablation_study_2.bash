#!/bin/bash

# ablation study E F G H
# Replica Room 0
for exp in 6 5 7 8
do
    EXP=${exp}
    export EXP
    SEED=0
    export SEED
    
    SCENE_NUM=0
    export SCENE_NUM

    if [ $VIEW -eq 6 ]; then
        python3 -u  -W ignore scripts/splatam.py configs/replica/config.py
    else
        python3 -u  -W ignore scripts/gaus_mp.py configs/replica/config.py
    fi

done

# TUM-RGBD fr3/office
for exp in 6 5 7 8
do
    EXP=${exp}
    export EXP
    SEED=0
    export SEED
    SCENE_NUM=4
    export SCENE_NUM

    if [ $VIEW -eq 6 ]; then
        python3 -u  -W ignore scripts/splatam.py configs/tum/config.py
    else
        python3 -u  -W ignore scripts/gfslam_mp.py configs/tum/config.py
    fi
    
done
