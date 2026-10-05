#!/bin/bash

# ablation study A B C D
for exp in 1 2 3 4
do
    EXP=${exp}
    export EXP
    SEED=0
    export SEED
    SCENE_NUM=0
    export SCENE_NUM

    python3 -u  -W ignore scripts/gaus_mp.py configs/replica/config.py
done
