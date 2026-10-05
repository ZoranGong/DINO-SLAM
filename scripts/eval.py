import argparse
import os
import random
import sys

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

from scene import load_scence
from utils.eval import eval_final

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, default="./output/Replica_exp0_seed0/room0", help="Path to experiment file")
    args = parser.parse_args()


    model_path = os.path.join(args.m, "save")
    config, gaussians, w2cs, gt_w2cs = load_scence(model_path)
    
    # You can adjust the evaluation config here.
    config["eval"]["eval_mesh"] = True
    # config["eval"]["mesh_interval"] = 5 
    config["eval"]["save_mesh"] = True
    config["eval"]["save_renders"] = False
    # config['data']['meshdir'] = "./data/Replica/cull_replica_mesh"
    eval_final(config, gaussians, w2cs, gt_w2cs, os.path.join(args.m, "eval_result"))