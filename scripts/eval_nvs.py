import argparse
import os
import random
import sys
import shutil
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

from scene import load_scence
from utils.eval import eval_nvs

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, default="./output/Replica_exp0_seed0/room0", help="Path to experiment file")
    args = parser.parse_args()

    model_path = os.path.join(args.m, "save")
    config, gaussians, w2cs, gt_w2cs = load_scence(model_path)
    config["eval"]["save_renders"] = True
    
    if config['data']['dataset_name'] != "scannetpp":
        print("Only support NVS on ScanNet++ dataset !!!")
    else:
        eval_nvs(config, gaussians, os.path.join(args.m, "nvs_result"))