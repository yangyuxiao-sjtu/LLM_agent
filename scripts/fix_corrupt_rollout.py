import gzip
import sys
import compress_pickle as pickle
import torch
import numpy as np
import json
import os

sys.path.append("/mnt/sda/yuxiao_code/hlsm")
from lgp.env.alfred.segmentation_definitions import (
    object_string_to_intid,
    object_intid_to_string,
)
import torch

dir_path = "/mnt/sda/yuxiao_code/hlsm/data/rollouts/alfred_subgoal_rollouts"
ava_ls = []
for i in range(21026):
    file = os.path.join(dir_path, f"rollout_{i}.gz")
    if os.path.exists(file) == True:
        ava_ls.append(i)
pth = "/mnt/sda/yuxiao_code/hlsm/data/rollouts/progress_log.json"
with open(pth, "r") as f:
    data = json.load(f)
data["collected_rollouts"] = ava_ls
with open(pth, "w") as f:
    json.dump(data, f, indent=2)
