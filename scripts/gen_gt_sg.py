from PIL import Image

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


def load(rollout_path):
    return pickle.load(rollout_path)


def view_gz_file(file_path):
    try:

        with gzip.open(file_path, "rt", encoding="utf-8") as gz_file:

            content = gz_file.read()
            print(content)
    except FileNotFoundError:
        print(f"file not found: {file_path}")
    except gzip.BadGzipFile:
        print(f"not a valid gzip: {file_path}")
    except Exception as e:
        print(f"error: {e}")


def save(updated_file, rollout_path):
    with open(rollout_path, "w", encoding="utf-8") as json_file:
        json.dump(updated_file, json_file, indent=4)


if __name__ == "__main__":
    # predict =adap_model()
    read_path = "/mnt/sda/yuxiao_code/hlsm/data/rollouts/alfred_subgoal_rollouts/"
    load_path = "/mnt/sda/yuxiao_code/subgoal_gt/"
    for i in range(1, 21025):

        try:
            file_path = os.path.join(read_path, f"rollout_{i}.gz")
            roll = load(file_path)
            traj =roll[0]["task"].get_task_id()
            save_path = os.path.join(load_path, f"{traj}.json")
            nw_ls = []

            for sg in roll:
                nw_ls.append(str(sg["subgoal"]).replace("HLA: ",""))
            save(nw_ls, save_path)
        except Exception as e:
            print("errpr", e)
