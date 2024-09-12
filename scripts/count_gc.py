import gzip
import sys
import compress_pickle as pickle
import torch
import numpy as np
import json
import os

sys.path.append("/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm")
from lgp.env.alfred.segmentation_definitions import (
    object_string_to_intid,
    object_intid_to_string,
)
import torch
import cv2

hlsm_unseen = (
    "/mnt/sda/yuxiao_code/ori_code/hlsm/data/results/eval_hlsm_valid_unseen/rollouts"
)
hlsm_seen = (
    "/mnt/sda/yuxiao_code/ori_code/hlsm/data/results/eval_hlsm_valid_seen/rollouts"
)
seen = "//mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/eval_hlsm_valid_seen/rollouts"
deepseek_unseen = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/deepseek_eval_hlsm_valid_unseen/rollouts"
ablation = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/openai_valid_seen_ablation_look/rollouts"
unseen = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/eval_hlsm_valid_unseen/rollouts"
gpt_seen = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/openai_valid_seen_hind/rollouts"

no_adapt = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/openai_valid_seen_no_adapt/rollouts"


def load(rollout_path):
    return pickle.load(rollout_path)


few_shot_ablation = (
    "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_aba3/rollouts"
)
few_shot_act_hind = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_few_act_hind/rollouts"
few_shot_critic_hind = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_critic_hind2/rollouts"
few_shot_hind = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_act_hind3/rollouts"
ori = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/deep_seek_eval_hlsm_valid_unseen/rollouts"

few_shot_relable_critic_aba = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_aba_finetune_critic/rollouts"
few_shot_relable_aba4 = (
    "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_aba_4act/rollouts"
)
valid_seen_aba1 = (
    "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_aba_full/rollouts"
)
valid_seen_aba2 = (
    "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_aba2/rollouts"
)

valid_seen_aba_relable_critic_4samples = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_aba_full_relable_critic/rollouts"
valid_seen_aba_relable_critic_2samples = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_full_aba_relable_2samples/rollouts"
valid_seen_hind = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_actor_critic_hind_full/rollouts"
valid_seen_critic = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_critic_hind_full/rollouts"
valid_seen_no_adapt = (
    "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_no_adapt/rollouts"
)
valid_unseen_aba = (
    "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_unseen_aba/rollouts"
)
valid_unseen_hind = (
    "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_unseen_hind/rollouts"
)
valid_unseen_no_adapt = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_unseen_no_adapt/rollouts"
gt = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_gt/rollouts"
SR = [0, 0]
valid_seen_gt = (
    "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_gt/rollouts"
)
valid_seen_aba_final = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_aba_relabel/rollouts"
valid_seen_full_aba_relable = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_aba_relabel/rollouts"
valid_seen_few_aba_relable = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_few_aba_relabel/rollouts"
valid_seen_few_gt_relable = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_few_gt_aba_relabel/rollouts"
valid_seen_few_gt_hind = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_few_gt_hind/rollouts"
valid_seen_full_gt_hind = (
    "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_seen_gt_hind/rollouts"
)
valid_unseen_full_gt_hind = "/mnt/sda/yuxiao_code/ALFRED_PROJECT/hlsm/data/results/valid_unseen_gt_hind/rollouts"
dir = valid_unseen_hind
dict = {}
complete = {}
cnt = 0
for _, __, files in os.walk(dir):
    for i, file in enumerate(files):
        path = os.path.join(dir, file)
        try:
            dt = load(path)
        except Exception as e:
            print(e)
            continue

        task = dt[-1]["task"].get_task_type()
        if str(task) not in dict:
            dict[str(task)] = [0, 0]
            complete[str(task)] = [0, 0]
        if "clean" in str(task):
            cnt += 1
            if cnt > 160:
                continue
        dict[str(task)][0] += dt[-1]["md"]["goal_conditions_met"][0]
        dict[str(task)][1] += dt[-1]["md"]["goal_conditions_met"][1]
        complete[str(task)][1] += 1
        SR[1] += 1
        if (
            dt[-1]["md"]["goal_conditions_met"][1]
            == dt[-1]["md"]["goal_conditions_met"][0]
        ):
            complete[str(task)][0] += 1
            SR[0] += 1
GC = [0, 0]
for k, v in dict.items():
    print(k)
    print("GC:", v[0] / v[1])
    GC[0] += v[0]
    GC[1] += v[1]
    print("SR:", complete[k][0] / complete[k][1])
    print("num:", complete[k][1])
print("TOTAL GC:", GC[0] / GC[1])
print("TOTAL SR:", SR[0] / SR[1])
