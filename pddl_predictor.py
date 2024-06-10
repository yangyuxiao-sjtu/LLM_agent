import requests
from PIL import Image
from io import BytesIO
import os
import numpy as np
import torch
import torchvision.io as io
from torchvision import transforms
from torchvision.utils import save_image
import sys
import math
from statistics import mean
import json
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

sentence_embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
from utils.LLM_utils import (
    his_to_str,
    choose_examples,
    call_llm,
    call_llm_thread,
)

n = 10
prompt = f"""Predict the necessary components for the following household task:
-**Moveable Receptacle (mrecep_target)**: Identify any container or vessel required for the task. Return `None` if not applicable.
-**Object Slicing (object_sliced)**: Determine if the object needs to be sliced. Provide a boolean value (`True` for yes, `False` for no).
-**Object Target (object_target)**: Identify the specific object that is the focus of the task and will be interacted with. This could be the item that needs to be moved, cleaned, heated, cooled, sliced.
-**Parent Target (parent_target)**: Specify the final resting place for the object or its parts. Return `None` if there is no designated location.
-**Toggle Target (toggle_target)**: Indicate any appliance or device that must be toggled during the task. Return `None` if no toggling is required.
-**Object State (object_state)**: Indicate whether the target object needs to be clean, heat, or cool. Return 'None' if no such action is required.
-**Two Objects (two_object)**: Specify whether the task requires the agent to handle and place two identical objects into the parent target location. Set to True if needed, otherwise False.
Here is {n} example:
 """
example = ""
pddl = "/mnt/sda/yuxiao_code/LLM_subgoal/prompts/pddl_n.json"


def knn_retriver(data, key_func, get_prompt, input, n):
    encoded = sentence_embedder.encode(input)
    ls = []
    for item in data:
        cmp_list = key_func(item)
        if isinstance(cmp_list, str):
            cmp_list = [cmp_list]
        dist = 0
        for cmp in cmp_list:
            tmp = cos_sim(sentence_embedder.encode(cmp), encoded)
            if tmp > dist:
                dist = tmp
        ls.append((item, dist))
    top_k = sorted(ls, key=lambda x: x[1], reverse=True)
    top_k = top_k[:n]
    ret = [item for (item, _) in top_k]
    knn_prompt = get_prompt(ret)
    return knn_prompt


def key_func(item):
    return item[0]["task_desc"]


def get_example(sample):
    ret = ""
    for item in sample:
        ret += item[0]["task"] + "\n"
        ret += "The objects you seen are: " + item[0]["object"] + "\n"
        ret += "Predict:\n"
        for k, v in item[0]["pddl"].items():
            ret += k + ": "
            if v == "":
                ret += "None"
            else:
                ret += str(v)
            ret += "\n"
        ret += "\n"
    return ret


def make_desc(pddl):
    ls = [
        "mrecep_target",
        "object_sliced",
        "object_target",
        "parent_target",
        "toggle_target",
        "object_state",
        "two_object",
    ]
    dict = {}
    pddl = pddl.split("\n")[:7]
    for i, item in enumerate(pddl):
        dict[ls[i]] = item.split(":")[1].strip()
        if dict[ls[i]] == "" or dict[ls[i]] == "None":
            dict[ls[i]] = None
    task_desc = ""
    num = "one"
    print(dict)
    if (
        dict["two_object"] == True
        or dict["two_object"] == "True"
        or dict["two_object"] == "true"
    ):
        num = "two"
    target_obj = dict["object_target"]
    if (
        dict["object_sliced"] == "True"
        or dict["object_sliced"] == "true"
        or dict["object_sliced"] == True
    ):
        task_desc += f"I need to pick the knife to slice {num} {target_obj} and put down the knife first."
    else:
        task_desc += f"I need to pick up {num} {target_obj} first."
    if dict["mrecep_target"] != None:
        task_desc += f"Then I need to put the {target_obj} on the {dict['mrecep_target']} and pickup the {dict['mrecep_target']}."
        target_obj = dict["mrecep_target"]
    if dict["object_state"] != None:
        if "heat" in dict["object_state"]:
            task_desc += f"Then I need to use microwave to heat the {target_obj}."
        elif "cool" in dict["object_state"]:
            task_desc += f"Then I need to use fridge to cool the {target_obj}."
        elif "clean" in dict["object_state"]:
            task_desc += f"Then I need to clean the {target_obj}."
    if dict["toggle_target"] != None:
        task_desc += f"Then I need to toggle on the {dict['toggle_target']}."
    if dict["parent_target"] != None:
        task_desc += f"Then I should put {target_obj} on the {dict['parent_target']}."
    if num == "two":
        task_desc += f"Note that I need to pick two {dict['object_target']}, but I can only hold one thing at a time, so I need to do this one by one."
    return task_desc


with open(pddl, "r") as f:
    data = json.load(f)

task = "Examine a glass bowl using the light from a floor lamp."
obs = "AlarmClock,BaseballBat,Bed,Book,Box,CD,CellPhone,Chair,Cloth,CreditCard,Desk,DeskLamp,Drawer,Dresser,GarbageCan,KeyChain,Laptop,Mirror,Mug,Pen,Pencil,Pillow,Poster,Shelf,SideTable,TennisRacket,Vase,Window"
example = knn_retriver(data, key_func, get_example, task, n)

user_prompt = task + "\n" + "The objects you seen are: " + obs + "\n" + "Predict:\n"
print(prompt + example)
print("--" * 13)
print(user_prompt)
ans = call_llm("llama", 150, 0.8, None, prompt + example, user_prompt, 1)
print(ans[0])
print(make_desc(ans[0]))
