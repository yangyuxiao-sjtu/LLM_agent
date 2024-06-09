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
from utils.LLM_utils import (
    his_to_str,
    choose_examples,
    call_llm,
    call_llm_thread,
)

prompt = """Predict the necessary components for the following household task:

- **Moveable Receptacle (mrecep_target)**: Identify any container or vessel required for the task. Return `None` if not applicable.
- **Object Slicing (object_sliced)**: Determine if the object needs to be sliced. Provide a boolean value (`True` for yes, `False` for no).
- **Parent Target (parent_target)**: Specify the final resting place for the object or its parts. Return `None` if there is no designated location.
- **Toggle Target (toggle_target)**: Indicate any appliance or device that must be toggled during the task. Return `None` if no toggling is required.
Here is two example:



 """
example = ""
pddl = "/mnt/sda/yuxiao_code/LLM_subgoal/prompts/pddl.json"
with open(pddl, "r") as f:
    data = json.load(f)
for i, item in enumerate(data):
    example += item["task_desc"][0] + "\n"
    for k, v in item["pddl"].items():
        example += k + ":"
        if v == "":
            example += "None"
        else:
            example += str(v)
        example += " "
    if i == 2:
        break
task = "You task is : Move the green sponge from the tub to the toilet\n"
ans = call_llm("llama", 150, 0.8, ["\n"], prompt + example, task, 1)
print(ans[0])
