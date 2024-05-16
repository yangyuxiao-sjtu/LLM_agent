import openai
import os
import json
import sys
import re

with open("/mnt/sda/yuxiao_code/LLM_subgoal/prompts/value.json", "r") as f:
    gt = json.load(f)

dir = "/mnt/sda/yuxiao_code/critic"
new_dt = []
save_path = "/mnt/sda/yuxiao_code/LLM_subgoal/prompts/value_1.json"
for item in gt:
    text = item["prompts"]
    match = re.search(r"The objects might be useful in the tasks are:(.*?)\n", text)
    if match:
        objects_part = match.group(1)
        item["predict"] = objects_part
    text = re.sub(r"Your task is: .*?\n", "", text)
    text = re.sub(r"The objects might be useful in the tasks are:(.*?)\n", "", text)
    item["prompts"] = text
    new_dt.append(item)
with open(save_path, "w") as f:
    json.dump(new_dt, f, indent=4)
