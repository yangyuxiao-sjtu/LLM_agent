import os
import json
import random
import sys 
sys.path.append("..") 
###choose 20 examples of each class, use knn retriver to get tasks which are closest to main task and give them as examples to GPT
file_path ="/mnt/sda/yuxiao_code/LLM_subgoal/prompts/llm_prompt.json"
save_path = "../prompts/llm_samples.json"
pick=[]
stack = []
pick2 = []
clean = []
heat = []
cool = []
examine = []
TASK_TYPES = [
    "look_at_obj_in_light",
    "pick_and_place_simple",
    "pick_and_place_with_movable_recep",
    "pick_clean_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_two_obj_and_place"
]

with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
for item in data:
        if  item[0]['task_type']=='pick_cool_then_place_in_recep':
            cool.append(item)
        elif item[0]['task_type']=='pick_heat_then_place_in_recep':
            heat.append(item)
        elif item[0]['task_type']=='pick_and_place_with_movable_recep':
            stack.append(item)
        elif item[0]['task_type']=='look_at_obj_in_light':
            examine.append(item)
        elif item[0]['task_type']=='pick_clean_then_place_in_recep':
            clean.append(item)
        elif item[0]['task_type']=='pick_two_obj_and_place':
            pick2.append(item)
        elif item[0]['task_type']=='pick_and_place_simple':
            pick.append(item)
lists = [cool, heat, stack, examine, clean, pick2, pick]

 
selected_items = [random.sample(lst, k=min(20, len(lst))) for lst in lists]
 
new_ls = [item for sublist in selected_items for item in sublist]
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(new_ls, f, ensure_ascii=False, indent=4)