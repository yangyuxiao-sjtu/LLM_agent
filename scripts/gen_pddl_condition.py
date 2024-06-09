import os
import json
import sys

dir_path = "/mnt/sda/yuxiao_code/alfred/data/json_2.1.0/train"
new_data = []
for root, dirs, files in os.walk(dir_path):

    for dir in dirs:
        sub = os.path.join(dir_path, dir)
        for r, d, ff in os.walk(sub):
            for dd in d:
                file_path = os.path.join(sub, dd + "/traj_data.json")
                with open(file_path, "r") as f:
                    data = json.load(f)
                    pddl = data["pddl_params"]
                    task_id = data["task_id"]
                    task_type = data["task_type"]
                    task_desc = []
                    ann = data["turk_annotations"]["anns"]
                    for item in ann:
                        task_desc.append(item["task_desc"])
                    new_data.append(
                        {
                            "pddl": pddl,
                            "task_id": task_id,
                            "task_type": task_type,
                            "task_desc": task_desc,
                        }
                    )
    with open("/mnt/sda/yuxiao_code/LLM_subgoal/prompts/pddl.json", "w") as g:
        json.dump(new_data, g, indent=2)
