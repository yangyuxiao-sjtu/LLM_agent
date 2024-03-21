import os
import json
folder_path ="../subgoal_metadata"
save_path = "../prompts/prompts.json"
json_prompts=[]
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path,file_name)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        json_prompts.append(data)
    except json.JSONDecodeError as e:
                print(f"Error parsing JSON file {file_name}: {e}")
with open(save_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_prompts, json_file, indent=4)  

