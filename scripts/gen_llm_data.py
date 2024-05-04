import os
import json
import sys 
sys.path.append("..") 
 
folder_path ="/mnt/sda/yuxiao_code/hlsm/data/rollouts/subgoal_metadata_n"
save_path = "../prompts/llm_prompt.json"
json_prompts=[]
instruct = "You are an AI predictor. You will be given a house task and some objects you have seen. Your goal is to predict the objects you need to interact with to finish the task."
i = 0
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path,file_name)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        obj_ls = []
        for item in data:
            hl = item['subgoal'].replace('HLA: ','')
            obj = hl.split(":")[1].strip()
            if(obj!= None and obj != 'NIL'):
                  obj_ls.append(obj)
        obj_ls = list(set(obj_ls))
        new_data = []
        for item in data:
            Task = item['task']
            Object= ",".join(item["obj_detector"])
            subgoal = item['subgoal'].replace('HLA: ','')
            answer =','.join(obj_ls)
            new_data.append({'task_type':item['task_type'],'task':Task.replace('\n',''),'object':Object,'predict':answer,'subgoal':subgoal})
        
        json_prompts.append(new_data)
 
    except json.JSONDecodeError as e:
                print(f"Error parsing JSON file {file_name}: {e}")
with open(save_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_prompts, json_file, indent=4)  
