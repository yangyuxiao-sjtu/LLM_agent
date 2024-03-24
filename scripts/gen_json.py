import os
import json
import sys 
sys.path.append("..") 
import process_predict
folder_path ="/mnt/sda/yuxiao_code/hlsm/data/rollouts/subgoal_metadata"
save_path = "../prompts/prompts_n.json"
json_prompts=[]
processor = process_predict.predict_processor()
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path,file_name)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
              item['predict']=processor.process(item['predict'])
        json_prompts.append(data)
    except json.JSONDecodeError as e:
                print(f"Error parsing JSON file {file_name}: {e}")
with open(save_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_prompts, json_file, indent=4)  

