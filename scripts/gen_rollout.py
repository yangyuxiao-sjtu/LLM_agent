#aims to add metadata predicted by llava to rollout
import gzip
import sys
import compress_pickle as pickle
import torch
import numpy as np
import json
import os
sys.path.append('/mnt/sda/yuxiao_code/hlsm')
from lgp.agents.LLM_subgoal.adapt_model import adap_model
from PIL import Image
def load(rollout_path):
    return pickle.load(rollout_path)
def view_gz_file(file_path):
    try:
     
        with gzip.open(file_path, 'rt', encoding='utf-8') as gz_file:
 
            content = gz_file.read()
            print(content)
    except FileNotFoundError:
        print(f"file not found: {file_path}")
    except gzip.BadGzipFile:
        print(f"not a valid gzip: {file_path}")
    except Exception as e:
        print(f"error: {e}")
def save(updated_file, rollout_path):
    with open(rollout_path, 'w', encoding='utf-8') as json_file:
        json.dump(updated_file, json_file, indent=4)  
 
if __name__ == "__main__":   
    predict =adap_model()
    read_path = "/mnt/sda/yuxiao_code/hlsm/data/rollouts/alfred_subgoal_rollouts/"
    load_path = "/mnt/sda/yuxiao_code/hlsm/data/rollouts/subgoal_metadata/"
    for i in range(1000,8000):
        if i%100 !=0:
            continue
        try:
            file_path=os.path.join(read_path,f'rollout_{i}.gz')
            save_path =os.path.join(load_path,f'rollout_{i}.json')
            roll=load(file_path)
            # for sg in roll:
            #     print(sg['done'])
            #     print(sg['reward'])
            nw_ls = []
            for sg in roll:
                task =sg['task']
                obs = sg['observation']
                metadata=predict.act(obs,str(task))
                new_item={'task':str(task),'predict':metadata,'subgoal':str(sg['subgoal']),'action':str(sg['action'])}
                nw_ls.append(new_item)
                save(nw_ls,save_path)
        except Exception as e:
            print('errpr',i)
