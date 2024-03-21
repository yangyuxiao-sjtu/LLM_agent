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
from utils.LLM_utils import his_to_str,choose_examples,call_openai
sys.path.append('/mnt/sda/yuxiao_code/hlsm')
from lgp.abcd.observation import Observation
from lgp.abcd.functions.observation_function import ObservationFunction
import openai
class predict_model():
    def __init__(self,
                model ="gpt-4",
                max_tokens=100,
                top_p=0.8,
                prompt_path='./prompts/action_prompts.json',
                example_num=1,
                stop='\n'):
        self.model = model
        self.max_tokens=max_tokens
        self.top_p=top_p
        self.stop = stop
        self.base_prompt=f"""
You are a predict model, given previous observation,action pair, you should predict next observation,here are {2*example_num} examples
"""
        short_examples,long_examples=choose_examples(prompt_path,example_num)
        self.sys_prompt=self.base_prompt+''.join(short_examples)+''.join(long_examples)
    def reset(self):
        ...
    def act(self,task,history):
        str_history = his_to_str(history)
        task_prompt ="Your task is: "+ task+'\n'
        task_prompt+=str_history+'\n'+'The objects that might help you to solve the task are:'
        response = call_openai(model=self.model,
                          max_token=self.max_tokens,
                          top_p=self.top_p,
                          stop=self.stop,
                          sys_prompt=self.sys_prompt,
                          user_prompt=task_prompt,
                          n=1
                          )
        return response.choices[0].message['content']
if __name__ == "__main__":
    folder_path ="/mnt/sda/yuxiao_code/hlsm/data/rollouts/subgoal_metadata/rollout_7600.json"
    proposal = predict_model()
    with open(folder_path,"r",encoding="utf-8")as f:
        data=json.load(f)
    history =data[:3]
    for i,his in enumerate(history):
        his['action']=his['subgoal']
        his['metadata']=his['predict']
    task=data[0]['task']
    metadata=data[3]['subgoal']
    # print(history)
    actions = proposal.act(task,history)
    print(actions)

