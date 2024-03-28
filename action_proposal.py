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
from LLM_subgoal.utils.LLM_utils import his_to_str,choose_examples,call_openai
sys.path.append('/mnt/sda/yuxiao_code/hlsm')
from lgp.abcd.observation import Observation
from lgp.abcd.functions.observation_function import ObservationFunction
import openai
ACTION_TYPES = [
    "OpenObject",
    "CloseObject",
    "PickupObject",
    "PutObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "SliceObject",
    "Explore",
    "Stop"
]
action_instr=f"""
The allowed types of actions are: {','.join(ACTION_TYPES)}
The target of OpenObject,CloseObject,PickupObject,ToggleObjectOn,ToggleObjectOff,SliceObject is the object agent interacts with and the target of PutObjectis the place to put the object.
Explore and Stop have no target.Note if all requirements are satisfied, you just need to output Stop\n
"""


class action_proposal():
    def __init__(self,
                 model ="gpt-4",
                 max_tokens=100,
                top_p=0.8,
                prompt_path='prompts/action_prompts.json',
                example_num=2,
                stop=['\n','.']):
        self.model = model
        self.max_tokens=max_tokens
        self.top_p=top_p
        self.baseprompt = f"""Interact with a household to solve a task. At each step, you will be provided with the previous observations and action pairs, as well as the current observation.
        You need to return an action.The answer should contain two parts, the action type and a target. f{action_instr}
        Here are {2*example_num} examples.Note that you need to put down one object before you can pick up another.\n
        """
        self.stop=stop
        #return a short example and a long example
        base_dir = os.path.dirname(os.path.abspath(__file__))
        short_examples,long_examples=choose_examples(os.path.join(base_dir,prompt_path),example_num)
        #print(short_examples)
        self.sys_prompt=self.baseprompt+''.join(short_examples)+''.join(long_examples)
    def reset(self):
        ...
    #here we not use the failed_info, but it might be used someday

#TODO now the output action are same, we need to add diversity
    def get_actions(self,task:str,history,metadata,n,failed_info=None):
        task_prompt = "Your task is: "+ task+'\n'
        str_history=his_to_str(history,metadata)
        task_prompt+=str_history
        response = call_openai(model=self.model,
                          max_token=self.max_tokens,
                          top_p=self.top_p,
                          stop=self.stop,
                          sys_prompt=self.sys_prompt,
                          user_prompt=task_prompt,
                          n=n,
                          )
        return [choice.message['content'] for choice in response.choices]
#naive test
if __name__ == "__main__":
    folder_path ="/mnt/sda/yuxiao_code/hlsm/data/rollouts/subgoal_metadata/rollout_7600.json"
    proposal = action_proposal()
    with open(folder_path,"r",encoding="utf-8")as f:
        data=json.load(f)
    history =data[:-1]
    for his in history:
        his['action']=his['subgoal']
        his['metadata']=his['predict']
    task=data[0]['task']
    metadata=data[-1]['predict']
    # print(history)
    actions = proposal.get_actions(task,history,metadata,5)
    print(actions)







    

       
