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
def get_critic_example(data):
    tmp_prompt ='Your task is: '+data[0][0]['task']+'\n'
    for i in range(4):
        tmp_prompt+='The objects that might help you to solve the task are:' +data[0][i]['predict']+'\n'+data[0][i]['subgoal']+'\n'
    tmp_prompt+='Critic: '+data[0][3]['critic']+'\n'
    tmp_prompt+='Your task is'+data[1][0]['task']+'\n'
    eg=data[1][0]
    tmp_prompt+='The objects that might help you to solve the task are:' +eg['predict']+'\n'+eg['subgoal']+'\n'
    tmp_prompt+='Critic: '+data[1][-1]['critic']+'\n'
    for eg in data[2]:
        tmp_prompt+='The objects that might help you to solve the task are:' +eg['predict']+'\n'+eg['subgoal']+'\n'
        tmp_prompt+='Critic: '+data[1][-1]['critic']+'\n'

    return tmp_prompt

class LLM_critic():
    def __init__(self,
                model ="gpt-4",
                max_tokens=100,
                top_p=0.8,
                prompt_path='./prompts/value_prompts.json',
                stop='\n'):
        self.model=model
        self.max_tokens = max_tokens
        self.prompt_path = prompt_path
        self.top_p=top_p
        self.stop=stop
        self.base_prompt = """
        You are a value critic of states in a household task. You would be given a task description, some observations and actions, you need to give a critic about them.  
        Here are two examples:
        """
        with open(prompt_path,"r",encoding="utf-8") as f:
            data = json.load(f)
        example_prompts= get_critic_example(data)
        self.sys_prompt= self.base_prompt+example_prompts
    #now we don't use failed_info
    def reset(self):
        ...
    def act(self,task,history,failed_info=None):
        task_prompt = "Your task is: "+ task+'\n'+'Note that you need to put down one object before you can pick up another.'

        str_his = his_to_str(history)
        task_prompt+= str_his
        print('critic_task_prompt',task_prompt)
        response= call_openai(self.model,
                              self.max_tokens,
                              self.top_p,
                              self.stop,
                              self.sys_prompt,
                              task_prompt,
                              n=1)
        return response.choices[0].message['content']
if __name__ == "__main__":
    folder_path ="/mnt/sda/yuxiao_code/hlsm/data/rollouts/subgoal_metadata/rollout_7600.json"
    critic = LLM_critic()
    with open(folder_path,"r",encoding="utf-8")as f:
        data=json.load(f)
    history =data[:1]
    for his in history:
        his['action']=his['subgoal'].replace("HLA:","")
        his['metadata']=his['predict']
    task=data[0]['task']
    metadata=data[2]['predict']
    # print(history)
    actions = critic.act(task,history)
    print(actions)

    

        
