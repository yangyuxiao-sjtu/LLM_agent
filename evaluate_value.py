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
from LLM_subgoal.utils.LLM_utils import (
    his_to_str,
    choose_examples,
    call_llm,
    call_llm_thread,
)

# sys.path.append('/mnt/sda/yuxiao_code/hlsm')
# from lgp.abcd.observation import Observation
# from lgp.abcd.functions.observation_function import ObservationFunction
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
    "Stop",
]
action_instr = f"""
The allowed types of actions are: {','.join(ACTION_TYPES)}
The target of OpenObject,CloseObject,PickupObject,ToggleObjectOn,ToggleObjectOff,SliceObject is the object agent interacts with and the target of PutObjectis the place to put the object.
Explore and Stop should be fllowed with NIL.Note if all requirements are satisfied, you just need to output Stop. You might need to OpenObject so you can see the object you need to interact with\n
"""


def get_critic_example(data, use_predict):
    tmp_prompt = "Your task is: " + data[0][0]["task"] + "\n"
    if use_predict:
        tmp_prompt += (
            "The objects might be useful in the tasks are:"
            + data[0][0]["predict"]
            + "\n"
        )
    for i in range(4):
        tmp_prompt += (
            "The objects you have seen are:"
            + data[0][i]["metadata"]
            + "\n"
            + data[0][i]["subgoal"]
            + "\n"
        )
    tmp_prompt += "Critic: " + data[0][3]["critic"] + "\n"

    tmp_prompt += "Your task is" + data[1][0]["task"] + "\n"
    if use_predict:
        tmp_prompt += (
            "The objects might be useful in the tasks are:"
            + data[1][0]["predict"]
            + "\n"
        )
    eg = data[1][0]
    tmp_prompt += (
        "The objects you have seen are:" + eg["metadata"] + "\n" + eg["subgoal"] + "\n"
    )

    eg = data[1][1]
    tmp_prompt += (
        "The objects you have seen are:" + eg["metadata"] + "\n" + eg["subgoal"] + "\n"
    )
    tmp_prompt += "Critic: " + data[1][1]["critic"] + "\n"
    tmp_prompt += "Your task is" + data[2][0]["task"] + "\n"
    if use_predict:
        tmp_prompt += (
            "The objects might be useful in the tasks are:"
            + data[2][0]["predict"]
            + "\n"
        )
    for eg in data[2]:
        tmp_prompt += (
            "The objects you have seen are:"
            + eg["metadata"]
            + "\n"
            + eg["subgoal"]
            + "\n"
        )
    tmp_prompt += "Critic: " + data[2][-1]["critic"] + "\n"

    return tmp_prompt


class LLM_critic:
    def __init__(
        self,
        model="llama",
        max_tokens=150,
        top_p=0.8,
        prompt_path="prompts/value_prompts.json",
        use_predict=False,
        stop="\n",
    ):
        self.model = model
        self.use_predict = use_predict
        self.max_tokens = max_tokens
        self.prompt_path = prompt_path
        self.top_p = top_p
        self.stop = stop
        self.base_prompt = f"""
        You are a value critic of states in a household task. You would be given a task description, some observations and actions, you need to give a critic about them.  
        {action_instr}
        Here are two examples:
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_dir, prompt_path), "r", encoding="utf-8") as f:
            data = json.load(f)
        example_prompts = get_critic_example(data, self.use_predict)
        self.sys_prompt = self.base_prompt + example_prompts
        self.log = None

    # now we don't use failed_info
    def reset(self): ...
    def act_threads(
        self, task, his_list, failed_info=None, reflection=None, predict=None
    ):
        task_prompt = (
            "Your task is: "
            + task
            + "\n"
            + "Note that you need to put down one object before you can pick up another."
        )
        if self.use_predict == True and predict != None:
            task_prompt += (
                "The objects might be useful in the tasks are:" + predict + "\n"
            )
        if reflection != None:
            task_prompt += (
                "Your previous memory about this task are:" + reflection + "\n"
            )

        sys_prompt_ls = []
        user_prompt_ls = []
        tags = []
        for i in range(len(his_list)):
            user_prompt_ls.append(task_prompt + his_to_str(his_list[i]) + "Critic:")

            sys_prompt_ls.append(self.sys_prompt)
            tags.append(i)
        # print(sys_prompt_ls[0] + user_prompt_ls[0])

        response_list = call_llm_thread(
            model=self.model,
            max_token=self.max_tokens,
            top_p=self.top_p,
            sys_prompts=sys_prompt_ls,
            user_prompts=user_prompt_ls,
            tags=tags,
            stop=self.stop,
            n=1,
        )

        val_ls = [None] * len(his_list)
        print("critic_ls:", response_list)
        for response, tag in response_list:
            if self.model == "GPT-4":
                val_ls[tag] = response.choices[0].message["content"]
            elif self.model == "llama":
                val_ls[tag] = response[0]
        return val_ls

    def set_log(self, log):
        self.log = log

    def act(self, task, history, failed_info=None, reflection=None):
        task_prompt = (
            "Your task is: "
            + task
            + "\n"
            + "Note that you need to put down one object before you can pick up another."
        )
        if reflection != None:
            task_prompt += (
                "Your previous memory about this task are:" + reflection + "\n"
            )
        str_his = his_to_str(history)
        task_prompt += str_his

        response = call_llm(
            self.model,
            self.max_tokens,
            self.top_p,
            self.stop,
            self.sys_prompt,
            task_prompt,
            n=1,
        )
        return response.choices[0].message["content"]


if __name__ == "__main__":
    folder_path = (
        "/mnt/sda/yuxiao_code/hlsm/data/rollouts/subgoal_metadata/rollout_7600.json"
    )
    critic = LLM_critic()
    with open(folder_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    history = data[:1]
    for his in history:
        his["action"] = his["subgoal"].replace("HLA:", "")
        his["metadata"] = his["metadata"]
    task = data[0]["task"]
    metadata = data[2]["metadata"]
    # print(history)
    actions = critic.act(task, history)
    print(actions)
