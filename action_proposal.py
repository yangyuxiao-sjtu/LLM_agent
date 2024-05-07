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
    call_openai,
    call_openai_thread,
)


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
    "Stop",
]
action_instr = f"""
The allowed types of actions are: {','.join(ACTION_TYPES)}
The target of OpenObject,CloseObject,PickupObject,ToggleObjectOn,ToggleObjectOff,SliceObject is the object agent interacts with and the target of PutObjectis the place to put the object.
Stop should end with NIL.Note if all requirements are satisfied, you just need to output Stop\n
"""


class action_proposal:
    def __init__(
        self,
        model="gpt-4",
        max_tokens=100,
        top_p=0.8,
        example_num=2,
        use_predict=True,
        stop=["\n", "."],
    ):
        self.use_predict = use_predict
        self.model = model
        self.task = None
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.baseprompt = f"""Interact with a household to solve a task. At each step, you will be provided with the previous observations and action pairs.
        You need to return an action.The answer should contain two parts, the action type and a target. {action_instr}
        Here are {example_num} examples. \n
        """
        self.example_num = example_num
        self.stop = stop
        self.log = None
        # return a short example and a long example
        base_dir = os.path.dirname(os.path.abspath(__file__))

    def reset(self):
        self.task = None
        self.sys_prompt = self.baseprompt

    def set_log(self, log):
        self.log = log

    def _log(self, name, obj=None):
        with open(self.log, "a") as f:
            if obj != None:
                f.write(f"{name}: {obj}\n")
            else:
                f.write(f"{name}\n")

    # here we not use the failed_info, but it might be used someday

    def get_actions_threads(
        self,
        task,
        his_list,
        metadata_list,
        n,
        predict_processor,
        failed_info=None,
        reflection=None,
        predict=None,
    ):
        if self.task != task:
            self.task = task
            self.sys_prompt = self.baseprompt + predict_processor.knn_retrieval(
                task, self.example_num
            )
        if len(his_list) != len(metadata_list):
            print("len of his_list should equal to metadata!!")
            return None
        task_prompt = "Your task is: " + task + "\n"
        if self.use_predict == True and predict != None:
            task_prompt += (
                "The objects might be useful in the tasks are:"
                + predict
                + "\n"
                + "Note that these predict might be wrong, you should consider carefully.\n"
            )
        if reflection != None:

            task_prompt += (
                "Your previous memory about this task are:" + reflection + "\n"
            )
        sys_prompt_ls = []
        user_prompt_ls = []
        tags = []
        for i in range(len(his_list)):
            user_prompt_ls.append(
                task_prompt + his_to_str(his_list[i], metadata_list[i])
            )
            sys_prompt_ls.append(self.sys_prompt)
            tags.append(i)
        print(sys_prompt_ls[0] + user_prompt_ls[0])
        sys.exit()
        response_list = call_openai_thread(
            model=self.model,
            max_token=self.max_tokens,
            top_p=self.top_p,
            sys_prompts=sys_prompt_ls,
            user_prompts=user_prompt_ls,
            tags=tags,
            stop=self.stop,
            n=n,
        )
        # print(response_list)
        acts_ls = [None] * len(his_list)
        for response, tag in response_list:
            acts = [ch.message["content"] for ch in response.choices]
            acts = list(set(acts))

            acts = predict_processor.regular_actions(acts)
            print(acts)
            if len(acts) < n:
                acts = acts + predict_processor.gen_actions(
                    response.choices[0].logprobs.content,
                    sys_prompt_ls[tag],
                    user_prompt_ls[tag],
                    n - len(acts),
                )
            while len(acts) < n:
                # to ensure we would have n actions
                acts.append(acts[0])
            acts_ls[tag] = acts
        self._log("acts", acts_ls)
        return acts_ls

    def get_actions(self, task: str, history, metadata, n, failed_info=None):
        task_prompt = "Your task is: " + task + "\n"
        str_history = his_to_str(history, metadata)
        task_prompt += str_history
        response = call_openai(
            model=self.model,
            max_token=self.max_tokens,
            top_p=self.top_p,
            stop=self.stop,
            sys_prompt=self.sys_prompt,
            user_prompt=task_prompt,
            n=n,
        )
        return [choice.message["content"] for choice in response.choices]


# naive test
if __name__ == "__main__":
    folder_path = (
        "/mnt/sda/yuxiao_code/hlsm/data/rollouts/subgoal_metadata/rollout_7600.json"
    )
    proposal = action_proposal()
    with open(folder_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    history = data[:-1]
    for his in history:
        his["action"] = his["subgoal"]
        his["metadata"] = his["predict"]
    task = data[0]["task"]
    metadata = data[-1]["predict"]
    # print(history)
    actions = proposal.get_actions(task, history, metadata, 5)
    print(actions)
