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
    knn_retriver,
    call_llm,
    call_llm_thread,
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


def get_predict_prompt(predict, predict_type):
    if predict_type == "object":
        return (
            "The objects might be useful in the tasks are:"
            + predict
            + "\n"
            + "Note that these predict might be wrong, you should consider carefully.\n"
        )
    elif predict_type == "pddl":
        return "Your knowledge about this task is: " + predict + "\n"


def get_task_desc(item):
    return item[0]["task_desc"]


def get_action_prompt(sample, predict_type):
    ret = ""
    for task in sample:
        ret += "Task:" + task[0]["task"] + "\n"
        if predict_type != None:
            if predict_type == "object":
                ret += (
                    "The objects might be useful in the task are:"
                    + task[0]["predict"]
                    + "\n"
                )

        for stp in task:
            ret += "The objects you have seen are:" + stp["object"] + "\n"
            ret += stp["subgoal"] + "\n"
    return ret


class action_proposal:
    def __init__(
        self,
        config,
        model="llama",
        max_tokens=100,
        top_p=0.8,
        example_num=2,
        stop=["\n", "."],
    ):
        self.use_predict = config["use_predict"]
        self.config = config
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
        knn_data_path = config["action_prompt"]
        base_path = os.path.abspath(__file__)

        base_directory = os.path.dirname(base_path)
        knn_data_path = os.path.join(base_directory, knn_data_path)
        with open(knn_data_path, "r", encoding="utf-8") as f:
            self.knn_set = json.load(f)

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
            prompt_func = lambda a: get_action_prompt(a, self.config["predict_type"])
            self.sys_prompt = self.baseprompt + knn_retriver(
                self.knn_set,
                get_task_desc,
                prompt_func,
                self.task,
                self.example_num,
            )
        task_prompt = ""
        if len(his_list) != len(metadata_list):
            print("len of his_list should equal to metadata!!")
            return None

        if reflection != None:

            task_prompt += (
                "Your previous memory about this task are:" + reflection + "\n"
            )
        if self.use_predict == True and predict != None:
            task_prompt += get_predict_prompt(predict, self.config["predict_type"])
        task_prompt += "Task: " + task + "\n"

        sys_prompt_ls = []
        user_prompt_ls = []
        tags = []

        for i in range(len(his_list)):
            user_prompt_ls.append(
                task_prompt + his_to_str(his_list[i], metadata_list[i])
            )
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
            n=n,
        )
        # print(response_list)
        acts_ls = [None] * len(his_list)
        for response, tag in response_list:
            if self.model == "GPT-4":
                acts = [ch.message["content"] for ch in response.choices]
            elif self.model == "llama":
                acts = response
                print("llama_act:", acts)
            ori_acts = list(set(acts))

            acts = predict_processor.regular_actions(ori_acts)

            if len(acts) < n:
                acts = predict_processor.gen_actions_from_predict(
                    acts, predict, his_list[tag], n
                )
            if len(acts) == 0:
                # here no feasible action is provided, we must gets some
                self._log("prompt", user_prompt_ls[tag])
                self._log("ori_acts", ori_acts)
                acts = predict_processor.regular_actions(ori_acts, 0)
                if len(acts) == 0:
                    acts = ["Stop : NIL"]

            ##this is for GPT-4
            # if len(acts) < n:
            #     acts = acts + predict_processor.gen_actions(
            #         response.choices[0].logprobs.content,
            #         sys_prompt_ls[tag],
            #         user_prompt_ls[tag],
            #         n - len(acts),
            #     )
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
        response = call_llm(
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
