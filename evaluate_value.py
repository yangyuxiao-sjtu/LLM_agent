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

from sentence_transformers.util import cos_sim
from LLM_subgoal import sentence_embedder


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


def get_task_desc(item):
    return item["task_desc"]


def get_knn_example(task, use_predict=True, n=2):
    traj_emb = sentence_embedder.encode(task)
    topK = []
    with open(knn_data_path, "r") as f:
        knn_set = json.load(f)
    for trainItem in knn_set:
        train_emb = sentence_embedder.encode(trainItem["task"])
        dist = -1 * cos_sim(traj_emb, train_emb)
        topK.append((trainItem, dist))

    topK = sorted(topK, key=lambda x: x[1])
    topK = topK[:n]
    relvant_task = [entry[0] for entry in topK]
    prompt = ""
    for task in relvant_task:
        prompt += "Your task is:" + task["task"] + "\n"
        if use_predict:
            prompt += (
                "The objects might be useful in the tasks are:" + task["predict"] + "\n"
            )
        prompt += task["prompts"]
        prompt += "Critic:" + task["Critic"] + "\n\n"

    return prompt


def get_prompt(sample, predict_type, multi_obs=True):
    ret = ""
    if multi_obs == True:
        obs_num = 100
    else:
        obs_num = 1
    for task in sample:
        ret += "Task:" + task["task"] + "\n"
        if predict_type == "object":
            ret += (
                "The objects might be useful in the tasks are:" + task["predict"] + "\n"
            )
        prompts = task["prompts"].split("\n")
        for item in prompts:
            if "The objects you have seen are" in item and obs_num > 0:
                ret += item + "\n"
                obs_num -= 1
            elif (
                "The objects you have seen are" not in item and ":" in item
            ):  # this is action
                ret += item + "\n" + ">OK\n"

        ret += "Critic:" + task["Critic"] + "\n\n"
    return ret


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


def debug(config, name, obj=None):
    with open(config["debug"], "a") as f:
        if obj != None:
            f.write(f"{name}: {obj}\n")
        else:
            f.write(f"{name}\n")


def get_act_pair(item):
    act = item["action"].split()[0]
    act_obj = item["action"].split()[1]
    return act, act_obj


def obj_is_picked(his, obj):
    obj_emb = sentence_embedder.encode(obj)
    for item in reversed(his):
        act, act_obj = get_act_pair(item)
        if "Pick" in act and cos_sim(sentence_embedder.encode(act_obj), obj_emb) > 0.9:
            return True
        elif "Put" in act:
            return False
    return False


def get_obj_position(his):
    for item in his:
        act, act_obj = get_act_pair(item)
        if "Put" in act:
            return act_obj
    return "In hand"


def get_obj_status(his, obj, num=1):
    obj_emb = sentence_embedder.encode(obj)
    for i, item in enumerate(his):
        act, act_obj = get_act_pair(item)
        if "Pick" in act and cos_sim(sentence_embedder.encode(act_obj), obj_emb) > 0.9:
            position = get_obj_position(his[i:])


class LLM_critic:
    def __init__(
        self,
        config,
        model="llama",
        max_tokens=300,
        top_p=0.8,
        stop=["\n"],
    ):
        self.config = config

        base_path = os.path.abspath(__file__)

        base_directory = os.path.dirname(base_path)
        knn_data_path = config["value_prompt"]
        knn_data_path = os.path.join(base_directory, knn_data_path)
        with open(knn_data_path, "r") as knn:
            self.knn_set = json.load(knn)
        self.model = model
        self.use_predict = config["use_predict"]
        self.max_tokens = max_tokens
        self.prompt_path = config["value_prompt"]
        self.top_p = top_p
        self.stop = stop
        self.task = None
        self.example_num = 2
        self.base_prompt = f"""
        You are a value critic of states in a household task. You would be given a task description, some observations and actions, you need to give a critic about them.  
        {action_instr}
        Here are {self.example_num} examples:\n
        """

        self.sys_prompt = self.base_prompt
        self.log = None

    # now we don't use failed_info
    def reset(self):
        self.sys_prompt = None
        self.task = None

    def get_value(self, task, his, pddl):
        if pddl["two_object"] == True:
            ...

    def act_threads(
        self, task, his_list, failed_info=None, reflection=None, predict=None, pddl=None
    ):
        # if self.config["LLM_critic"] == False:
        #     val_ls = []
        #     for item in his_list:
        #         self.get_value(task, item, pddl)
        prompt_func = lambda a: get_prompt(
            a, self.config["predict_type"], self.config["multi_obs"]
        )
        if task != self.task:
            self.sys_prompt = self.base_prompt + knn_retriver(
                self.knn_set, get_task_desc, prompt_func, task, self.example_num
            )
            self.task = task
        task_prompt = ""
        if self.use_predict == True and predict != None:
            task_prompt += get_predict_prompt(predict, self.config["predict_type"])
        if reflection != None:
            task_prompt += (
                "Your previous memory about this task are:" + reflection + "\n"
            )
        task_prompt += (
            "Task: " + task + "\n"
        )  # + "Note that you need to put down one object before you can pick up another.\n"

        sys_prompt_ls = []
        user_prompt_ls = []
        tags = []

        for i in range(len(his_list)):
            user_prompt_ls.append(
                task_prompt
                + his_to_str(his_list[i], multi_obs=self.config["multi_obs"])
                + "Critic:"
            )

            sys_prompt_ls.append(self.sys_prompt)
            tags.append(i)
        debug(self.config, "value_prompt", sys_prompt_ls[0] + user_prompt_ls[0])
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

        for response, tag in response_list:
            if self.model == "GPT-4":
                val_ls[tag] = response.choices[0].message["content"]
            elif self.model == "llama":
                val_ls[tag] = response[0]
        debug(self.config, "critic", val_ls)
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
