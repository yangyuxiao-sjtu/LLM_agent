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
from LLM_subgoal.utils.LLM_utils import his_to_str, choose_examples, call_llm

# sys.path.append('/mnt/sda/yuxiao_code/hlsm')
from lgp.abcd.observation import Observation
from lgp.abcd.functions.observation_function import ObservationFunction
import openai


class predict_model:
    def __init__(
        self, model="llama", max_tokens=100, top_p=0.8, example_num=2, stop="\n"
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.example_num = example_num
        self.top_p = top_p
        self.stop = stop
        self.task = None
        self.base_prompt = f"""
You are a predict model, given previous observation,action pair, you should predict next observation,here are {example_num} examples
"""

        self.sys_prompt = self.base_prompt

    def reset(self):
        self.task = None

    def act(self, task, history, predict_processor):
        if self.task != task:
            self.task = task
            task_prompt = self.base_prompt + predict_processor.knn_retrieval(
                self.task, self.example_num, use_predict=False
            )
        str_history = his_to_str(history)
        task_prompt = "Your task is: " + task + "\n"
        task_prompt += str_history + "\n" + "The objects you have seen are:"
        response = call_llm(
            model=self.model,
            max_token=self.max_tokens,
            top_p=self.top_p,
            stop=self.stop,
            sys_prompt=self.sys_prompt,
            user_prompt=task_prompt,
            n=1,
        )
        res = None
        if self.model == "GPT-4":
            res = response.choices[0].message["content"]
        elif self.model == "llama":
            res = response[0]
            print("predic:", res)
        return res


if __name__ == "__main__":
    folder_path = (
        "/mnt/sda/yuxiao_code/hlsm/data/rollouts/subgoal_metadata/rollout_7600.json"
    )
    proposal = predict_model()
    with open(folder_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    history = data[:3]
    for i, his in enumerate(history):
        his["action"] = his["subgoal"]
        his["metadata"] = his["predict"]
    task = data[0]["task"]
    metadata = data[3]["subgoal"]
    # print(history)
    actions = proposal.act(task, history)
    print(actions)
# Countertop Desk
