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
import re
from transformers import AutoTokenizer
import transformers
from LLM_subgoal.utils.LLM_utils import (
    his_to_str,
    knn_retriver,
    call_llm,
    call_llm_thread,
)
from lgp.env.alfred.segmentation_definitions import (
    _INTERACTIVE_OBJECTS,
    _OPENABLES,
    _TOGGLABLES,
    _PICKABLES,
    _RECEPTACLE_OBJECTS,
    _MOVABLE_RECEPTACLES,
)
from LLM_subgoal import sentence_embedder

sys.path.append("/mnt/sda/yuxiao_code/hlsm")
from lgp.abcd.observation import Observation
from lgp.abcd.functions.observation_function import ObservationFunction


def key_func(item):
    return item[0]["task_desc"]


def get_example(sample):
    ret = ""
    for item in sample:
        ret += item[0]["task"] + "\n"
        ret += "The objects you seen are: " + item[0]["object"] + "\n"
        ret += "Predict: "
        for k, v in item[0]["pddl"].items():
            ret += k + ": "
            if v == "":
                ret += "None"
            else:
                ret += str(v)
            ret += "\n"

    return ret


def make_desc(pddl):
    ls = [
        "mrecep_target",
        "object_sliced",
        "object_target",
        "parent_target",
        "toggle_target",
        "object_state",
        "two_object",
    ]
    ret = {
        "mrecep_target": None,
        "object_sliced": False,
        "object_target": None,
        "parent_target": None,
        "toggle_target": None,
        "object_state": None,
        "two_object": False,
    }
    dict = {}
    pddl = pddl.split("\n")[:7]
    for i, item in enumerate(pddl):
        dict[ls[i]] = item.split(":")[1].strip()
        if dict[ls[i]] == "" or dict[ls[i]] == "None":
            dict[ls[i]] = None
    task_desc = ""
    num = "one"
    if (
        dict["mrecep_target"] != None
        and dict["mrecep_target"] not in _MOVABLE_RECEPTACLES
    ):
        dict["mrecep_target"] = None
    if dict["toggle_target"] != None and dict["toggle_target"] not in _TOGGLABLES:
        dict["toggle_target"] = None
    if (
        dict["parent_target"] != None
        and dict["parent_target"] not in _RECEPTACLE_OBJECTS
    ):
        dict["parent_target"] = None
    if (
        dict["two_object"] == True
        or dict["two_object"] == "True"
        or dict["two_object"] == "true"
    ):
        ret["two_object"] = True
        num = "two"
    target_obj = dict["object_target"]
    if (
        dict["object_sliced"] == "True"
        or dict["object_sliced"] == "true"
        or dict["object_sliced"] == True
    ):
        ret["object_sliced"] = True
        task_desc += f"I need to pick the knife to slice {num} {target_obj} and put down the knife on Sink/SinkBasin first. "
    else:
        task_desc += f"I need to pick up {num} {target_obj} first. "
    if dict["mrecep_target"] != None:
        ret["mrecep_target"] = dict["mrecep_target"]
        task_desc += f"Then I should  put the {target_obj} on the {dict['mrecep_target']} and pickup the {dict['mrecep_target']}. "
        target_obj = dict["mrecep_target"]
    if dict["object_state"] != None:
        if "heat" in dict["object_state"]:
            ret["object_state"] = "heat"
            task_desc += f"Then I should use  open the microwave, put the {target_obj} into microwave, close the microwave, turn on the microwave, turnoff the  microwave, open the microwave and pickup the {target_obj}, then I should close the microwave."
        elif "cool" in dict["object_state"]:
            ret["object_state"] = "cool"
            task_desc += f"Then I should use fridge to cool the {target_obj}. "
        elif "clean" in dict["object_state"]:
            ret["object_state"] = "clean"
            task_desc += f"Then I should put the {target_obj} to  Sink/SinkBasin and toggle on faucet to clean it. "
    if dict["toggle_target"] != None:
        ret["toggle_target"] = dict["toggle_target"]
        task_desc += f"Then I should toggle on the {dict['toggle_target']}. "
    if dict["parent_target"] != None:
        ret["parent_target"] = dict["parent_target"]
        task_desc += f"Then I should put {target_obj} on the {dict['parent_target']}. "
    if num == "two":
        task_desc += f"Note that I need to pick two {dict['object_target']}, but I can only hold one thing at a time, so I need to do this one by one. "
    return task_desc, ret


class adap_model:
    def __init__(self, config):
        self.config = config

        self.base_prompt = "As an AI assistant, your role is to identify potential objects present within a scene. You will be provided with a high-level description of each event, along with an image. Please keep in mind that certain objects may not be visible in the current frame but could be relevant to the overall context.The high-level description is:"
        self.process_url = "http://localhost:5000/process"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.img_folder = os.path.join(base_dir, "tmp_img")
        self.tokenizer = None
        self.pipeline = None
        knn_data_path = config["action_prompt"]
        base_path = os.path.abspath(__file__)

        base_directory = os.path.dirname(base_path)
        knn_data_path = os.path.join(base_directory, knn_data_path)
        with open(knn_data_path, "r", encoding="utf-8") as f:
            self.knn_set = json.load(f)

    def load_model(self, model="/mnt/sda/yuxiao_code/models/llama2-7b-sft"):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            device_map="balanced_low_0",
            model_kwargs={"load_in_8bit": True},
        )

    def reset(self): ...
    def _save_img(self, obs):
        if isinstance(obs, Observation):
            obs = obs.rgb_image
            obs = obs[0].cpu().numpy().transpose((1, 2, 0))
            obs = (obs * 255).astype(np.uint8)
        else:
            obs = obs.cpu().numpy()

        files = [f for f in os.listdir(self.img_folder)]
        if files:
            max_number = max(int(re.sub(r"\.png$", "", f)) for f in files)
            new_number = str(max_number + 1).zfill(8) + ".png"
        else:
            new_number = "00000000.png"
        img_path = os.path.join(self.img_folder, new_number)
        img = Image.fromarray(obs, "RGB")
        img.save(img_path)
        return img_path

    def act_obj(self, obs, task):
        if self.tokenizer == None:
            self.load_model()
        text = "You are an AI predictor. You will be given a house task and some objects you have seen. Your goal is to predict the objects you need to interact with to finish the task.\n"
        text += f"Task:{task}\n Object:{obs}."
        print("adap_model:", text)
        sequences = self.pipeline(
            text,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=100,
            truncation=True,
        )
        ans = sequences[0]["generated_text"].replace(text, "")
        #  ans = ans.replace("Object:", "")
        return ans

    def act_pddl(self, obs, task):
        example_num = 10
        prompt = f"""Predict the necessary components for the following household task:
-**Moveable Receptacle (mrecep_target)**: Identify any container or vessel required for the task. Return `None` if not applicable.
-**Object Slicing (object_sliced)**: Determine if the object needs to be sliced. Provide a boolean value (`True` for yes, `False` for no).
-**Object Target (object_target)**: Identify the specific object that is the focus of the task and will be interacted with. This could be the item that needs to be moved, cleaned, heated, cooled, sliced.
-**Parent Target (parent_target)**: Specify the final resting place for the object or its parts. Return `None` if there is no designated location.
-**Toggle Target (toggle_target)**: Indicate any appliance or device that must be toggled during the task. Return `None` if no toggling is required.
-**Object State (object_state)**: Indicate whether the target object needs to be clean, heat, or cool. Return 'None' if no such action is required.
-**Two Objects (two_object)**: Specify whether the task requires the agent to handle and place two identical objects into the parent target location. Set to True if needed, otherwise False.
Here is {example_num} example:
 """

        example = knn_retriver(self.knn_set, key_func, get_example, task, example_num)
        user_prompt = (
            task + "\n" + "The objects you seen are: " + obs + "\n" + "Predict:"
        )
        ans = call_llm("llama", 150, 0.8, None, prompt + example, user_prompt, 1)

        return make_desc(ans[0])

    def act_llm(
        self,
        obs,
        task,
    ):
        method = self.config["predict_type"]
        if method == "obj":
            return self.act_obj(obs, task)
        elif method == "pddl":
            return self.act_pddl(obs, task)

    def act_vlm(self, obs, task):
        img_path = self._save_img(obs)
        prompt = self.base_prompt + task
        msg = {"prompt": prompt, "image_file": img_path}
        responds = requests.post(self.process_url, data=msg)
        return responds.json()["output"]

    def act(self, obs, task):
        if isinstance(obs, str):
            return self.act_llm(obs, task)
        else:
            self.act_vlm(obs, task)


# a simple test
import json

if __name__ == "__main__":
    adp = adap_model()
    with open("/mnt/sda/yuxiao_code/LLM_subgoal/prompts/llm_samples.json", "r") as f:
        data = json.load(f)
    new_data = []
    for rollout in data:
        new_rollout = []
        task = rollout[0]["task"]
        for item in rollout:
            obs = item["object"]
            ret = adp.act_llm(obs, task)
            item["predict"] = ret
            new_rollout.append(item)
        new_data.append(new_rollout)
    with open("/mnt/sda/yuxiao_code/LLM_subgoal/prompts/llm_samples_n.json", "w") as f:
        json.dump(new_data, f, indent=4)
