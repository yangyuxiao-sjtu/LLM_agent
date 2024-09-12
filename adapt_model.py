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
sys.path.append("/mnt/sda/yuxiao_code/hlsm")
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


from lgp.abcd.observation import Observation
from lgp.abcd.functions.observation_function import ObservationFunction


def key_func(item):
    return item["task_desc"]


def debug(config, task, name, obj=None):
    if config["debug"] == None:
        return
    if not os.path.exists(config["debug"]):
        os.makedirs(config["debug"])
    path = os.path.join(config["debug"], task.replace("/", "_") + ".txt")
    with open(path, "a") as f:
        if obj != None:
            f.write(f"{name}: {obj}\n")
        else:
            f.write(f"{name}\n")


def get_example(sample):
    ret = ""
    for item in sample:
        ret += "Task: " + item["task"] + "\n"
        ret += "The objects you seen are: " + item["gt"][0]["object"] + "\n"
        ret += "Predict: "
        for k, v in item["pddl"].items():
            ret += k + ": "
            if v == "":
                ret += "None"
            else:
                ret += str(v)
            ret += "\n"

    return ret


def trans(pddl):
    ls = [
        "mrecep_target",
        "object_sliced",
        "object_target",
        "parent_target",
        "toggle_target",
        "object_state",
        "two_object",
    ]

    dict = {}
    pddl = pddl.strip()
    pddl = pddl.split("\n")[:7]
    for i, item in enumerate(pddl):
        dict[ls[i]] = item.split(":")[1].strip()
        if dict[ls[i]] == "" or dict[ls[i]] == "None":
            dict[ls[i]] = None

    if (
        dict["mrecep_target"] != None
        and dict["mrecep_target"] not in _MOVABLE_RECEPTACLES
    ):
        dict["mrecep_target"] = None
    if dict["toggle_target"] != None and dict["toggle_target"] not in _TOGGLABLES+ ["Faucet"]:
        dict["toggle_target"] = None
    if (
        dict["parent_target"] != None
        and dict["parent_target"] not in _RECEPTACLE_OBJECTS+ ["Sink"]+['Bathtub']
    ):
        dict["parent_target"] = None
    for k, v in dict.items():
        if v == "True" or v == "true":
            dict[k] = True
        if v == "False" or v == "false":
            dict[k] = False
    return dict


def make_desc(dict):

    ret = {
        "mrecep_target": None,
        "object_sliced": False,
        "object_target": None,
        "parent_target": None,
        "toggle_target": None,
        "object_state": None,
        "two_object": False,
    }
    task_desc = ""
    num = "one"

    if dict["two_object"] == True:
        ret["two_object"] = True
        num = "two"
    target_obj = dict["object_target"]
    if dict["object_sliced"] == True:
        ret["object_sliced"] = True
        task_desc += f"I need to pick the knife to slice {num} {target_obj} and put down the knife first. Then I should pick up {target_obj}."
    elif dict["object_target"] != None and dict["object_target"] != "None":
        task_desc += f"I need to pick up {num} {target_obj} first. "
    if dict["mrecep_target"] != None:
        ret["mrecep_target"] = dict["mrecep_target"]
        task_desc += f"Then I should  put the {target_obj} in/on the {dict['mrecep_target']} and pickup the {dict['mrecep_target']}. "
        target_obj = dict["mrecep_target"]
    if dict["object_state"] != None:
        if "heat" in dict["object_state"]:
            ret["object_state"] = "heat"
            task_desc += f"Then I should use microwave to heat the {target_obj}.*Important:Note that I need to turn on and turn off the microwave and the microwave is initially closed. After that I should pick up {target_obj} from microwave ."
        elif "cool" in dict["object_state"]:
            ret["object_state"] = "cool"
            task_desc += f"Then I should put the {target_obj} in fridge to cool it. *Important:Note that I don't need to turn on the fridge and the fridge is initially closed. After thatm I should pick up {target_obj} from fridge . "
        elif "clean" in dict["object_state"]:
            ret["object_state"] = "clean"
            task_desc += f"Then I should put the {target_obj} in Sink/SinkBasin and toggle on faucet to clean it. After that, I should pickup {target_obj}. "
    if dict["toggle_target"] != None:
        ret["toggle_target"] = dict["toggle_target"]
        if task_desc != "":
            task_desc += "Then "
        task_desc += f"I should toggle on the {dict['toggle_target']}."
        if "Lamp" in ret["toggle_target"]:
            task_desc += (
                f"*Important:Note I must hold {target_obj} in my hand while turning on the lamp."
            )
    if dict["parent_target"] != None:
        ret["parent_target"] = dict["parent_target"]
        task_desc += (
            f"Then I should put {target_obj} in/on the {dict['parent_target']}. "
        )

    if num == "two":
        task_desc += f"Important:Note that I need to pick two {dict['object_target']} one by one."
    # else:
    #     task_desc += f"*Important:Note that I can only pick one object at once, so I need to put down one object before I pick a new object."
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
        knn_data_path = config["adapt_prompt"]
        base_path = os.path.abspath(__file__)

        base_directory = os.path.dirname(base_path)
        knn_data_path = os.path.join(base_directory, knn_data_path)
        with open(knn_data_path, "r", encoding="utf-8") as f:
            self.knn_set = json.load(f)

    def load_model(self, model):
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
        example_num = 5
        prompt = f"""Predict the necessary components for the following household task:
-**Moveable Receptacle (mrecep_target)**: Identify any container or vessel required for the task. Return `None` if not applicable.
-**Object Slicing (object_sliced)**: Determine if the object needs to be sliced. Provide a boolean value (`True` for yes, `False` for no).
-**Object Target (object_target)**: Identify the specific object that is the focus of the task and will be interacted with. This could be the item that needs to be moved, cleaned, heated, cooled, sliced or examined.
-**Parent Target (parent_target)**: Specify the final resting place for the object or its parts. Return `None` if there is no designated location.
-**Toggle Target (toggle_target)**: Indicate any appliance or device that must be toggled during the task. Return `None` if no toggling is required.
-**Object State (object_state)**: Indicate whether the target object needs to be clean, heat, or cool. Return 'None' if no such action is required.
-**Two Objects (two_object)**: Specify whether the task requires the agent to handle and place two *identical* objects into the parent target location. Set to True if needed, otherwise False.  Note that this parameter should be True only when the task demands picking and placing two of the *same* items.
-**Note that the objects you need to predict might not been seen yet.
Here is {example_num} example:
 """

        example = knn_retriver(
            self.knn_set,
            key_func,
            get_example,
            task,
            example_num,
            self.config["same_ICL"],
        )
        user_prompt = (
            "Task: "
            + task
            + "\n"
            + "The objects you seen are: "
            + obs
            + "\n"
            + "Predict:"
        )
        # print(prompt + example)
        # print(user_prompt)
        debug(self.config, task, "sys_p:", prompt + example)
        debug(self.config, task, "user_p", user_prompt)

        ans = call_llm(
            self.config["model"], 150, 0.2, None, prompt + example, user_prompt, 1
        )
        debug(self.config, task, "pddl_predict", ans)
        dict = trans(ans[0])
        return make_desc(dict)

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
    with open("/mnt/sda/yuxiao_code/ALFRED_PROJECT/LLM_subgoal/config.json", "r") as f:
        config = json.load(f)
    adp = adap_model(config=config)
    ret=trans("mrecep_target: None\nobject_sliced: False\nobject_target: Pot\nparent_target: Sink\ntoggle_target: None\nobject_state: None\ntwo_object: False")
    print(ret)