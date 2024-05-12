import sys

sys.path.append("/mnt/sda/yuxiao_code/hlsm")
from lgp.env.alfred.segmentation_definitions import (
    _INTERACTIVE_OBJECTS,
    _OPENABLES,
    _TOGGLABLES,
    _PICKABLES,
    _RECEPTACLE_OBJECTS,
)

OBJECT_CLASSES = _INTERACTIVE_OBJECTS
import re
from transformers import BertTokenizer, BertModel
import numpy as np
from scipy.spatial.distance import cosine
import torch

from .utils.LLM_utils import call_llm, call_llm_thread
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import json
import os
from typing import Union, List

_ACTIONS = [
    "PickupObject",
    "PutObject",
    "OpenObject",
    "CloseObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "SliceObject",
    "Stop",
]


class predict_processor:
    def __init__(
        self,
        llama_model_name="meta-llama/Llama-2-7b-chat-hf",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        threshold=0.8,
    ):
        self.sentence_embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.word_embedding = []
        knn_data_path = "prompts/llm_samples.json"
        base_path = os.path.abspath(__file__)

        base_directory = os.path.dirname(base_path)

        knn_data_path = os.path.join(base_directory, knn_data_path)

        with open(knn_data_path, "r", encoding="utf-8") as f:
            self.knn_set = json.load(f)
        for obj in OBJECT_CLASSES:
            self.word_embedding.append(self.sentence_embedder.encode(obj))

        self.threshold = threshold

    def process(self, input_text, threshold=None):
        return_str = False
        if threshold == None:
            threshold = self.threshold
        if isinstance(input_text, str):
            return_str = True
            input_words = input_text.replace(" ", "").split(",")
        elif isinstance(input_text, list):
            input_words = input_text
        else:
            assert "input of process should be str or list!"
            return
        results = []

        for word in input_words:
            single_input = self.sentence_embedder.encode(word)
            similarity = 0
            idx = None
            for i, obj_emb in enumerate(self.word_embedding):
                dist = cos_sim(obj_emb, single_input)
                if dist > similarity:
                    idx = i
                    similarity = dist

            if similarity > threshold:
                closest_word = OBJECT_CLASSES[idx]
                results.append(closest_word)
        if return_str == True:
            return ", ".join(results)
        return results

    def regular_input(self, input, allowed_set: Union[str, List[str]], threshold=0):
        if isinstance(allowed_set, str):
            if allowed_set == "PickupObject" or allowed_set == "SliceObject":
                allowed_set = _PICKABLES
            elif allowed_set == "OpenObject" or allowed_set == "CloseObject":
                allowed_set = _OPENABLES
            elif allowed_set == "ToggleObjectOn" or allowed_set == "ToggleObjectOff":
                allowed_set = _TOGGLABLES
            elif allowed_set == "PutObject":
                allowed_set = _RECEPTACLE_OBJECTS
            else:
                allowed_set = ",".split(allowed_set)
        word_emb = self.sentence_embedder.encode(input)
        similarity = 0
        idx = None
        for i, allowed_word in enumerate(allowed_set):
            emb = self.sentence_embedder.encode(allowed_word)
            dist = cos_sim(word_emb, emb)
            if dist > similarity and dist > threshold:
                similarity = dist
                idx = i
        if similarity > 0:

            return allowed_set[idx]
        else:
            return None

    def regular_actions(self, actions):
        new_actions = []
        for act_pair in actions:
            act_pair = act_pair.strip()
            act_pair = act_pair.split(":")
            act = self.regular_input(act_pair[0], _ACTIONS)
            if act == "Stop":
                new_actions.append("Stop: NIL")
            else:
                obj = self.regular_input(act_pair[1], act, 0.8)
                if obj != None:
                    new_actions.append(f"{act}: {obj}")
        return new_actions

    def process_prefix(self, word, k, threshold):
        OBJ_lS = _INTERACTIVE_OBJECTS
        if k == "PickupObject" or k == "SliceObject":
            OBJ_lS = _PICKABLES
        elif k == "PutObject":
            OBJ_lS = _RECEPTACLE_OBJECTS
        elif k == "OpenObject" or k == "CloseObject":
            OBJ_lS = _OPENABLES
        elif k == "ToggleObjectOn" or k == "ToggleObjectOff":
            OBJ_lS = _TOGGLABLES

        for obj in OBJ_lS:
            if re.match(r"^" + re.escape(word), obj) is not None:
                return obj
        return None

    def knn_retrieval(self, curr_task, k, use_predict=True):
        # Find K train examples with closest sentence embeddings to test example

        traj_emb = self.sentence_embedder.encode(curr_task)
        topK = []
        for trainItem in self.knn_set:
            train_emb = self.sentence_embedder.encode(trainItem[0]["task"])
            dist = -1 * cos_sim(traj_emb, train_emb)
            topK.append((trainItem, dist))

        topK = sorted(topK, key=lambda x: x[1])
        topK = topK[:k]
        relvant_task = [entry[0] for entry in topK]
        prompt = ""
        for task in relvant_task:

            prompt += "Your task is: " + task[0]["task"] + "\n"
            if use_predict:
                prompt += (
                    "The objects might be useful in the tasks are:"
                    + task[0]["predict"]
                    + "\n"
                    + "Note that these predict might be wrong, you should consider carefully.\n"
                )
            for item in task:
                prompt += "The objects you have seen are:" + item["object"] + "\n"
                prompt += item["subgoal"] + "\n"
        return prompt

    ## generate possbile action from top_logprobs of LLM's response
    def gen_actions_from_predict(self, ori_acts, predict, his, n):
        is_picked = False
        past_actions = []
        if predict == [] or predict == None or predict == "":
            return ori_acts
        if isinstance(predict, str):
            predict = predict.split(",")
        for item in his:
            regulared_act = self.regular_actions([item["action"]])[0]
            if "PickupObject" in regulared_act:
                is_picked = True
            if "PutObject" in regulared_act:
                is_picked = False
            past_actions.append(regulared_act)
        for obj in predict:
            if obj in _PICKABLES and is_picked == False:
                new_act = "PickupObject: " + obj
                ori_acts.append(new_act)
            elif obj in _RECEPTACLE_OBJECTS and is_picked == True:
                new_act = "PutObject: " + obj
                ori_acts.append(new_act)
            elif obj in _OPENABLES:
                new_act = "OpenPbject: " + obj
                if new_act in past_actions:
                    new_act = "CloseObject: " + obj
                ori_acts.append(new_act)
            elif obj in _TOGGLABLES:
                new_act = "ToggleObjectOn: " + obj
                ori_acts.append(new_act)
            if len(ori_acts) == n:
                return ori_acts
        return ori_acts

    def gen_actions(self, data, sys_prompt, task_prompt, N):
        possible_actions = []
        possible_obj = []
        adm_actions_dict = {}

        is_obj = False
        for tk in data:
            if ":" in tk["token"]:
                is_obj = True
                continue
            if is_obj == False:
                possible_actions.append(tk["top_logprobs"])
            else:
                possible_obj.append([tk["top_logprobs"]])

        if len(possible_actions) >= 1:
            for tk in possible_actions[0]:

                if len(adm_actions_dict) == 3:
                    break
                if tk["token"] == "Pick":
                    adm_actions_dict["PickupObject"] = {"prob": tk["logprob"]}

                elif tk["token"] == "Put":
                    adm_actions_dict["PutObject"] = {"prob": tk["logprob"]}
                elif tk["token"] == "Slice":
                    adm_actions_dict["SliceObject"] = {"prob": tk["logprob"]}
                elif tk["token"] == "Open":
                    adm_actions_dict["OpenObject"] = {"prob": tk["logprob"]}
                elif tk["token"] == "Close":
                    adm_actions_dict["CloseObject"] = {"prob": tk["logprob"]}
                elif tk["token"] == "Stop":
                    adm_actions_dict["Stop"] = {"prob": tk["logprob"]}
                # explore seems not helpful in the task
                # elif tk['token'] == 'Explore':
                #     adm_actions_dict['Explore'] =  {'prob':tk['logprob']}
                elif tk["token"] == "Toggle":
                    res = call_llm(
                        model="gpt-4",
                        sys_prompt=sys_prompt,
                        user_prompt=task_prompt + "Toggle",
                    )
                    tk1 = res.choices[0].logprobs.content[1].top_logprobs
                    if tk1 and "token" in tk1:
                        if tk1["token"] == "On":
                            adm_actions_dict["ToggleObjectOn"] = {"prob": tk["logprob"]}
                        elif tk1["token"] == "Off":
                            adm_actions_dict["ToggleObjectOff"] = {
                                "prob": tk["logprob"]
                            }
        actions = {}
        if len(adm_actions_dict) > 0:
            sys_prompts = []
            user_prompts = []
            tags = []
            for k, v in adm_actions_dict.items():
                # print(k,v)
                sys_prompts.append(sys_prompt)
                user_prompts.append(task_prompt + k + " :")
                tags.append((k, v))
            # as we can't know which thread stop first, so add tag to identify it
            results = call_llm_thread(
                model="gpt-4",
                max_token=100,
                top_p=0.8,
                sys_prompts=sys_prompts,
                user_prompts=user_prompts,
                tags=tags,
            )
            for res, tag in results:
                res = res.choices[0].logprobs.content[0].top_logprobs
                # print(task_prompt+k+" : ")
                k, v = tag
                for tk in res:

                    if len(tk["token"]) < 2:
                        continue

                    obj = self.process_prefix(tk["token"], k, threshold=0.5)

                    if obj == None:
                        continue
                    action = k + " :" + obj
                    if action not in actions:
                        actions[action] = tk["logprob"] + v["prob"]
                        # print(action,actions[action])

            sorted_actions = sorted(
                actions.items(), key=lambda item: item[1], reverse=True
            )
            top_N_actions = sorted_actions[:N]
            ret = [k for k, v in top_N_actions]
            if ret == None:
                ret = []
            return ret


if __name__ == "__main__":
    txt = "Potato ,Pen, Dresser, Mirror, Bowl, BaseballBat, Book, CellPhone, Window, Boots, BasketBall, Pillow, GarbageCan,Counter,Banana"
    data = [
        {
            "token": "Pick",
            "logprob": -0.002162279,
            "bytes": [80, 105, 99, 107],
            "top_logprobs": [
                {"token": "Pick", "logprob": -0.002162279, "bytes": [80, 105, 99, 107]},
                {"token": "Put", "logprob": -6.1427875, "bytes": [80, 117, 116]},
                {"token": "Stop", "logprob": -11.830287, "bytes": [83, 116, 111, 112]},
                {"token": "Open", "logprob": -13.861537, "bytes": [79, 112, 101, 110]},
            ],
        },
        {
            "token": "up",
            "logprob": -1.6240566e-06,
            "bytes": [117, 112],
            "top_logprobs": [
                {"token": "up", "logprob": -1.6240566e-06, "bytes": [117, 112]},
                {"token": "Up", "logprob": -13.484377, "bytes": [85, 112]},
                {"token": "UP", "logprob": -16.765627, "bytes": [85, 80]},
                {"token": "u", "logprob": -19.234377, "bytes": [117]},
            ],
        },
        {
            "token": "Object",
            "logprob": -3.1281633e-07,
            "bytes": [79, 98, 106, 101, 99, 116],
            "top_logprobs": [
                {
                    "token": "Object",
                    "logprob": -3.1281633e-07,
                    "bytes": [79, 98, 106, 101, 99, 116],
                },
                {
                    "token": "object",
                    "logprob": -15.4375,
                    "bytes": [111, 98, 106, 101, 99, 116],
                },
                {
                    "token": " Object",
                    "logprob": -16.671875,
                    "bytes": [32, 79, 98, 106, 101, 99, 116],
                },
                {"token": "Ob", "logprob": -17.765625, "bytes": [79, 98]},
            ],
        },
        {
            "token": " :",
            "logprob": -1.7432603e-06,
            "bytes": [32, 58],
            "top_logprobs": [
                {"token": " :", "logprob": -1.7432603e-06, "bytes": [32, 58]},
                {"token": ":", "logprob": -13.281252, "bytes": [58]},
                {"token": " ", "logprob": -18.187502, "bytes": [32]},
                {"token": " :\n", "logprob": -20.265627, "bytes": [32, 58, 10]},
            ],
        },
        {
            "token": " Apple",
            "logprob": -0.0015568782,
            "bytes": [32, 65, 112, 112, 108, 101],
            "top_logprobs": [
                {
                    "token": " Apple",
                    "logprob": -0.0015568782,
                    "bytes": [32, 65, 112, 112, 108, 101],
                },
                {
                    "token": " Slice",
                    "logprob": -6.954682,
                    "bytes": [32, 83, 108, 105, 99, 101],
                },
                {"token": " Pot", "logprob": -8.423432, "bytes": [32, 80, 111, 116]},
                {"token": " S", "logprob": -8.501557, "bytes": [32, 83]},
            ],
        },
        {
            "token": "Slice",
            "logprob": -0.86832184,
            "bytes": [83, 108, 105, 99, 101],
            "top_logprobs": [
                {"token": "S", "logprob": -0.55582184, "bytes": [83]},
                {
                    "token": "Slice",
                    "logprob": -0.86832184,
                    "bytes": [83, 108, 105, 99, 101],
                },
                {"token": "<|end|>", "logprob": -5.462072, "bytes": "null"},
                {
                    "token": ".Slice",
                    "logprob": -6.930822,
                    "bytes": [46, 83, 108, 105, 99, 101],
                },
            ],
        },
    ]
    ps = predict_processor()
    res = ps.regular_input("TableLamp", _INTERACTIVE_OBJECTS, 0.5)
    print(res)
