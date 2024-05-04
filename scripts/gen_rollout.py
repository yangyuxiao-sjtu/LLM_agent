# aims to add metadata predicted by llava to rollout
import gzip
import sys
import compress_pickle as pickle
import torch
import numpy as np
import json
import os

sys.path.append("/mnt/sda/yuxiao_code/hlsm")
from lgp.env.alfred.segmentation_definitions import (
    object_string_to_intid,
    object_intid_to_string,
)
import torch

_INTERACTIVE_OBJECTS = [
    "AlarmClock",
    "Apple",
    "ArmChair",
    "BaseballBat",
    "BasketBall",
    "Bathtub",
    "BathtubBasin",
    "Bed",
    "Blinds",
    "Book",
    "Boots",
    "Bowl",
    "Box",
    "Bread",
    "ButterKnife",
    "Cabinet",
    "Candle",
    "Cart",
    "CD",
    "CellPhone",
    "Chair",
    "Cloth",
    "CoffeeMachine",
    "CounterTop",
    "CreditCard",
    "Cup",
    "Curtains",
    "Desk",
    "DeskLamp",
    "DishSponge",
    "Drawer",
    "Dresser",
    "Egg",
    "FloorLamp",
    "Footstool",
    "Fork",
    "Fridge",
    "GarbageCan",
    "Glassbottle",
    "HandTowel",
    "HandTowelHolder",
    "HousePlant",
    "Kettle",
    "KeyChain",
    "Knife",
    "Ladle",
    "Laptop",
    "LaundryHamper",
    "LaundryHamperLid",
    "Lettuce",
    "LightSwitch",
    "Microwave",
    "Mirror",
    "Mug",
    "Newspaper",
    "Ottoman",
    "Painting",
    "Pan",
    "PaperTowel",
    "PaperTowelRoll",
    "Pen",
    "Pencil",
    "PepperShaker",
    "Pillow",
    "Plate",
    "Plunger",
    "Poster",
    "Pot",
    "Potato",
    "RemoteControl",
    "Safe",
    "SaltShaker",
    "ScrubBrush",
    "Shelf",
    "ShowerDoor",
    "ShowerGlass",
    "Sink",
    "SinkBasin",
    "SoapBar",
    "SoapBottle",
    "Sofa",
    "Spatula",
    "Spoon",
    "SprayBottle",
    "Statue",
    "StoveBurner",
    "StoveKnob",
    "DiningTable",
    "CoffeeTable",
    "SideTable",
    "TeddyBear",
    "Television",
    "TennisRacket",
    "TissueBox",
    "Toaster",
    "Toilet",
    "ToiletPaper",
    "ToiletPaperHanger",
    "ToiletPaperRoll",
    "Tomato",
    "Towel",
    "TowelHolder",
    "TVStand",
    "Vase",
    "Watch",
    "WateringCan",
    "Window",
    "WineBottle",
]
# from lgp.agents.LLM_subgoal.adapt_model import adap_model
from PIL import Image


def load(rollout_path):
    return pickle.load(rollout_path)


def view_gz_file(file_path):
    try:

        with gzip.open(file_path, "rt", encoding="utf-8") as gz_file:

            content = gz_file.read()
            print(content)
    except FileNotFoundError:
        print(f"file not found: {file_path}")
    except gzip.BadGzipFile:
        print(f"not a valid gzip: {file_path}")
    except Exception as e:
        print(f"error: {e}")


def save(updated_file, rollout_path):
    with open(rollout_path, "w", encoding="utf-8") as json_file:
        json.dump(updated_file, json_file, indent=4)


if __name__ == "__main__":
    # predict =adap_model()
    read_path = "/mnt/sda/yuxiao_code/hlsm/data/rollouts/alfred_subgoal_rollouts/"
    load_path = "/mnt/sda/yuxiao_code/hlsm/data/rollouts/subgoal_metadata_n/"
    for i in range(1, 21025):

        try:
            file_path = os.path.join(read_path, f"rollout_{i}.gz")
            save_path = os.path.join(load_path, f"rollout_{i}.json")
            roll = load(file_path)

            nw_ls = []

            for sg in roll:
                print(sg["task"].get_task_id())

                ls = []
                print(sg["task"])
                traj_data = sg["task"].traj_data
                task_type = traj_data.get_task_type()

                task = sg["task"]
                obs = sg["observation"]
                rep = sg["state_repr"].data.data
                for i in range(124):
                    if torch.any(rep[0][i] >= 1):
                        obj_name = object_intid_to_string(i)
                        if obj_name in _INTERACTIVE_OBJECTS:
                            ls.append(obj_name)
                new_item = {
                    "task_type": task_type,
                    "task": str(task),
                    "predict": None,
                    "subgoal": str(sg["subgoal"]),
                    "action": str(sg["action"]),
                    "obj_detector": ls,
                }
                nw_ls.append(new_item)
                save(nw_ls, save_path)
        except Exception as e:
            print("errpr", e)
