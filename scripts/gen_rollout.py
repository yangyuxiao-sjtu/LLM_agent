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
import cv2

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


def get_mv(images):
    images = [torch.randn(1, 3, 300, 300) for _ in range(10)]  # 示例数据

    # 将PyTorch张量转换为OpenCV格式
    # 使用.squeeze(0)去除批量维度，然后转换为HWC格式
    frames = [
        (img.squeeze(0).numpy() * 255).astype("uint8").transpose(1, 2, 0)
        for img in images
    ]  # HWC, BGR

    # 设置视频参数
    fps = 24  # 帧率
    size = (300, 300)  # 视频尺寸，与图片尺寸一致
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 视频编码

    # 创建视频写入器
    out = cv2.VideoWriter(
        "output.mp4", fourcc, fps, (size[1], size[0])
    )  # 注意这里需要交换width和height

    # 写入帧
    for frame in frames:
        # 由于已经转换为BGR，不需要再次转换颜色空间
        out.write(frame)

    # 释放资源


if __name__ == "__main__":
    # predict =adap_model()
    read_path = "/mnt/sda/yuxiao_code/hlsm/data/rollouts/alfred_subgoal_rollouts/"
    load_path = "/mnt/sda/yuxiao_code/hlsm/data/rollouts/subgoal_metadata_n/"
    for i in range(1167, 21025):

        try:
            file_path = os.path.join(read_path, f"rollout_{i}.gz")
            save_path = os.path.join(load_path, f"rollout_{i}.json")
            roll = load(
                "/mnt/sda/yuxiao_code/hlsm/data/results/eval_hlsm_valid_seen/rollouts/rollout_Task(datasplit='valid_seen', task_id='trial_T20190906_162502_940304', repeat_idx=0).gz"
            )

            nw_ls = []
            pic = []
            for sg in roll:
                print(sg.items())

                # ls = []
                # print()
                # obs = sg["observation"].rgb_image
                # obs = obs[0].cpu().numpy().transpose((1, 2, 0))
                # obs = (obs * 255).astype(np.uint8)
                # pic.append(obs)
                # print(obs.shape)
            # get_mv(pic)
            sys.exit()
            # traj_data = sg["task"].traj_data
            # task_type = traj_data.get_task_type()

            # task = sg["task"]
            # obs = sg["observation"]
            # rep = sg["state_repr"].data.data
            # for i in range(124):
            #     if torch.any(rep[0][i] >= 1):
            #         obj_name = object_intid_to_string(i)
            #         if obj_name in _INTERACTIVE_OBJECTS:
            #             ls.append(obj_name)
            # new_item = {
            #     "task_type": task_type,
            #     "task": str(task),
            #     "predict": None,
            #     "subgoal": str(sg["subgoal"]),
            #     "action": str(sg["action"]),
            #     "obj_detector": ls,
            # }
            # nw_ls.append(new_item)
            # save(nw_ls, save_path)

        except Exception as e:
            print("errpr", e)
            if os.path.isfile(file_path):
                os.remove(file_path)
