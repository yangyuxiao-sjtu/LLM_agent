import openai
import os
import json
import sys

os.environ["OPENAI_API_KEY"] = (
    "sk-proj-jxew31rgcBtYjchHn8ziT3BlbkFJf3H5tds737YtWMTz4RS3"
)

from openai import OpenAI

sys_p = """
You will be provided with a household task roll-out conducted by an agent and a ground truth roll-out. Your task is to write a critic of the agent's roll-out based on the ground truth roll-out. The critic should follow the form:In this task, I need do the follwing things in order:... There are ... subgoals I need to achieve,My current state achieve ... , The value is a/b=...\n
Here is a example:
"""
example = """The rollout by agent is:
Your task is: Chill a slice of bread.
The objects might be useful in the tasks are:Bread, Fridge, SideTable, ButterKnife
The objects you have seen are:Apple,Bowl,Bread,Cabinet,Chair,CoffeeMachine,CounterTop,Drawer,Egg,Fork,Fridge,GarbageCan,Lettuce,Microwave,Mirror,Plate,Pot,Potato,SaltShaker,Sink,SoapBottle,Spatula,Spoon,StoveBurner,StoveKnob,DiningTable,SideTable,Toaster,Tomato,Window
 PickupObject : Bread
The objects you have seen are:Apple,Bowl,Bread,Cabinet,Chair,CoffeeMachine,CounterTop,Drawer,Egg,Fork,Fridge,GarbageCan,Lettuce,Microwave,Mirror,Plate,Pot,Potato,SaltShaker,Sink,SoapBottle,Spatula,Spoon,StoveBurner,StoveKnob,DiningTable,SideTable,Toaster,Tomato,Window
 SliceObject : Bread
The objects you have seen are:Apple,Bowl,Bread,Cabinet,Chair,CoffeeMachine,CounterTop,Drawer,Egg,Fork,Fridge,GarbageCan,Lettuce,Microwave,Mirror,Plate,Pot,Potato,SaltShaker,Sink,SoapBottle,Spatula,Spoon,StoveBurner,StoveKnob,DiningTable,SideTable,Toaster,Tomato,Window
 PutObject : SideTable
The objects you have seen are:Apple,Bowl,Bread,Cabinet,Chair,CoffeeMachine,CounterTop,Drawer,Egg,Fork,Fridge,GarbageCan,Lettuce,Microwave,Mirror,Plate,Pot,Potato,SaltShaker,Sink,SoapBottle,Spatula,Spoon,StoveBurner,StoveKnob,DiningTable,SideTable,Toaster,Tomato,Window
 PickupObject : Bread
The objects you have seen are:Apple,Bowl,Bread,Cabinet,Chair,CoffeeMachine,CounterTop,Drawer,Egg,Fork,Fridge,GarbageCan,Lettuce,Microwave,Mirror,Plate,Pot,Potato,SaltShaker,Sink,SoapBottle,Spatula,Spoon,StoveBurner,StoveKnob,DiningTable,SideTable,Toaster,Tomato,Window
 OpenObject : Fridge
The objects you have seen are:Apple,Bowl,Bread,Cabinet,Chair,CoffeeMachine,CounterTop,Drawer,Egg,Fork,Fridge,GarbageCan,Lettuce,Microwave,Mirror,Plate,Pot,Potato,SaltShaker,Sink,SoapBottle,Spatula,Spoon,StoveBurner,StoveKnob,DiningTable,SideTable,Toaster,Tomato,Window
 PutObject : Fridge
The objects you have seen are:Apple,Bowl,Bread,Cabinet,Chair,CoffeeMachine,CounterTop,Drawer,Egg,Fork,Fridge,GarbageCan,Lettuce,Microwave,Mirror,Plate,Pot,Potato,SaltShaker,Sink,SoapBottle,Spatula,Spoon,StoveBurner,StoveKnob,DiningTable,SideTable,Toaster,Tomato,Window
 CloseObject : Fridge
The objects you have seen are:Apple,Bowl,Bread,Cabinet,Chair,CoffeeMachine,CounterTop,Drawer,Egg,Fork,Fridge,GarbageCan,Lettuce,Microwave,Mirror,Plate,Pot,Potato,SaltShaker,Sink,SoapBottle,Spatula,Spoon,StoveBurner,StoveKnob,DiningTable,SideTable,Toaster,Tomato,Window
 PickupObject : Bread
The objects you have seen are:Apple,Bowl,Bread,Cabinet,Chair,CoffeeMachine,CounterTop,Drawer,Egg,Fork,Fridge,GarbageCan,Lettuce,Microwave,Mirror,Plate,Pot,Potato,SaltShaker,Sink,SoapBottle,Spatula,Spoon,StoveBurner,StoveKnob,DiningTable,SideTable,Toaster,Tomato,Window
 CloseObject : Fridge
The objects you have seen are:Apple,Bowl,Bread,Cabinet,Chair,CoffeeMachine,CounterTop,Drawer,Egg,Fork,Fridge,GarbageCan,Lettuce,Microwave,Mirror,Plate,Pot,Potato,SaltShaker,Sink,SoapBottle,Spatula,Spoon,StoveBurner,StoveKnob,DiningTable,SideTable,Toaster,Tomato,Window
 PutObject : SideTable
The objects you have seen are:Apple,Bowl,Bread,Cabinet,Chair,CoffeeMachine,CounterTop,Drawer,Egg,Fork,Fridge,GarbageCan,Lettuce,Microwave,Mirror,Plate,Pot,Potato,SaltShaker,Sink,SoapBottle,Spatula,Spoon,StoveBurner,StoveKnob,DiningTable,SideTable,Toaster,Tomato,Window
 PickupObject : Bread
The objects you have seen are:Apple,Bowl,Bread,Cabinet,Chair,CoffeeMachine,CounterTop,Drawer,Egg,Fork,Fridge,GarbageCan,Lettuce,Microwave,Mirror,Plate,Pot,Potato,SaltShaker,Sink,SoapBottle,Spatula,Spoon,StoveBurner,StoveKnob,DiningTable,SideTable,Toaster,Tomato,Window
 Stop : OutOfBounds
The ground truth rollout is: PickupObject : ButterKnife, SliceObject : Bread, PutObject : SideTable, PickupObject : Bread, OpenObject : Fridge, PutObject : Fridge, CloseObject : Fridge, OpenObject : Fridge, PickupObject : Bread, CloseObject : Fridge, PutObject : SideTable, Stop : NIL
Critic: In this task, I need do the follwing things in order: Pickup the butterknife, slice the bread, put the knife on the sidetable, pick up the bread, open the fridge, put the bread into the fridge,close the fridge, open the fridge, pickup the bread, close the fridge, put the bread on the sidetable. There are eleven subgoals I need to achieve.My current state achieve 0 of them, this is because I can't slice bread without pickup knife. The value is 0/11=0.0
"""


def call_openai(sys_prompt, user_prompt):
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


dir_path = "/mnt/sda/yuxiao_code/critic"
for i in range(0, 140):
    file_path = os.path.join(dir_path, f"{i}.json")
    x = ""
    with open(file_path, "r") as f:
        dt = json.load(f)
    tmp = dt["Critic"]
    if "Bound" in tmp:
        print(i)
    # x += "The rollout by agent is:\n" + dt["prompts"] + "\n"
    # x += "The ground truth rollout is: " + ", ".join(dt["gt"]) + "\n" + "Critic:"
    # critic = call_openai(sys_prompt=sys_p + example, user_prompt=x)
    # dt["Critic"] = critic
    # save_json(f"/mnt/sda/yuxiao_code/critic/{i}.json", dt)
