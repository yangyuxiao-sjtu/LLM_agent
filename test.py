import pandas as pd

import random

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

INTERACTIVE_OBJECTS = [
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


class similar:
    def __init__(self):
        self.sentence_embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        from transformers import GPT2Tokenizer

    def knn_retrieval(self, a, b):
        # Find K train examples with closest sentence embeddings to test example
        dic = {}
        traj_emb = self.sentence_embedder.encode(a)
        topK = []
        for item in b:
            train_emb = self.sentence_embedder.encode(item)

            dist = -1 * cos_sim(traj_emb, train_emb)
            dic[item] = dist
        sorted_items = sorted(dic.items(), key=lambda item: item[1])
        top_k_items = sorted_items[:10]
        for k, v in top_k_items:
            print(k, v)


res = "In this task, I need to do the following things in order: Pick up the book, put the book on the coffee table, put the bowl on the coffee table. There are three subgoals I need to achieve. My current state achieved the first subgoal, but I failed in the second and third as I missed putting the bowl on the coffee table. Therefore, I could not complete the third either. The value is 1/3=0.33.\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n"

print(res)
# pre = similar()
# pre.knn_retrieval(a='Counter',b=INTERACTIVE_OBJECTS)
