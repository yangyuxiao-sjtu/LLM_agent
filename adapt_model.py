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


sys.path.append("/mnt/sda/yuxiao_code/hlsm")
from lgp.abcd.observation import Observation
from lgp.abcd.functions.observation_function import ObservationFunction


class adap_model:
    def __init__(self):
        self.base_prompt = "As an AI assistant, your role is to identify potential objects present within a scene. You will be provided with a high-level description of each event, along with an image. Please keep in mind that certain objects may not be visible in the current frame but could be relevant to the overall context.The high-level description is:"
        self.process_url = "http://localhost:5000/process"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.img_folder = os.path.join(base_dir, "tmp_img")
        model = "/mnt/sda/yuxiao_code/models/llama2-7b-sft"

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

    def act_llm(self, obs, task):
        text = "You are an AI predictor. You will be given a house task and some objects you have seen. Your goal is to predict the objects you need to interact with to finish the task.\n"
        text += f"Task:{task}\n Object:{obs}."
        print("adap_model:", text)
        sequences = self.pipeline(
            text,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=500,
            truncation=True,
        )
        ans = sequences[0]["generated_text"].replace(text, "")
        #  ans = ans.replace("Object:", "")
        print("predict:", ans)
        return ans

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

if __name__ == "__main__":
    obs = "AlarmClock, ArmChair, BaseballBat, Bed, Book, Boots, Box, CD, CellPhone, Chair, Desk, DeskLamp, Drawer, Dresser, FloorLamp, Fridge, GarbageCan, KeyChain, Laptop, Mirror, Mug, Pen, Pencil, Pillow, Shelf, Sofa, Statue, DiningTable, TennisRacket, Watch, Window"

    task = "Look at a bowl by the light of a lamp."
    predict = adap_model()
    res = predict.act(obs, task)
    print(res)
