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
sys.path.append('/mnt/sda/yuxiao_code/hlsm')
from lgp.abcd.observation import Observation
from lgp.abcd.functions.observation_function import ObservationFunction
class adap_model():
    def __init__(self):
      self.base_prompt=  "As an AI assistant, your role is to identify potential objects present within a scene. You will be provided with a high-level description of each event, along with an image. Please keep in mind that certain objects may not be visible in the current frame but could be relevant to the overall context.The high-level description is:"
      self.process_url='http://localhost:5000/process'
      base_dir = os.path.dirname(os.path.abspath(__file__))
      self.img_folder =os.path.join(base_dir,"tmp_img")
    def reset(self):
       ...
    def _save_img(self,obs):
        if isinstance(obs, Observation):
            obs = obs.rgb_image
            obs = obs[0].cpu().numpy().transpose((1, 2, 0))
            obs = (obs * 255).astype(np.uint8)
        else:obs=obs.cpu().numpy()
        
        files = [f for f in os.listdir(self.img_folder)]
        if files:
            max_number = max(int(re.sub(r'\.png$', '', f)) for f in files)
            new_number = str(max_number + 1).zfill(8) + '.png'   
        else:
            new_number = '00000000.png'
        img_path = os.path.join(self.img_folder,new_number)
        img = Image.fromarray(obs, 'RGB')
        img.save(img_path)
        return img_path

    def act(self,obs,task):
        img_path= self._save_img(obs)
        prompt =self.base_prompt+task
        msg = {'prompt':prompt,'image_file':img_path}
        responds=  requests.post(self.process_url, data=msg)
        return responds.json()['output']
       


 
# a simple test
    
if __name__ == "__main__":
    image = Image.open('/mnt/sda/yuxiao_code/hlsm/img_0.png').convert('RGB')
    task = 'take the laptop to the table.'
    transform = transforms.ToTensor()
    tensor_image = transform(image)
    predict =adap_model()
    res= predict.act(tensor_image,task)
    print(res)
