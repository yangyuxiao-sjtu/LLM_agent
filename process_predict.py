_INTERACTIVE_OBJECTS = [
    'AlarmClock',
    'Apple',
    'ArmChair',
    'BaseballBat',
    'BasketBall',
    'Bathtub',
    'BathtubBasin',
    'Bed',
    'Blinds',
    'Book',
    'Boots',
    'Bowl',
    'Box',
    'Bread',
    'ButterKnife',
    'Cabinet',
    'Candle',
    'Cart',
    'CD',
    'CellPhone',
    'Chair',
    'Cloth',
    'CoffeeMachine',
    'CounterTop',
    'CreditCard',
    'Cup',
    'Curtains',
    'Desk',
    'DeskLamp',
    'DishSponge',
    'Drawer',
    'Dresser',
    'Egg',
    'FloorLamp',
    'Footstool',
    'Fork',
    'Fridge',
    'GarbageCan',
    'Glassbottle',
    'HandTowel',
    'HandTowelHolder',
    'HousePlant',
    'Kettle',
    'KeyChain',
    'Knife',
    'Ladle',
    'Laptop',
    'LaundryHamper',
    'LaundryHamperLid',
    'Lettuce',
    'LightSwitch',
    'Microwave',
    'Mirror',
    'Mug',
    'Newspaper',
    'Ottoman',
    'Painting',
    'Pan',
    'PaperTowel',
    'PaperTowelRoll',
    'Pen',
    'Pencil',
    'PepperShaker',
    'Pillow',
    'Plate',
    'Plunger',
    'Poster',
    'Pot',
    'Potato',
    'RemoteControl',
    'Safe',
    'SaltShaker',
    'ScrubBrush',
    'Shelf',
    'ShowerDoor',
    'ShowerGlass',
    'Sink',
    'SinkBasin',
    'SoapBar',
    'SoapBottle',
    'Sofa',
    'Spatula',
    'Spoon',
    'SprayBottle',
    'Statue',
    'StoveBurner',
    'StoveKnob',
    'DiningTable',
    'CoffeeTable',
    'SideTable',
    'TeddyBear',
    'Television',
    'TennisRacket',
    'TissueBox',
    'Toaster',
    'Toilet',
    'ToiletPaper',
    'ToiletPaperHanger',
    'ToiletPaperRoll',
    'Tomato',
    'Towel',
    'TowelHolder',
    'TVStand',
    'Vase',
    'Watch',
    'WateringCan',
    'Window',
    'WineBottle',
]
_STRUCTURAL_OBJECTS = [
    "Books",
    "Ceiling",
    "Door",
    "Floor",
    "KitchenIsland",
    "LightFixture",
    "Rug",
    "Wall",
    "StandardWallSize",
    "Faucet",
    "Bottle",
    "Bag",
    "Cube",
    "Room",
]
OBJECT_CLASSES = _STRUCTURAL_OBJECTS + _INTERACTIVE_OBJECTS 

from transformers import BertTokenizer, BertModel
import numpy as np
from scipy.spatial.distance import cosine
import torch
class predict_processor():
    def __init__(self,
                 model_name = 'bert-base-uncased',
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 threshold = 0.9):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)
        word_tokens = self.tokenizer(OBJECT_CLASSES, return_tensors='pt', padding=True, truncation=True)
        word_tokens=word_tokens.to(device)
        self.device =device
        self.threshold = 0.8
        with torch.no_grad():
            word_output = self.model(**word_tokens)
            word_embedding = word_output.last_hidden_state.mean(dim=1)
        self.word_embedding=word_embedding.detach()
        # for  cosine similarity calculate
        self.word_norms = torch.norm(self.word_embedding, dim=1, p=2, keepdim=True).view(1,-1)
    def process(self, input_text):
        return_str = False
        if isinstance(input_text, str):
            return_str = True
            input_words = input_text.replace(" ", "").split(',')
        results = []
        # make sure each word are independent
        for word in input_words:

            single_input = self.tokenizer(word, return_tensors='pt', padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                single_output = self.model(**single_input)
                single_embedding = single_output.last_hidden_state.mean(dim=1).squeeze()
            single_embedding = single_embedding.detach()

            if(single_embedding.dim() == 1):
                single_embedding = single_embedding.unsqueeze(0)

            single_norms = torch.norm(single_embedding, dim=1, p=2, keepdim=True)

            single_dot_product = torch.matmul(single_embedding, self.word_embedding.t())

            single_cosine_similarity = single_dot_product / (single_norms * self.word_norms)
            v, idx = torch.max(single_cosine_similarity, dim=1)
        
            if(v>self.threshold):
                closest_word = OBJECT_CLASSES[idx] 
                results.append(closest_word)
        if return_str ==True:
            return ", ".join(results)   
        return results

        
if __name__=="__main__":
    txt="Potato ,Pen, Dresser, Mirror, Bowl, BaseballBat, Book, CellPhone, Window, Boots, BasketBall, Pillow, GarbageCan"

    ps=predict_processor()
    res= ps.process(txt)
    print(res)

 


 

 
 

