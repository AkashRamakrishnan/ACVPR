import torch
from torch.utils.data import Dataset
import os
import math
import json
import numpy as np

lut = {'Abuse':0,
     'Arrest':1,
     'Arson':2,
     'Assault':3,
     'Explosion':4,
     'Fighting':5,
     'RoadAccidents':6,
     'Burglary':7,
     'Shooting':8,
     'Shoplifting':9,
     'Stealing':10,
     'Vandalism':11,
     'Robbery':12}

class traj_features(Dataset):
    def __init__(self, json_path):
        self.json_path = json_path
        
    def __len__(self):
        length = 0
        with open(self.json_path) as f:
            data = json.load(f)
        return len(data)
        
    def __getitem__(self, index):
        with open(self.json_path) as f:
            data = json.load(f)
        items = list(data.items())
        class_name, features = items[index]
        class_name = class_name[:-8]
        features = torch.tensor(features)
        label = lut[class_name]
        return features, label