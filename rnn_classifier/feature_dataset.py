import torch
from torch.utils.data import Dataset
import os
import math
import json
import numpy as np

lut = {'abuse':0,
     'arrest':1,
     'arson':2,
     'assault':3,
     'explosion':4,
     'fighting':5,
     'roadaccidents':6,
     'burglary':7,
     'shooting':8,
     'shoplifting':9,
     'stealing':10,
     'vandalism':11,
     'robbery':12}

class feature_set(Dataset):
    def __init__(self, json_path):
        with open(json_path) as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data['class_list'])
        
    def __getitem__(self, index):
        data = self.data
        class_name = data['class_list'][index]
        feature_seq = []
        for clip in data['features'][index]['clips']:
            feature_seq.append(clip['features'])
        features = torch.tensor(feature_seq)
        label = lut[class_name]
        return features, label
