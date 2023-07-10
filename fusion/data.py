import torch
from torch.utils.data import Dataset
import os
import math
import json
import numpy as np
import time

def find_video_index(video_list, target_video_name):
    for index, video_dict in enumerate(video_list['features']):
        if video_dict['video'] == target_video_name:
            return index
    return False 

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

class merge_features(Dataset):
    def __init__(self, frames_json, traj_json):
        self.frames_json = frames_json
        self.traj_json = traj_json
        with open(frames_json) as f:
            self.data_f = json.load(f)
        with open(traj_json) as f:
            self.data_t = json.load(f)
        
    def __len__(self):
        return len(self.data_t)
        
    def __getitem__(self, index):    
        items_t = list(self.data_t.items())
        vid_name, features_t = items_t[index]
        class_name = vid_name[:-8]
        features_t = torch.tensor(features_t)
        vid_name = vid_name+'.mp4'
        f_index = find_video_index(self.data_f, vid_name)
        features_f = []
        for clip in self.data_f['features'][f_index]['clips']:
            features_f.append(clip['features'])
        features_f = torch.tensor(features_f)
        label = lut[class_name]
        return features_t, features_f, label

# now = time.time()
# dataset = merge_features('frames.json', 'traj.json')
# print('Loaded Data', time.time() - now)
# now = time.time()
# for i in range(515):
#     x = dataset.__getitem__(i)
#     print(i, time.time() - now)
#     now = time.time()