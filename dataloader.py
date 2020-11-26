import torch
import os
import cv2
import json
import argparse
from random import shuffle
import warnings
import numpy as np

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_json_name, image_root, saliency_root):
        'Initialization'
        data = open(data_json_name, 'r').read()
        self.db = json.loads(data)
        self.image_root = image_root
        self.saliency_root = saliency_root

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.db)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        item = self.db[index]

        # Load data and get label
        filename = os.path.split(item['url'])[-1]
        img_path = self.image_root + filename
        saliency_path = self.saliency_root + filename
        ori_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        saliency_map = cv2.imread(saliency_path, 0)
        crop = item['crop']
        x = torch.tensor(crop[0] / ori_img.shape[0]).float()
        y = torch.tensor(crop[1] / ori_img.shape[1]).float()
        w = torch.tensor(crop[2] / ori_img.shape[0]).float()
        h = torch.tensor(crop[3] / ori_img.shape[1]).float()
        ori_img_t = torch.tensor([ori_img/256]).float()
        saliency_map_t = torch.tensor([saliency_map/256]).float()

        return ori_img_t, saliency_map_t, x, y, w, h