import torch
import os
import cv2
import json
import argparse
from random import shuffle
import warnings
import numpy as np
import pdb

class Dataset(torch.utils.data.Dataset):
      def __init__(self, data_json_name, image_root, saliency_root):
            data = open(data_json_name, 'r').read()
            ori_db = json.loads(data)
            
            self.image_root = image_root
            self.saliency_root = saliency_root

            self.db = []
            self.ori_imgs = []
            self.saliency_maps = []
            for i, item in enumerate(ori_db):
                  filename = os.path.split(item['url'])[-1]
                  img_path = self.image_root + filename
                  saliency_path = self.saliency_root + filename
                  isExist = os.path.exists(img_path) and os.path.exists(saliency_path)
                  # Error handling
                  if isExist:
                        self.db.append(item)
                        ori_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        self.ori_imgs.append(ori_img)
                        saliency_map = cv2.imread(saliency_path, 0)
                        self.saliency_maps.append(saliency_map)

      def __len__(self):
            #   'Denotes the total number of samples'
            return len(self.db)

      def __getitem__(self, index):
            # Select sample
            item = self.db[index]
            # Load data and get label
            ori_img = self.ori_imgs[index]
            saliency_map = self.saliency_maps[index]
            crop = item['crop']
            x = torch.tensor(crop[0] / ori_img.shape[1]).float()
            y = torch.tensor(crop[1] / ori_img.shape[0]).float()
            w = torch.tensor(crop[2] / ori_img.shape[1]).float()
            h = torch.tensor(crop[3] / ori_img.shape[0]).float()
            # ori_img_t = torch.tensor(ori_img/256).float()
            # saliency_map_t = torch.tensor(saliency_map/256).float()
            ori_img_t = torch.tensor(ori_img/128 - 1).float()
            saliency_map_t = torch.tensor(saliency_map/128 - 1).float()
            saliency_map_t = saliency_map_t.unsqueeze(dim = -1)

            return ori_img_t, saliency_map_t, x, y, w, h