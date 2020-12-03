from model import *
from dataloader import *
import torch
import os
import cv2
import json
import argparse
from random import shuffle
import warnings
import numpy as np
import pdb


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 6}

def inspect(filename = "2157516909_de12572d95_b.jpg"):
    # load validation set
    image_root = "./flickr-cropping-dataset/data/"
    saliency_root = "./flickr-cropping-dataset/saliency_map/"
    img_path = image_root + filename
    saliency_path = saliency_root + filename
    

    model = CropProposalModel().to(device)
    model.load_state_dict(torch.load('./si670_final_project/saved_model/best_model.pt'))
    print('Model loaded')
    model.eval()
    ori_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    saliency_map = cv2.imread(saliency_path, 0)
    ori_img_t = torch.tensor(ori_img/128 - 1).float().to(device)
    saliency_map_t = torch.tensor(saliency_map/128 - 1).float()
    saliency_map_t = saliency_map_t.unsqueeze(dim = -1).to(device)

    outputs = model(ori_img_t, saliency_map_t).detach().numpy()
    outputs[0] = outputs[0] * ori_img_t.shape[1]
    outputs[1] = outputs[1] * ori_img_t.shape[2]
    outputs[2] = outputs[2] * ori_img_t.shape[1]
    outputs[3] = outputs[3] * ori_img_t.shape[2]
    print(outputs)

    data = open('../flickr-cropping-dataset/cropping_testing_set.json', 'r').read()
    db = json.loads(data)
    image_root = "../flickr-cropping-dataset/data/"
    for i, item in enumerate(db):
        
        img_name = os.path.split(item['url'])[-1]
        if img_name == filename:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            crop = item['crop']
            x = crop[0]
            y = crop[1]
            w = crop[2]
            h = crop[3]




if __name__ == '__main__':
    inspect()