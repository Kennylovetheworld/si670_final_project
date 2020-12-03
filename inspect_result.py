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
from PIL import Image, ImageDraw


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 6}

def inspect(filename = "23432512185_b02fd2fa29_c.jpg"):
    # load validation set
    image_root = "../flickr-cropping-dataset/data/"
    saliency_root = "../flickr-cropping-dataset/saliency_map/"
    img_path = image_root + filename
    saliency_path = saliency_root + filename
    

    model = CropProposalModel().to(device)
    model.load_state_dict(torch.load('../si670_final_project/saved_model/best_model.pt'))
    print('Model loaded')
    model.eval()
    ori_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    saliency_map = cv2.imread(saliency_path, 0)
    ori_img_t = torch.tensor(ori_img/128 - 1).float().to(device)
    saliency_map_t = torch.tensor(saliency_map/128 - 1).float()
    saliency_map_t = saliency_map_t.unsqueeze(dim = -1).to(device)

    outputs = model(ori_img_t.unsqueeze(dim = 0), saliency_map_t.unsqueeze(dim = 0)).cpu().detach().numpy()[0]
    print(outputs)
    outputs[0] = outputs[0] * ori_img_t.shape[1]
    outputs[1] = outputs[1] * ori_img_t.shape[0]
    outputs[2] = outputs[2] * ori_img_t.shape[1]
    outputs[3] = outputs[3] * ori_img_t.shape[0]

    data = open('../flickr-cropping-dataset/cropping_training_set.json', 'r').read()
    db = json.loads(data)
    image_root = "../flickr-cropping-dataset/data/"
    for i, item in enumerate(db):
        img_name = os.path.split(item['url'])[-1]
        if img_name == filename:
            print('img found')
            crop = item['crop']
            x = crop[0]
            y = crop[1]
            w = crop[2]
            h = crop[3]
            start_point = (x, y)
            end_point = (x + w, y + h)
            color = (0, 0, 255)
            thickness = 10
            ori_img = cv2.rectangle(ori_img, start_point, end_point, color, thickness) 

    start_point = (outputs[0], outputs[1])
    end_point = (outputs[2], outputs[3])
    color = (0, 255, 0)
    thickness = 10
    ori_img = cv2.rectangle(ori_img, start_point, end_point, color, thickness) 
    cv2.imwrite(filename, ori_img)


if __name__ == '__main__':
    inspect()