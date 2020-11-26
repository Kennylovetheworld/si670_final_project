from model import *
from dataloader import *
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os
import cv2
import json
import argparse
from random import shuffle
import warnings
import numpy as np



use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100

def build_model():
    model = CropProposalModel().to(device)
    return model

def train():
    image_root = "../flickr-cropping-dataset/data/"
    saliency_root = "../flickr-cropping-dataset/saliency_map/"

    # Generators
    training_set = Dataset('../flickr-cropping-dataset/cropping_training_set.json', image_root, saliency_root)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = Dataset('../flickr-cropping-dataset/cropping_testing_set.json', image_root, saliency_root)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    # build the model and set the hyperparameter
    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001, momentum=0.9)

    best_loss = 100

    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        for i, (ori_img, saliency_map, x, y, w, h) in enumerate(training_generator):
            # Transfer to GPU
            ori_img, saliency_map, x, y, w, h = ori_img.to(device), saliency_map.to(device), x.to(device), y.to(device), w.to(device), h.to(device)

            # Model computations
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(ori_img_t, saliency_map_t)
            loss = IOU_Loss(outputs[:,0], outputs[:,1], outputs[:,2], outputs[:,3], x21, y21, w2, h2)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

        # Validation
        with torch.set_grad_enabled(False):
            iterations = 0
            for i, (ori_img, saliency_map, x, y, w, h) in enumerate(validation_generator):
                # Transfer to GPU
                ori_img, saliency_map, x, y, w, h = ori_img.to(device), saliency_map.to(device), x.to(device), y.to(device), w.to(device), h.to(device)

                # Model computations
                outputs = model(ori_img_t, saliency_map_t)
                loss = criterion(outputs, (x, y, w, h))
                # print statistics
                running_loss += loss.item()
                iterations += 1
            
            cur_loss = running_loss / iterations

            # remember best acc@1 and save checkpoint
            is_best = cur_loss < best_loss
            best_acc1 = min(cur_loss, best_loss)

            if is_best:
                torch.save(model.state_dict(), 'saved_model/best_model.pt')
                print('Best Model Updated')
            
            print('Validation [%d] loss: %.3f' % (epoch + 1, cur_loss))
            running_loss = 0.0

        torch.save(best_model.state_dict(), 'model.pt')        

    print('Finished Training')

if __name__ == '__main__':
    train()