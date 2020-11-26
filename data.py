"""
load the image data from the flickr-cropping-dataset
"""
import os
import cv2
import json
import argparse
from random import shuffle
import warnings


if __name__ == '__main__':
    data = open('../flickr-cropping-dataset/cropping_testing_set.json', 'r').read()
    db = json.loads(data)

    image_root = "../flickr-cropping-dataset/data/"
    for i, item in enumerate(db):
        
        filename = os.path.split(item['url'])[-1]
        path = image_root + filename 
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        crop = item['crop']
        x = crop[0]
        y = crop[1]
        w = crop[2]
        h = crop[3]

        # Error handling
        if img is None:
            warnings.warn('Warnings: loading image failed!')
            continue
        
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow('Cropping', img)
        key = cv2.waitKey(0)

        if i == 10:
            break