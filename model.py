import math
from collections import OrderedDict
import torch.nn as nn
import torch.nn.init as init
import torch
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

def L2_Loss(x11, y11, x12, y12, x21, y21, w2, h2):
    x22 = x21 + w2
    y22 = y21 + h2
    loss = (x11-x21)**2 + (y11-y21)**2 + (x12-x22)**2 + (y12-y22)**2
    return loss[0]

def IOU_Loss(x11, y11, x12, y12, x21, y21, w2, h2):
    w1 = (x12 - x11).clamp(min = 0)
    h1 = (y12 - y11).clamp(min = 0)
    areas1 = w1 * h1
    areas2 = w2 * h2
    x22 = x21 + w2
    y22 = y21 + h2
    lt_x, lt_y = torch.max(x11, x21), torch.max(y11, y21) 
    rb_x, rb_y = torch.min(x12, x22), torch.min(y12, y22)
    I_w = (rb_x - lt_x).clamp(min=0)
    I_h = (rb_y - lt_y).clamp(min=0)
    I = I_w * I_h
    IOU =  I / (areas1 + areas2 - I)
    loss = 1 - IOU.mean()
    return loss

class SPPLayer(nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        # bs: the number of samples
        # c: the number of channels
        # h: height
        # w: width
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = (h // (2 ** i), w // (2 ** i))
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).reshape(bs, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).reshape(bs, -1)
            pooling_layers.append(tensor)
        # pdb.set_trace()
        x = torch.cat(pooling_layers, dim=-1)
        return x

class DetectionNetSPP(nn.Module):
    """
    Expected input size is 64x64
    """

    def __init__(self, spp_level=3, in_channels = 3, name = '', resnet = False):
        super(DetectionNetSPP, self).__init__()
        self.spp_level = spp_level
        self.in_channels = in_channels
        self.num_grids = 0
        for i in range(spp_level):
            self.num_grids += 2**(i*2)

        if resnet == True:
            self.conv_model = models.resnet18(pretrained=True)
            self.conv_model = nn.Sequential(*list(self.conv_model.children())[:-2]) # Remove the last classifier and the avg pooling
        else: 
            self.conv_model = nn.Sequential(OrderedDict([
            ('conv1' + name, nn.Conv2d(self.in_channels, 128, 3)), 
            ('relu1' + name, nn.ReLU()),
            ('pool1' + name, nn.MaxPool2d(2)),
            ('conv2' + name, nn.Conv2d(128, 128, 3)),
            ('relu2' + name, nn.ReLU()),
            ('pool2' + name, nn.MaxPool2d(2)),
            ('conv3' + name, nn.Conv2d(128, 128, 3)), 
            ('relu3' + name, nn.ReLU()),
            ('pool3' + name, nn.MaxPool2d(2)),
            ('conv4' + name, nn.Conv2d(128, 128, 3)),
            ('relu4' + name, nn.ReLU())
            ]))

        self.spp_layer = SPPLayer(spp_level)

    def forward(self, x):
        # pdb.set_trace()
        x = self.conv_model(x)
        x = self.spp_layer(x)
        return x

class CropProposalModel(nn.Module):
    def __init__(self, spp_level = 3):
        super(CropProposalModel, self).__init__()
        self.saliency_net = DetectionNetSPP(spp_level = 3, in_channels = 1, name = '_saliency')
        self.ori_img_net = DetectionNetSPP(spp_level = 2, in_channels = 3, name = '_ori', resnet = True)
        # 128 is the number of out_channel of the last convolutional layer of our self defined conv_model
        # 512 is the number of out_channel of the last convolutional layer of resnet18
        self.in_features = self.saliency_net.num_grids * 128 + self.ori_img_net.num_grids * 512
        self.out_features = 4
        self.linear_model = nn.Sequential(OrderedDict([
          ('fc1_crop', nn.Linear(self.in_features, 1024)),
          ('fc1_relu_crop', nn.ReLU()),
          ('dropout', nn.Dropout(p=0.4)),
          ('fc2_crop', nn.Linear(1024, self.out_features)),
        ]))
    
    def forward(self, ori_img, saliency_map):
        # pdb.set_trace()
        saliency_map = saliency_map.permute(0, 3, 1, 2)
        ori_img = ori_img.permute(0, 3, 1, 2)
        # ori_img_w = ori_img.shape[-2]
        # ori_img_w = ori_img.shape[-1]
        x1 = self.ori_img_net(ori_img)
        x2 = self.saliency_net(saliency_map)
        x = torch.cat([x1, x2], dim = -1)
        x = self.linear_model(x)
        x = torch.sigmoid(x)
        return x