from model import *
from dataloader import *
import torch


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 6}

def test():
    # load validation set
    image_root = "./flickr-cropping-dataset/data/"
    saliency_root = "./flickr-cropping-dataset/saliency_map/"
    validation_set = Dataset('./flickr-cropping-dataset/cropping_testing_set.json', image_root, saliency_root)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    model = CropProposalModel().to(device)
    model.load_state_dict(torch.load('./si670_final_project/saved_model/best_model.pt'))
    print('Model loaded')
    model.eval()

    with torch.set_grad_enabled(False):
        total_iou, numberOfSamples = 0, 0
        for i, (ori_img, saliency_map, x, y, w, h) in enumerate(validation_generator):
            print(i)
            ori_img, saliency_map, x, y, w, h = ori_img.to(device), saliency_map.to(device), x.to(device), y.to(device), w.to(device), h.to(device)
            # Model computations
            outputs = model(ori_img, saliency_map)
            iou = 1 - IOU_Loss(outputs[:,0], outputs[:,1], outputs[:,2], outputs[:,3], x, y, w, h)
            total_iou += iou
            numberOfSamples += 1
        print('Avg iou is: ', total_iou/numberOfSamples)


if __name__ == '__main__':
    test()
