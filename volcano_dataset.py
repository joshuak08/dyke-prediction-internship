import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
from torchvision import transforms
from scipy import ndimage
from PIL import Image

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class Volcano_dataset(Dataset):
    def __init__(self, data_dir, learning_param, partition, transform=None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        with open(self.data_dir + 'file_split.json', 'r') as f:
            if partition == 'train' or partition == "val":
                self.images = json.load(f)['train']
            else:
                self.images = json.load(f)['test']

        self.param_map = {"strike": 1, "opening": 5, "top": 7, "bottom": 9, "length": 11}
        if isinstance(learning_param, list):
            self.param = learning_param
        elif isinstance(learning_param, str):
            self.param = [learning_param]

    def __getitem__(self, index):
        file = self.images[index]
        # 3 channels for now for gray image, if input is coloured needs to be converted to grayscale first
        image = np.float64(cv2.imread(file))

        # replace nan with mean values
        image = np.nan_to_num(image, nan=np.mean(image))

        image = cv2.resize(image, (512, 512))

        image = np.divide(image, 255.0)
        filename = os.path.basename(file).split('/')[-1]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.transform:
            image = self.transform(image)
        image = image.to(device=device, dtype=torch.float)
        sample= {'image': image}
        for p in self.param:
            sample[p] = torch.tensor([float(filename.split("_")[self.param_map[p]])]).to(device=device, dtype=torch.float)

        return sample
    
    def __len__(self):
        return len(self.images)
    
class RandomCrop(object):
    def __init__(self, output_size, topleft=False):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.topleft = topleft
    def __call__(self, sample):
        image = sample
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        if (h > new_h) and (not self.topleft):
            top = np.random.randint(0, h - new_h)
        else:
            top = 0
        if (w > new_w) and (not self.topleft):
            left = np.random.randint(0, w - new_w)
        else:
            left = 0
        image = image[top: top + new_h,
                      left: left + new_w]
        return image
    
class Resize(object):
    def __init__(self, output_size):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, tuple):
            self.output_size = output_size
        elif isinstance(output_size, list):
            self.output_size = tuple(output_size)
    def __call__(self, sample):
        image = sample
        image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_NEAREST)
        return image

# might only be good for strike, might not pick up small intricacies of other parameters
class CannyEdgeDetection(object):
    def __call__(self, sample):
        image = sample
        canny = cv2.Canny(image, 100, 200)
        return canny
    