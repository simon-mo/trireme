import os
import numpy as np
import logging
import cv2

import torch
from torch.utils.data import Dataset

from . import data_transforms


class SatDataset(Dataset):
    def __init__(self, imgs, labels, img_size, is_training, is_debug=False):
        self.imgs = imgs
        self.labels = labels
        self.img_size = img_size  # (w, h)
        self.max_objects = 50
        self.is_debug = is_debug

        #  transforms and augmentation
        self.transforms = data_transforms.Compose()
        if is_training:
            self.transforms.add(data_transforms.ImageBaseAug())
        # self.transforms.add(data_transforms.KeepAspect())
        self.transforms.add(data_transforms.ResizeImage(self.img_size))
        self.transforms.add(data_transforms.ToTensor(self.max_objects, self.is_debug))

    def __getitem__(self, index):
        img = self.imgs[index % len(self.imgs)]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index % len(self.labels)]
        sample = {'image': img, 'label': label}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.imgs)