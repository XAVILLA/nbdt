import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from . import transforms as transforms_custom
from torch.utils.data import Dataset
from pathlib import Path
import zipfile
import urllib.request
import shutil
import time

import os
import json
from PIL import Image

import cv2
import numpy as np
import random

import torch
from torch.nn import functional as F
from torch.utils import data

__all__ = names = ('Office_Home',)

splits = ['Art', 'Clipart', 'Product', 'Real']


class Office_Home(data.Dataset):

    def __init__(self, split='Art', train=True):
        assert split in splits
        data_root = '/rscratch/xyyue/data/officehome/'
        norm_file = split + '-info.json'
        if train:
            imfo_file = split + '_train.txt'
        else:
            imfo_file = split + '_val.txt'
        img_infos = os.path.join(data_root, 'meta', imfo_file)

        normalize_info = os.path.join(data_root, norm_file)

        with open(normalize_info, 'r') as f:
            norm = json.load(f)
            self.mean = norm['mean']
            self.std = norm['std']

        self.imgs = []
        with open(img_infos, 'r') as f:
            self.imgs = f.read().splitlines()

        def is_valid(path):
            return any([path in im for im in self.imgs])

        self.dataset = datasets.ImageFolder('/rscratch/xyyue/data/officehome/' + split,
                                            is_valid_file = is_valid)

        self.classes = self.dataset.classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    @staticmethod
    def transform_train():
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    self.mean, self.std
                ),
            ]
        )

    @staticmethod
    def transform_val():
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    self.mean, self.std
                ),
            ]
        )

    @staticmethod
    def transform_val_inverse():
        return transforms_custom.InverseNormalize(
            self.mean, self.std
        )


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return len(self.dataset)


