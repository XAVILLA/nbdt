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

    def __init__(self, split='Art', train=True, transform=None, root = None, download = None):
        assert split in splits
        data_root = '/rscratch/xyyue/data/officehome/'
        data_root = '/Users/zixianzang/Downloads/OfficeHomeDataset_10072016/'
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
        # print(self.imgs)
        # def is_valid(path):
        #     # print(path)
        #     result = [path in im.replace('zangwei/datasets', 'data') for im in self.imgs]
        #     return any(result)
        #
        # self.dataset = datasets.ImageFolder('/rscratch/xyyue/data/officehome/' + split,
        #                                     transform=transform,
        #                                     is_valid_file=is_valid)


        def is_valid(path):
            # print(path)
            result = [path in im.replace('/rscratch/xyyue/zangwei/datasets/officehome/', '/Users/zixianzang/Downloads/OfficeHomeDataset_10072016/') for im in self.imgs]
            return any(result)

        self.dataset = datasets.ImageFolder('/Users/zixianzang/Downloads/OfficeHomeDataset_10072016/' + split,
                                            transform=transform,
                                            is_valid_file=is_valid)

        self.classes = self.dataset.classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    @staticmethod
    def transform_train():
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(200),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5072319249078396, 0.4708995484264786, 0.43519951206887564], [0.3277489440473064, 0.32484368518264295, 0.32752388590993836]
                ),
            ]
        )

    @staticmethod
    def transform_val():
        return transforms.Compose(
            [
                transforms.Scale((200)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5072319249078396, 0.4708995484264786, 0.43519951206887564], [0.3277489440473064, 0.32484368518264295, 0.32752388590993836]
                ),
            ]
        )

    @staticmethod
    def transform_val_inverse():
        return transforms_custom.InverseNormalize(
            [0.5072319249078396, 0.4708995484264786, 0.43519951206887564], [0.3277489440473064, 0.32484368518264295, 0.32752388590993836]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
