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

__all__ = names = ('Office_Home_Art', 'Office_Home_Clipart', 'Office_Home_Product', 'Office_Home_Real', )

splits = ['Art', 'Clipart', 'Product', 'Real']

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class Office_Home(data.Dataset):

    def __init__(self, split='Art', train=True, transform=None, root = None, download = None):
        local = False

        assert split in splits
        data_root = '/rscratch/xyyue/data/officehome/'
        if local:
            data_root = '/Users/zixianzang/Downloads/OfficeHomeDataset_10072016/'
        if train:
            imfo_file = split + '_train.txt'
        else:
            imfo_file = split + '_val.txt'
        img_infos = os.path.join(data_root, 'meta', imfo_file)

        self.imgs = []
        with open(img_infos, 'r') as f:
            self.imgs = f.read().splitlines()
        # print(self.imgs)


        if not local:
            def is_valid(path):
                # print(path)
                result = [path in im.replace('zangwei/datasets', 'data') for im in self.imgs]
                return any(result)

            self.dataset = datasets.ImageFolder('/rscratch/xyyue/data/officehome/' + split,
                                                transform=transform,
                                                is_valid_file=is_valid)

        else:
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
        # return transforms.Compose(
        #     [
        #         transforms.RandomResizedCrop(128,
        #                                      # scale=(0.3, 1.0)
        #                                      ),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             [0.5072319249078396, 0.4708995484264786, 0.43519951206887564], [0.3277489440473064, 0.32484368518264295, 0.32752388590993836]
        #         ),
        #     ]
        # )

        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    @staticmethod
    def transform_val():
        # return transforms.Compose(
        #     [
        #         transforms.Scale((128, 128)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             [0.5072319249078396, 0.4708995484264786, 0.43519951206887564], [0.3277489440473064, 0.32484368518264295, 0.32752388590993836]
        #         ),
        #     ]
        # )
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    @staticmethod
    def transform_val_inverse():
        return transforms_custom.InverseNormalize(
            mean, std
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]



class Office_Home_Art(Office_Home):
    def __init__(self):
        super().__init__(split='Art', train=True, transform=None, root = None, download = None)

class Office_Home_Clipart(Office_Home):
    def __init__(self):
        super().__init__(split='Clipart', train=True, transform=None, root = None, download = None)

class Office_Home_Product(Office_Home):
    def __init__(self):
        super().__init__(split='Product', train=True, transform=None, root = None, download = None)

class Office_Home_Real(Office_Home):
    def __init__(self):
        super().__init__(split='Real', train=True, transform=None, root = None, download = None)