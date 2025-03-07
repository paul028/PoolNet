import os
from PIL import Image
import cv2
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import numpy as np
import random

class ImageDataTrain(data.Dataset):
    def __init__(self, sal_data_root, sal_data_list, edge_data_root, edge_data_list):
        self.sal_root = sal_data_root
        self.sal_source = sal_data_list
        self.edge_root = edge_data_root
        self.edge_source = edge_data_list

        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]
        with open(self.edge_source, 'r') as f:
            self.edge_list = [x.strip() for x in f.readlines()]

        self.sal_num = len(self.sal_list)
        self.edge_num = len(self.edge_list)


    def __getitem__(self, item):
        # edge data loading
        edge_im_name = self.edge_list[item % self.edge_num].split()[0]
        edge_gt_name = self.edge_list[item % self.edge_num].split()[1]
        edge_image = load_image(os.path.join(self.edge_root, edge_im_name))
        edge_label = load_edge_label(os.path.join(self.edge_root, edge_gt_name))
        edge_image = torch.Tensor(edge_image)
        edge_label = torch.Tensor(edge_label)

        # sal data loading
        sal_im_name = self.sal_list[item % self.sal_num].split()[0]
        sal_gt_name = self.sal_list[item % self.sal_num].split()[1]
        sal_image = load_image(os.path.join(self.sal_root, sal_im_name))
        sal_label = load_sal_label(os.path.join(self.sal_root, sal_gt_name))
        sal_image, sal_label = cv_random_flip(sal_image, sal_label)
        sal_image = torch.Tensor(sal_image)
        sal_label = torch.Tensor(sal_label)

        sample = {'edge_image': edge_image, 'edge_label': edge_label, 'sal_image': sal_image, 'sal_label': sal_label}
        return sample

    def __len__(self):
        return max(self.sal_num, self.edge_num)

class ImageDataTest(data.Dataset):
    def __init__(self, data_root, data_list):
        self.data_root = data_root
        self.data_list = data_list
        with open(self.data_list, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, im_size = load_image_test(os.path.join(self.data_root, self.image_list[item]))
        image = torch.Tensor(image)

        return {'image': image, 'name': self.image_list[item % self.image_num], 'size': im_size}

    def __len__(self):
        return self.image_num


def get_loader(config, mode='train', pin=False):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(config.train_root, config.train_list, config.train_edge_root, config.train_edge_list)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread)
    else:
        dataset = ImageDataTest(config.test_root, config.test_list)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread, pin_memory=pin)
    return data_loader

def load_image(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2,0,1))
    return in_

def load_image_test(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2,0,1))
    return in_, im_size

def load_sal_label(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    label = label / 255.
    label = label[np.newaxis, ...]
    return label

def load_edge_label(path):
    """
    pixels > 0.5 -> 1.
    """
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    label = label / 255.
    label[np.where(label > 0.5)] = 1.
    label = label[np.newaxis, ...]
    return label

def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img[:,:,::-1].copy()
        label = label[:,:,::-1].copy()
    return img, label
