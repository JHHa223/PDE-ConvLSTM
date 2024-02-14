from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import sys
import torch
from PIL import Image, ImageFilter
import torchvision
from torchvision import transforms
import glob

def max_pooling(arr, pool_size=(2, 2)):
    batch_size, height, width, channels = arr.shape
    h_pool, w_pool = pool_size

    # Apply max pooling
    pooled_arr = np.zeros((batch_size, height // h_pool, width // w_pool, channels))

    for i in range(height // h_pool):
        for j in range(width // w_pool):
            # Find the maximum value in each pooling region
            pooled_arr[:, i, j, :] = np.max(arr[:, i*h_pool:(i+1)*h_pool, j*w_pool:(j+1)*w_pool, :], axis=(1, 2))

    return pooled_arr

def normalize(x):
    torch_x_norm = torch.zeros_like(x)
    cls_idx = 1.
    cls_thres=[0.1,1,5,10]
    for ct in cls_thres:
        torch_x_norm[x>=ct] = cls_idx
        cls_idx += 1.
    torch_x_norm = torch_x_norm/4.
    return torch_x_norm

class Loader(Dataset):
    """
    is_train == 0 training dataset
    is_train == 1 validation dataset
    """
    def __init__(self, is_train=0, path='', transform=None):
        self.is_train = is_train
        if is_train == 0:
            self.path = glob.glob('/data0/jhha223/conv_LSTM_PDE_tuning/training/*.npy')
        else:
            self.path = glob.glob('/data0/jhha223/conv_LSTM_PDE_tuning/test/*.npy')

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        #image loading
        img = np.load(self.path[idx])
        #max pooling
        pooled_img = max_pooling(img)
        torch_img = torch.from_numpy(pooled_img)
        #preprocessing
        torch_img_norm = normalize(torch_img)
        return torch_img_norm

if __name__ == '__main__':
    loader = Loader()

