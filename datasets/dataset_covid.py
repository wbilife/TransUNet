import os
import random
from tracemalloc import start
from cv2 import split
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

class Covid_dataset(Dataset):
    def __init__(self, base_dir, start_id, end_id, transform=None):
        # mode = "train" or "test"; [start_id, end_id] 当前训练或测试的数据集范围，如[20, 100]
        self.transform = transform  # using transform in torch!
        self.split = [start_id, end_id + 1]
        all_imgs = np.load(os.path.join(base_dir, 'imgs.npy'))
        all_masks = np.load(os.path.join(base_dir, 'masks.npy'))
        self.imgs_data = all_imgs[start_id : end_id + 1, :, :]
        self.masks_data = all_masks[start_id : end_id + 1, :, :]

    def __len__(self):
        return self.imgs_data.shape[0]

    def __getitem__(self, idx):
        image = self.imgs_data[idx]
        label = self.masks_data[idx].astype(np.uint8)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = "slice_" + str(self.split[0] + idx)
        return sample

def test() :
    pass

if __name__ == "__main__" :
    test()