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
    def __init__(self, base_dir, test_start_id, test_end_id, mode = "train", transform=None):
        all_imgs = np.load(os.path.join(base_dir, 'imgs.npy'))
        all_masks = np.load(os.path.join(base_dir, 'masks.npy'))
        self.transform = transform  # using transform in torch!
        if mode == "train" :
            imgs_data1 = all_imgs[0 : test_start_id, :, :]
            imgs_data2 = all_imgs[test_end_id : all_imgs.shape[0] + 1, :, :]
            self.imgs_data = np.vstack([imgs_data1, imgs_data2])
            masks_data1 = all_masks[0 : test_start_id, :, :]
            masks_data2 = all_masks[test_end_id : all_imgs.shape[0] + 1, :, :]
            self.masks_data = np.vstack([masks_data1, masks_data2])
        else :
            self.imgs_data = all_imgs[test_start_id : test_end_id, :, :]
            self.masks_data = all_masks[test_start_id : test_end_id, :, :]

    def __len__(self):
        return self.imgs_data.shape[0]

    def __getitem__(self, idx):
        image = self.imgs_data[idx]
        label = self.masks_data[idx].astype(np.uint8)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = "slice_" + str(idx)
        return sample

def test() :
    pass

if __name__ == "__main__" :
    test()