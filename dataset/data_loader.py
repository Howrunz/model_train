import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from torch.utils.data import Dataset, DataLoader
import torch

class TrainDataset(Dataset):
    def __init__(self, cfg, img_dir, mask_dir, csv_dir, status='train'):
        self.config = cfg
        self.imgs_dir = img_dir
        self.masks_dir = mask_dir
        self.status = status
        assert self.status in ['train', 'validation'], 'Please input status as train or validation'
        height = cfg.Height
        width = cfg.Width
        channel = cfg.Channel

        self.csv_file = csv_dir + self.status + '_list.csv'
        data = pd.read_csv(self.csv_file, dtype=str)
        self.data_num = data['Type'][data.Type == self.status].value_counts().values[0]
        image_set = np.empty((self.data_num, height, width, channel))
        mask_set = np.empty((self.data_num, height, width))
        for num in range(self.data_num):
            image_path = self.imgs_dir + data['ID'][num] + '.jpg' # the name of image may different from the mask
            mask_path = self.masks_dir + data['ID'][num] + '_mask.jpg' # the name of mask may different from the image
            image = load_img(image_path, target_size=(height, width), color_mode='rgb') # 'target_size' can resize image
            mask = load_img(mask_path, target_size=(height, width), color_mode='grayscale')
            image_arr = img_to_array(image)
            mask_arr = img_to_array(mask)
            image_set[num] = image_arr
            mask_set[num] = mask_arr
        self.image = np.transpose(image_set, (0,3,1,2))
        self.mask = np.resize(mask_set, (self.data_num, 1, height, width))

    def __getitem__(self, item):
        image = self.image[item]
        mask = self.mask[item]
        return torch.from_numpy(image), torch.from_numpy(mask)

    def __len__(self):
        return self.data_num

class TestDataset(Dataset):
    def __init__(self, cfg, image_dir, csv_file):
        self.image_dir = image_dir
        self.csv = csv_file
        height = cfg.Height
        width = cfg.Width
        channel = cfg.Channel
        self.data = pd.read_csv(self.csv, dtype=str)
        self.data_num = self.data['Type'][self.data.Type == 'test'].value_counts().values[0]
        image_set = np.empty((self.data_num, height, width, channel))
        for num in range(self.data_num):
            image_path = self.image_dir + self.data['ID'][num] + '.jpg'
            image = load_img(image_path, target_size=(height, width), color_mode='rgb')
            image_arr = img_to_array(image)
            image_set[num] = image_arr
        self.image = np.transpose(image_set, (0,3,1,2))

    def __getitem__(self, item):
        image = self.image[item]
        return image, self.data['ID'][item]

    def __len__(self):
        return self.data_num

def get_loader(cfg, img_dir, mask_dir, csv_dir, shuffle=True, status='train'):
    dataset = TrainDataset(cfg, img_dir, mask_dir, csv_dir, status=status)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.Batch_size,
        shuffle=shuffle,
        num_workers=cfg.Num_workers
    )
    return data_loader

def get_test_loader(cfg, img_dir, csv_file, shuffle=False):
    dataset = TestDataset(cfg, img_dir, csv_file)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.Batch_size,
        shuffle=shuffle,
        num_workers=cfg.Num_workers
    )
    return data_loader