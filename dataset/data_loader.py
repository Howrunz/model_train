import glob
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from os.path import splitext
from os import listdir
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class DatasetTemplate(Dataset):
    def __init__(self, img_dir, mask_dir, csv_file):
        self.imgs_dir = img_dir
        self.masks_dir = mask_dir

        ##### temp now, to yaml ###
        height = 256
        width = 256
        channel = 3
        ###### ###### ###### ######

        data = pd.read_csv(csv_file, dtype=str)
        train_num = data['Type'][data.Type == 'train'].value_counts().values[0]
        val_num = data['Type'][data.Type == 'validation'].value_counts().values[0]
        # test_num = data['Type'][data.Type == 'test'].value_counts().values[0]
        train_image = np.empty((train_num, height, width, channel))
        train_mask = np.empty((train_num, height, width))
        valid_image = np.empty((val_num, height, width, channel))
        valid_mask = np.empty((val_num, height, width))
        # test_image = np.empty((test_num, height, width, channel))
        # test_mask = np.empty((test_num, height, width))
        for num in range(train_num):
            image_path = self.imgs_dir + data['ID'][num] + '.jpg' # the name of image may different from the mask
            mask_path = self.masks_dir + data['ID'][num] + '_mask.jpg' # the name of mask may different from the image
            image = load_img(image_path, target_size=(height, width), color_mode='rgb') # 'target_size' can resize image
            mask = load_img(mask_path, target_size=(height, width), color_mode='grayscale')
            image_arr = img_to_array(image)
            mask_arr = img_to_array(mask)
            train_image[num] = image_arr
            train_mask[num] = mask_arr
        for num in range(train_num, train_num+val_num):
            image_path = self.imgs_dir + data['ID'][num] + '.jpg'
            mask_path = self.masks_dir + data['ID'][num] + '_mask.jpg'
            image = load_img(image_path, target_size=(height, width), color_mode='rgb')
            mask = load_img(mask_path, target_size=(height, width), color_mode='grayscale')
            image_arr = img_to_array(image)
            mask_arr = img_to_array(mask)
            valid_image[num] = image_arr
            valid_mask[num] = mask_arr
        self.train_image = np.transpose(train_image, (0,3,1,2))
        self.valid_image = np.transpose(valid_image, (0,3,1,2))
        self.train_mask = np.resize(train_mask, (train_num, 1, height, width))
        self.valid_mask = np.resize(valid_mask, (val_num, 1, height, width))

    def __getitem__(self, item):
        image = self.train_image[item]
        mask = self.train_mask[item]
        return torch.from_numpy(image), torch.from_numpy(mask)

    def __len__(self):
        return len(self.f_list)

def get_loader(img_dir, mask_dir, args, shuffle=True):
    dataset = DatasetTemplate(img_dir, mask_dir)
    data_loader = DataLoader(dataset, args.batch_size, shuffle, num_workers=args.workers)
    return data_loader