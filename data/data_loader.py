import glob
from os.path import splitext
from os import listdir
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class DatasetTemplate(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.imgs_dir = img_dir
        self.masks_dir = mask_dir
        self.f_list = [splitext(file)[0] for file in listdir(self.imgs_dir)]

    def __getitem__(self, item):
        index = self.f_list[item]
        image = Image.open(glob.glob(self.imgs_dir + index + '*'))
        mask = Image.open(glob.glob(self.masks_dir + index + '*')) # the name of mask may different from the image
        return torch.from_numpy(image), torch.from_numpy(mask)

    def __len__(self):
        return len(self.f_list)

def get_loader(img_dir, mask_dir, args, shuffle=True):
    dataset = DatasetTemplate(img_dir, mask_dir)
    data_loader = DataLoader(dataset, args.batch_size, shuffle, num_workers=args.workers)
    return data_loader