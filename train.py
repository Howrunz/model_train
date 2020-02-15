import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import cv2 as cv
import time
from validation import validation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.UNet import UNet
from loss.Dice_loss import Diceloss
from dataset.data_loader import get_loader
from utils import tools

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', type=str, default='UNet', choices=['UNet'])
    arg('--config-file', type=str, default='./config/train_config.yaml')
    arg('--checkpoint', type=str, default='checkpoint/')
    arg('--image-dir', type=str, default='./data/image')
    arg('--mask-dir', type=str, default='./data/mask')
    args = parser.parse_args()

    # get config
    cfg = tools.read_yaml(args.config_file)

    # folder for checkpoint
    checkpoint = Path(args.checkpoint)
    # create directory which not exist in every part of path while parents is True
    # ignore 'FileExistsError' while exist_ok is True
    checkpoint.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint / 'model_'+args.model+'.pt'

    # print initial parameters
    print('-★-' * 10)
    print(args)
    print('-★-' * 10)

    # load model
    if args.model == 'Unet':
        network = UNet(in_channels=3, out_channels=1)
    else:
        network = UNet(in_channels=3, out_channels=1)

    # multiple GPUs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = nn.DataParallel(network)
    network.to(device)

    # model summary
    tools.print_model_summary(network)

    # define loss
    dice_loss = Diceloss()
    criterion = dice_loss
    train_loss = 0.
    best_validation_loss = 0.

    # define dataloader
    csv_dir = cfg['DIRECTORY']['csv_dir']
    train_loader = get_loader(cfg, args.image_dir, args.mask_dir, csv_dir)
    valid_loader = get_loader(cfg, args.image_dir, args.mask_dir, csv_dir, status='validation')

    # save train image
    if True:
        print('-★-' * 10)
        print('check train image and mask')
        train_img, train_mask = next(iter(train_loader))
        train_img[:, :, 0] = train_img[:, :, 0] * 0.7 + train_mask * 0.3
        cv.imwrite('./result/random_train_image.jpg', train_img)
        print('save train image successful!')
        print('-★-' * 10)

    # optimizer
    if cfg['Model']['Optimizer'] == 'Adam':
        optimizer = Adam(network.parameters(), lr=args.lr)
    else:
        optimizer = SGD(network.parameters(), lr=args.lr)

    # change learning rate
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8)

    # start training
    print('start training')
    step = 0
    for epoch in range(1, cfg['Model']['Epochs']):
        start_time = time.time()
        network.train()
        try:
            for i, (image, mask) in enumerate(train_loader):
                image = image.to(device)
                mask = mask.to(device)
                output = network(image)
                train_loss = criterion(output, mask)
                print(f'epoch = {epoch: 3d}, iter = {i: 3d}, loss = {train_loss.item(): .4g}')
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                step += 1
            valid_loss = validation(network, criterion, valid_loader, device)
            print('valid_loss:', valid_loss)
            if valid_loss < best_validation_loss:
                tools.save_weight(network, model_path, train_loss, valid_loss, epoch, step)
                best_validation_loss = valid_loss
                print('Save best model by validation loss')
            scheduler.step(valid_loss)
        except:
            pass

if __name__ == '__main__':
    main()