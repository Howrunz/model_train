import os
import argparse
from pathlib import Path
import cv2 as cv
import time
from validation import validation
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.UNet import UNet
from loss.Dice_loss import Diceloss
from dataset.data_loader import get_loader
from utils import tools, config

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', type=str, default='UNet', choices=['UNet'])
    arg('--config-file', type=str, default='./config/train_config.yaml')
    arg('--checkpoint', type=str, default='checkpoint/')
    args = parser.parse_args()

    # get config
    cfg = config.Config(args.config_file)

    # init writer
    writer = SummaryWriter(log_dir='log_result')

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
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in cfg.GPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    network = nn.DataParallel(network)
    network.to(device)

    # model summary
    tools.print_model_summary(network)

    # define loss
    dice_loss = Diceloss()
    criterion = dice_loss
    best_validation_loss = 0.

    # define dataloader
    csv_dir = cfg.csv_dir
    image_dir = cfg.image_dir
    mask_dir = cfg.mask_dir
    train_loader = get_loader(cfg, image_dir, mask_dir, csv_dir)
    valid_loader = get_loader(cfg, image_dir, mask_dir, csv_dir, status='validation')

    # save train image
    if True:
        print('-★-' * 10)
        print('check train image and mask')
        train_img, train_mask = next(iter(train_loader))
        print('train image shape: {}'.format(train_img.shape))
        train_img[:, :, 0] = train_img[:, :, 0] * 0.7 + train_mask * 0.3
        cv.imwrite('./result/random_train_image.jpg', train_img)
        print('save train image successful!')
        print('-★-' * 10)

    # optimizer
    if cfg.Optimizer == 'Adam':
        optimizer = Adam(network.parameters(), lr=args.lr)
    else:
        optimizer = SGD(network.parameters(), lr=args.lr)

    # change learning rate
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8)

    # start training
    print('start training')
    step = 0
    for epoch in range(1, cfg.Epochs):
        start_time = time.time()
        train_loss = 0.
        network.train()
        for i, (image, mask) in enumerate(train_loader):
            image = image.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            torch.set_grad_enabled(True)
            output = network(image)
            loss = criterion(output, mask)
            train_loss += loss.item()
            writer.add_scalar('evaluation/train_loss', loss, epoch)
            print(f'epoch = {epoch: 3d}, iter = {i: 3d}, loss = {train_loss: .4g}')
            loss.backward()
            optimizer.step()
            step += 1
        valid_loss = validation(network, criterion, valid_loader, device, writer, epoch)
        writer.add_scalar('evaluation/valid_loss', valid_loss, epoch)
        if valid_loss < best_validation_loss:
            tools.save_weight(network, model_path, train_loss, valid_loss, epoch, step)
            best_validation_loss = valid_loss
            print('Save best model by validation loss')
        scheduler.step(valid_loss)

if __name__ == '__main__':
    main()