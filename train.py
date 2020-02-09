import pandas as pd
import numpy as np
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

from model.UNet import UNet
from loss.Dice_loss import Diceloss
from data.data_loader import get_loader
from utils import tools

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--batch-size', type=int, default=8, help='batch size')
    arg('--epochs', type=int, default=100, help='train epochs')
    arg('--optimizer', type=str, default='Adam', help='Adam or SGD')
    arg('--lr', type=float, default=0.0001, help='learning rate')
    arg('--workers', type=int, default=4, help='num workers')
    arg('--model', type=str, default='UNet', choices=['UNet'])
    arg('--checkpoint', type=str, default='checkpoint/UNet')
    args = parser.parse_args()

    # folder for checkpoint
    checkpoint = Path(args.checkpoint)
    # create directory which not exist in every part of path while parents is True
    # ignore 'FileExistsError' while exist_ok is True
    checkpoint.mkdir(parents=True, exist_ok=True)

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
    loss = Diceloss()