import argparse
from utils import tools
from dataset.data_loader import get_test_loader
import cv2
import os

import torch
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', type=str, default='UNet', choices=['UNet'])
    arg('--image-path', type=str, default='./data/test/')
    arg('--output-path', type=str, default='./output/')
    arg('--config-file', type=str, default='./config/train_config.yaml')
    args = parser.parse_args()

    cfg = tools.read_yaml(args.config_file)
    csv_dir = cfg['DIRECTORY']['csv_dir']

    network = args.model
    model_weight_path = 'checkpoint/model_' + args.model + '.pt'

    data_loader = get_test_loader(cfg, args.image_path, csv_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = nn.DataParallel(network)
    network.to(device)
    weight = torch.load(model_weight_path)
    network.load_state_dict(weight['model'])

    with torch.no_grad():
        network.eval()
        for test_image, test_name in data_loader:
            test_image = test_image.to(device)
            outputs = network(test_image)
            test_pred = outputs.squeeze().data.cpu().numpy()
            test_mask = (test_pred > cfg['MODEL']['Threshold']).astype('int') * 255
            cv2.imwrite(os.path.join(args.output_path, '%s.png' % test_name), test_mask)

if __name__ == '__main__':
    main()