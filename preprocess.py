from utils import config
import os
import random
import pandas as pd
import argparse

def main(args):
    opj = os.path.join
    cfg = config.Config(args.conf)
    file_name = os.listdir(cfg.image_dir)
    random.seed(cfg.Random_seed)
    files = []
    for file in file_name:
        files.append(file[0:12])  # different dataset use different range
    random.shuffle(files)

    all_data_num = len(files)
    train_data_num = all_data_num // 6
    valid_data_num = all_data_num // 2

    print('Start to create CSV file')
    train_df = pd.DataFrame({'ID': files[0:train_data_num], 'Type': 'train'})
    valid_df = pd.DataFrame({'ID': files[train_data_num:(train_data_num+valid_data_num)], 'Type': 'valid'})
    test_df = pd.DataFrame({'ID': files[(train_data_num+valid_data_num):all_data_num], 'Type': 'valid'})
    train_csv_path = opj(cfg.csv_dir, args.csvn + '_train.csv')
    valid_csv_path = opj(cfg.csv_dir, args.csvn + '_valid.csv')
    test_csv_path = opj(cfg.csv_dir, args.csvn + '_test.csv')
    train_df.to_csv(train_csv_path, index=False)
    valid_df.to_csv(valid_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    if os.path.exists(train_csv_path) and os.path.exists(valid_csv_path) and os.path.exists(test_csv_path):
        print('Create CSV file successfully!')

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    arg = parse.add_argument
    arg('--conf', type=str, default='./config/train_config.yaml', help='config file path')
    arg('--csvn', type=str, default='dataset', help='csv file name')
    args = parse.parse_args()
    main(args)