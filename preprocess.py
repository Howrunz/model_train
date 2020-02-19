from utils import tools
import os
import random
import pandas as pd

def main():
    cfg = tools.read_yaml('./config/train_config.yaml')
    file_name = os.listdir(cfg['DIRECTORY']['image_dir'])
    random.seed(42)
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
    train_df.to_csv('./data/train_list.csv', index=False)
    valid_df.to_csv('./data/valid_list.csv', index=False)
    test_df.to_csv('./data/test_list.csv', index=False)
    print('Create CSV file successfully')

if __name__ == '__main__':
    main()