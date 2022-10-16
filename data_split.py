
import glob
import os


import numpy as np
import cv2
import random
from seed_definer import *


def data_split():

    set_random_seed(0)

    data_path = '/home/okushi/KHI_Data/Data/*'
    train_path = '/home/okushi/KHI_Data/train'
    val_path = '/home/okushi/KHI_Data/val'
    test_path = '/home/okushi/KHI_Data/test'
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    data_folder_path = sorted(glob.glob(data_path))

    cut_index = [0, 2, 3, 7]
    classes = ['1','2','3','4','5','6','7','8','9']
    data_cut = {'1':5000,'2':0,'3':5000,'4':5000,'5':0,'6':0,'7':0,'8':5000,'9':0} # 多すぎるデータを削除する部分

    for e in classes:
        os.makedirs(os.path.join(train_path, e), exist_ok=True)
        os.makedirs(os.path.join(val_path, e), exist_ok=True)
        os.makedirs(os.path.join(test_path, e), exist_ok=True)
    
    for i, p in enumerate(data_folder_path):

        data_file_path = sorted(glob.glob(p + '/*'))

        if i in cut_index:
            data_file_path = random.sample(data_file_path, data_cut[classes[i]])

        data_num = len(data_file_path)
        train_data_num = int(data_num * 0.8)
        val_data_num = int(data_num * 0.1)
        test_data_num = data_num - train_data_num - val_data_num

        train_data_file_path = random.sample(data_file_path, train_data_num)
        remain_data_file_path = [e for e in data_file_path if e not in train_data_file_path]
        val_data_file_path = random.sample(remain_data_file_path, val_data_num)
        test_data_file_path = [e for e in remain_data_file_path if e not in val_data_file_path]
        print(train_data_file_path[:3])
        print(len(train_data_file_path))
        print(val_data_file_path[:3])
        print(len(val_data_file_path))
        print(test_data_file_path[:3])
        print(len(test_data_file_path))

        for j in range(len(train_data_file_path)):
            im_path = train_data_file_path[j]
            print(im_path)
            write_path = im_path.replace('/Data','/train')
            # write_path = os.path.join(write_path, 'train', classes[i], str(j))
            print(write_path)
            img = cv2.imread(im_path)
            cv2.imwrite(write_path, img)
    
        for j in range(len(val_data_file_path)):
            im_path = val_data_file_path[j]
            write_path = im_path.replace('/Data','/val')
            # write_path = os.path.join(write_path, 'val', classes[i], str(j))

            img = cv2.imread(im_path)
            cv2.imwrite(write_path, img)
    

        for j in range(len(test_data_file_path)):
            im_path = test_data_file_path[j]
            write_path = im_path.replace('/Data','/test')
            # write_path = os.path.join(write_path, 'test', classes[i], str(j))

            img = cv2.imread(im_path)
            cv2.imwrite(write_path, img)
    
        print('make ' + classes[i] + ' finished.')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    data_split()