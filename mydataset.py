import glob


import numpy as np
import cv2


import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms


from utils.import_libraries import *


class Mydataset(torch.utils.data.Dataset):

    def __init__(self, mode):
        path = '/data/dataset/okushi/Data/' + mode + '/*'
        #path = '/home/okushi/Data/' + mode + '/*'

        data_folder_path = sorted(glob.glob(path))

        self.pathes = []
        self.labels = []

        if mode == 'train':
            self.transform = transforms.Compose([
                    #transforms.Resize(resize, Image.ANTIALIAS),
                    #transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(p=0.5), #左右反転(全サンプルのp％を変換)
                    #transforms.RandomVerticalFlip(p=0.5), #上下反転(全サンプルのp％を変換)
                    transforms.ColorJitter(brightness=0.2),
                    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
        else:
            self.transform = transforms.Compose([
                    #transforms.Resize(resize, Image.ANTIALIAS),
                    #transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])

        for i,p in enumerate(data_folder_path):
            data_file_path = sorted(glob.glob(p+'/*'))
            logger.info(len(data_file_path))

            # if i in cut_index and mode == 'train':
            #     data_file_path = random.sample(data_file_path, data_cut[classes[i]])

            self.pathes += data_file_path
            for _ in range(len(data_file_path)):
                self.labels.append(torch.tensor(i, dtype=torch.long))

        self.datanum = len(self.pathes)
        # print(self.datanum)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        im_path = self.pathes[idx]
        label = self.labels[idx]

        img = cv2.imread(im_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        new_shape = (w // 4, h // 4)
        img = cv2.resize(img, dsize=new_shape)
        img = self.transform(img)
        #img = torch.FloatTensor(img)

        return img, label


if __name__ == '__main__':
    train_data = Mydataset('train')
    test_data = Mydataset('test')
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, drop_last=False, num_workers=2)
    for i, (x,t) in enumerate(train_loader):
        print(type(x))
        print(type(t))
        print(x.shape)
        print(t.shape)
