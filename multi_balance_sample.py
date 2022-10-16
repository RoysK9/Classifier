import glob


import numpy as np
import cv2


import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms

from utils.import_libraries import *


class MultiBalanceSampler:
    def __init__(self, n_samples=10, Out_unit=9, mode='train'):
    
        logger.info(f'use multi balance sampler for {mode} data')
        self.Out_unit = Out_unit
        
        path = '/data/dataset/okushi/Data/' + mode + '/*'
        #path = '/home/okushi/Data/' + mode + '/*'

        data_folder_path = sorted(glob.glob(path))

        self.pathes = []
        self.labels = []

        for i,p in enumerate(data_folder_path):
            data_file_path = sorted(glob.glob(p+'/*'))
            logger.info(len(data_file_path))
            # if i in cut_index and mode == 'train':
            #     data_file_path = random.sample(data_file_path, data_cut[classes[i]])

            self.pathes += data_file_path
            for _ in range(len(data_file_path)):
                self.labels.append(torch.tensor(i, dtype=torch.long))

        self.datanum = len(self.pathes)
        label_counts = np.bincount(self.labels)
        self.major_label = label_counts.argmax()
        # minor_label = label_counts.argmin()
        self.other_indices = []

        for i in range(self.Out_unit):
            self.other_indices.append(np.where(self.labels == np.int64(i))[0])
        
        self.major_indices = self.other_indices[self.major_label]
        # self.major_indices = np.where(labels == major_label)[0]
        # self.minor_indices = np.where(labels == minor_label)[0]
        
        np.random.shuffle(self.major_indices)
        
        self.used_indices = 0
        self.n_samples = n_samples
        self.count = len(self.pathes)
        self.mode = mode
        
        
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

    def __iter__(self):
        self.count = 0
        self.used_indices = 0
        while self.used_indices < len(self.major_indices):
            img_batch = []
            label_batch = []
            # 多数派データ(major_indices)からは順番に選び出し
            # 少数派データ(minor_indices)からはランダムに選び出す操作を繰り返す
            if self.used_indices + self.n_samples <= len(self.major_indices):
                RangeSize = self.n_samples
                indices = self.major_indices[self.used_indices:self.used_indices + RangeSize].tolist() 
                for i in range(self.Out_unit):
                    if not i == self.major_label:
                        indices += np.random.choice(self.other_indices[i], RangeSize, replace=False).tolist()
            else:
                RangeSize = len(self.major_indices) - self.used_indices
                indices = self.major_indices[self.used_indices:self.used_indices + RangeSize].tolist() 
                for i in range(self.Out_unit):
                    if not i == self.major_label:
                        indices += np.random.choice(self.other_indices[i], RangeSize, replace=False).tolist()

            for i in indices:
                im_path = self.pathes[i]
                label = self.labels[i]

                img = cv2.imread(im_path)
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, _ = img.shape
                new_shape = (w // 4, h // 4)
                img = cv2.resize(img, dsize=new_shape)

                img = self.transform(img)
                img_batch.append(img)
                label_batch.append(label)
            
            img_batch = torch.stack(img_batch, dim = 0)
            label_batch = torch.tensor(label_batch)
            self.used_indices += self.n_samples

            yield img_batch, label_batch