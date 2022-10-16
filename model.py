import os
import random


import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from utils.timeDistributed import *
import torchvision.models


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EncoderBlock(nn.Module): # ConvとBatchNormとreluをあわせたブロック
    def __init__(self, in_feature, out_feature):
        super(EncoderBlock, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.conv = nn.Conv2d(in_channels=in_feature, out_channels=out_feature, kernel_size=3, stride=1, padding=1, dilation=1)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        layers = []
        layers.append(self.conv) 
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module): # モデルで行う畳み込みをまとめたブロック
    def __init__(self, dim):
        super(Encoder, self).__init__()

        layers = []
        layers.append(EncoderBlock(3  , dim))
        layers.append(EncoderBlock(dim, dim))
        layers.append(EncoderBlock(dim, dim))
        layers.append(nn.MaxPool2d(kernel_size=2))

        layers.append(EncoderBlock(dim, dim*2))
        layers.append(EncoderBlock(dim*2, dim*2))
        layers.append(EncoderBlock(dim*2, dim*2))
        layers.append(nn.MaxPool2d(kernel_size=2))

        layers.append(EncoderBlock(dim*2, dim*4))
        layers.append(EncoderBlock(dim*4, dim*4))
        layers.append(EncoderBlock(dim*4, dim*4))
        layers.append(nn.MaxPool2d(kernel_size=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Model(nn.Module):
  def __init__(self, param, class_num=9, enc_dim=32, in_w=64, in_h=64, fc1=1024, fc2=64):
    super(Model, self).__init__()

    self.enc_dim = param.conv_channel
    self.in_w = in_w
    self.in_h = in_h
    self.fc_dim = self.enc_dim * 4 * int(in_h/2/2/2) * int(in_w/2/2/2)

    self.Encoder = Encoder(self.enc_dim)
    self.fc1 = nn.Linear(self.fc_dim, param.fc1_dim)
    self.fc2 = nn.Linear(param.fc1_dim, param.fc2_dim)
    self.fc3 = nn.Linear(param.fc2_dim, class_num)

    nn.init.xavier_uniform_(self.fc1.weight)
    nn.init.zeros_(self.fc1.bias)

    nn.init.xavier_uniform_(self.fc2.weight)
    nn.init.zeros_(self.fc2.bias)

    nn.init.xavier_uniform_(self.fc3.weight)
    nn.init.zeros_(self.fc3.bias)
  
  def forward(self, x):
    out = self.Encoder(x)
    out = out.view(-1, self.fc_dim)

    out = F.relu(self.fc1(out))
    out = F.relu(self.fc2(out))
    out = self.fc3(out)

    return out


class Color_Model(nn.Module):
  def __init__(self, class_num=7, enc_dim=64, in_w=64, in_h=64, fc1=1024, fc2=64):
    super(Color_Model, self).__init__()

    self.enc_dim = enc_dim
    self.in_w = in_w
    self.in_h = in_h
    self.fc_dim = enc_dim * 4 * int(in_h/2/2/2) * int(in_w/2/2/2)

    self.Encoder = Encoder(self.enc_dim)
    self.fc1 = nn.Linear(self.fc_dim, fc1)
    self.fc2 = nn.Linear(fc1, fc2)
    self.fc3 = nn.Linear(fc2, class_num)

    nn.init.xavier_uniform_(self.fc1.weight)
    nn.init.zeros_(self.fc1.bias)

    nn.init.xavier_uniform_(self.fc2.weight)
    nn.init.zeros_(self.fc2.bias)

    nn.init.xavier_uniform_(self.fc3.weight)
    nn.init.zeros_(self.fc3.bias)
  
  def forward(self, x):
    out = self.Encoder(x)
    out = out.view(-1, self.fc_dim)

    out = F.relu(self.fc1(out))
    out = F.relu(self.fc2(out))
    out = self.fc3(out)

    return out


class State_Model(nn.Module): 
  def __init__(self, class_num=3, enc_dim=64, in_w=64, in_h=64, fc1=1024, fc2=64):
    super(State_Model, self).__init__()

    self.enc_dim = enc_dim
    self.in_w = in_w
    self.in_h = in_h
    self.fc_dim = enc_dim * 4 * int(in_h/2/2/2) * int(in_w/2/2/2)

    self.Encoder = Encoder(self.enc_dim)
    self.fc1 = nn.Linear(self.fc_dim, fc1)
    self.fc2 = nn.Linear(fc1, fc2)
    self.fc3 = nn.Linear(fc2, class_num)

    nn.init.xavier_uniform_(self.fc1.weight)
    nn.init.zeros_(self.fc1.bias)

    nn.init.xavier_uniform_(self.fc2.weight)
    nn.init.zeros_(self.fc2.bias)

    nn.init.xavier_uniform_(self.fc3.weight)
    nn.init.zeros_(self.fc3.bias)
  
  def forward(self, x):
    out = self.Encoder(x)
    out = out.view(-1, self.fc_dim)

    out = F.relu(self.fc1(out))
    out = F.relu(self.fc2(out))
    out = self.fc3(out)

    return out
