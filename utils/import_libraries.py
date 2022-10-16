import os
import random
import copy
import decimal
import csv
import argparse


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from logging import INFO, getLogger, StreamHandler, Formatter, FileHandler, DEBUG
import cv2
from PIL import Image


import torch
from torchsummary import summary
from torch import nn,optim
from torch.functional import split
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from tqdm import *

from utils.logger import * 
from utils.parameter_loader import *
from utils.make_file_name import *


imported_flag = False

if not imported_flag: # はじめてimportされたときにだけdeviceとloggerを作成する。importしたファイルでそれらが使用可能になる

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_name", default='param0', help="setting yaml file name")
    args = parser.parse_args()

    yaml_file = args.yaml_name + ".yaml"
    base_parameters_dir = "./parameters"

    setting_yaml_file = os.path.join(base_parameters_dir, yaml_file)
    parameters = Parameters(setting_yaml_file)

    file_name = make_file_name(parameters)
    os.makedirs("./logs", exist_ok=True)

    logger = get_logger('./logs', file_name+'.log')

    imported_flag = True
