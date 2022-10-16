
import os
import random
import copy
import decimal


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2


import torch
from torch import nn,optim
from torch.functional import split
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
import tqdm


def make_loss_graph(record_loss_train, record_loss_val, file_name):
    plt.plot(range(1,len(record_loss_val)+1),record_loss_val, label="Val Loss", color = "orange")
    plt.plot(range(1,len(record_loss_train)+1),record_loss_train, label="Train Loss", color = "cyan")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend(loc='upper right')
    plt.xlim(1, len(record_loss_train))
    plt.ylim(0, max(max(record_loss_train), max(record_loss_val)))
    os.makedirs("./loss", exist_ok=True)
    plt.savefig('./loss/'+file_name+'_Loss.png') 
    plt.gca().clear()


def make_acc_graph(record_train_accuracy, record_val_accracy, file_name):
    plt.plot(range(len(record_val_accracy)),record_val_accracy, label="Val Loss", color = "orange")
    plt.plot(range(len(record_train_accuracy)),record_train_accuracy, label="Train Loss", color = "cyan")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy')
    plt.legend(loc='lower right')
    plt.xlim(0, len(record_train_accuracy))
    plt.ylim(0, 1)
    os.makedirs("./acc", exist_ok=True)
    plt.savefig('./acc/'+file_name+'_Acc.png') 
    plt.gca().clear()
