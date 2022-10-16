import os
import yaml


import torch
from torch import nn,optim


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

class Parameters:

    def __init__(self, setting_yaml_file):
        all_parameters = self.read_yaml(setting_yaml_file)
        self.model_parameters = all_parameters["model_parameters"]
        self.learning_parameters = all_parameters["learning_parameters"]
        
        self.epochs = self.learning_parameters["epochs"]
        self.batch_size = self.learning_parameters["batch_size"]
        self.lr = self.learning_parameters["lr"]
        self.seed = self.learning_parameters["seed"]
    
        self.input_channel = self.model_parameters["input_channel"]
        self.conv_channel = self.model_parameters["conv_channel"]    
        self.fc1_dim = self.model_parameters["fc1_dim"]
        self.fc2_dim = self.model_parameters["fc2_dim"]  
        

    def read_yaml(self, setting_yaml_file):
        with open(setting_yaml_file) as f:
            return yaml.safe_load(f)

    #文字列のTrueをbool値のTrueに変換しそれ以外をFalseに変換する関数
    def str_to_bool(self, str):
        return str == "true"
