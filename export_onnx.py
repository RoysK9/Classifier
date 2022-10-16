
import os
import random
import copy
import decimal
import csv
import argparse


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torchvision.models


from utils.make_file_name import *
from model import *
from utils.parameter_loader import *
from seed_definer import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def export_onnx(param):

    file_name = make_file_name(parameters)
    # Instantiate your model. This is just a regular PyTorch model that will be exported in the following steps.
    model = Model(param)

    input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
    output_names = [ "output1" ]

    best_save_color_model_path = './weights/'+ file_name + '_best_model_weight.pth'
    model.load_state_dict(torch.load(best_save_color_model_path), strict=False)
    # Evaluate the model to switch some operations from training mode to inference.
    model.eval()
    # Create dummy input for the model. It will be used to run the model inside export function.
    dummy_input = torch.randn(10, 3, 64, 64)
    # Call the export function
    torch.onnx.export(model, dummy_input, "best_model.onnx", verbose=True, input_names=input_names, output_names=output_names)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_name", default='param0', help="setting yaml file name")
    args = parser.parse_args()

    yaml_file = args.yaml_name + ".yaml"
    base_parameters_dir = "./parameters"

    setting_yaml_file = os.path.join(base_parameters_dir, yaml_file)
    parameters = Parameters(setting_yaml_file)

    set_random_seed(parameters.seed)

    export_onnx(parameters)