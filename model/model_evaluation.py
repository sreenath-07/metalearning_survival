
import sys
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import torch
import random
from torch import nn
from torch.autograd import Variable
import pandas as pd
from operator import add
import time
import argparse
import json
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

class DAPLModel(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

        self.main = nn.Sequential(
            nn.Linear(336, 120),
            nn.ReLU(),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Linear(60, 20),
            nn.ReLU(),
            nn.Linear(20, 1, bias=False)
        )

    def forward(self, x):
        return self.main(x)

# model = DAPLModel()
# model.load_state_dict(torch.load("800_iterations_pancan_v1.pt"))
# # model = torch.load("800_iterations_pancan_v1.pt")
# print(model.eval())

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc


# metatrain_test_data_path= ""


sample_pancan = pd.read_csv("sample_data/pretrainPanCan/feature_train.csv")
sample_pancan.shape