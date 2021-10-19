
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

model = DAPLModel()
model.load_state_dict(torch.load("800_iterations_pancan_v1.pt"))
model = torch.load("800_iterations_pancan_v1.pt")