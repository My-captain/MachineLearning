# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2022-04-24 19:10
@file: models.py
"""
import torch.nn as nn

net = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5),
    nn.MaxPool2d(kernel_size=2),
    nn.ReLU(),
    nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),
    nn.Dropout2d(),
    nn.MaxPool2d(kernel_size=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(320, 50),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(50, 10),
    nn.LogSoftmax()
)
