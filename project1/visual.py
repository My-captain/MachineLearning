# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2022-05-04 22-53
@file: visual.py
"""
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import project1.models

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./mnistDataset/", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./mnistDataset/", train=False, transform=transform, download=True)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
train_dataloader = iter(train_dataloader)


writer = SummaryWriter(log_dir="./runs/visualize")
# net = torch.load("./runs/cnn_lr0.01_800epoch/model_serial")
# writer.add_graph(net, (train_dataloader.next()[0].to("cuda")))
net = project1.models.Model()
writer.add_graph(net.to("cuda"), (train_dataloader.next()[0].to("cuda")))



