# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2022-04-24 19:03
@file: mnist_cnn_train.py
"""
import os

import torch.optim
from torchvision.transforms import transforms
from torchvision import datasets

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as functional

from project1.models import net

n_epoch = 100
learning_rate = 0.01
train_batch_size = 512
test_batch_size = 512
writer = SummaryWriter(log_dir=f"./runs/cnn_lr{learning_rate}_{n_epoch}epoch", flush_secs=2)

least_loss = 10e5
cuda_available = torch.cuda.is_available()


def fit(epoch, model, dataloader, criterion, optimizer, phase="train"):
    if phase == "train":
        model.train()
    else:
        model.eval()
    run_loss = 0
    run_correct = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        if phase == "train":
            optimizer.zero_grad()
        if cuda_available:
            data = data.to("cuda")
            target = target.to("cuda")
        output = model(data)
        predict = output.data.max(dim=1, keepdim=True)[1].to("cuda")

        loss = criterion(output, target)
        run_loss += functional.nll_loss(output, target, reduction='sum').cpu().item()

        predict = predict.reshape(target.shape)
        run_correct += predict.eq(target).sum().cpu().item()

        if phase == "train":
            loss.backward()
            optimizer.step()

    loss = run_loss / len(dataloader.dataset)
    accuracy = float(run_correct) / len(dataloader.dataset)
    writer.add_scalars("Loss", {phase: loss}, epoch)
    writer.add_scalars("Accuracy", {phase: accuracy}, epoch)
    return loss, accuracy


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./mnistDataset/", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./mnistDataset/", train=False, transform=transform, download=True)

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.5)
if cuda_available:
    net = net.to("cuda")


for epoch in range(n_epoch):
    train_loss, train_acc = fit(epoch, net, train_dataloader, loss_fn, optimizer)
    valid_loss, valid_acc = fit(epoch, net, test_dataloader, loss_fn, optimizer, phase="valid")
    if valid_loss < least_loss:
        least_loss = valid_loss
        if not os.path.exists(f"./runs/cnn_lr{learning_rate}_{n_epoch}epoch"):
            os.mkdir(f"./runs/cnn_lr{learning_rate}_{n_epoch}epoch")
        torch.save(net, f"./runs/cnn_lr{learning_rate}_{n_epoch}epoch/model_serial")
        with open(f"./runs/cnn_lr{learning_rate}_{n_epoch}epoch/model_serial_info.txt", mode="w", encoding="utf8") as fp:
            fp.write(f"valid loss:{valid_loss}")



