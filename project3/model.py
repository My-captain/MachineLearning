# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2022-05-20 12-58
@file: model.py
"""
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, ngpu, nc, nz, ngf):
        """

        :param ngpu: GPU数量
        :param nc: 训练图片的通道数
        :param nz: latent space维度
        :param ngf: Size of feature maps in generator
        """
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.class_channel = nn.Sequential(
            nn.Linear(10, 128 * 64),
            nn.ReLU(),
            nn.BatchNorm1d(128 * 64)
        )
        self.noise_channel = nn.Sequential(
            nn.Linear(100, 128 * 64),
            nn.ReLU(),
            nn.BatchNorm1d(128 * 64)
        )

        self.main = nn.Sequential(
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(256, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, class_input, noise_input):
        class_input = self.class_channel(class_input).view(-1, 128, 8, 8)
        noise_input = noise_input.view(-1, 100)
        noise_input = self.noise_channel(noise_input).view(-1, 128, 8, 8)
        return self.main(torch.cat([class_input, noise_input], dim=1))


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
