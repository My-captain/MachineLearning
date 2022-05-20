# -*- coding:utf-8 -*-
"""
author: zliu.elliot
@time: 2022-05-10 14-54
@file: main.py
"""
from __future__ import unicode_literals, print_function, division

import os

import torch
from torch import nn
import random
import time
import math
from models import RNN
from project2.utils import findFiles, readLines, n_letters, lineToTensor


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
n_hidden = 128

criterion = nn.NLLLoss()
learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn
n_iters = 100000
print_every = 1000
plot_every = 1000
device = "cuda"

# Keep track of losses for plotting
current_loss = 0
all_losses = []

# rnn = RNN(n_letters, n_hidden, n_categories).to(device)
rnn = nn.LSTM(n_letters, n_hidden, bidirectional=False).to(device)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate, momentum=0.5)


def train(category_tensor, line_tensor):
    category_tensor = category_tensor.to(device)
    line_tensor = line_tensor.to(device)
    hidden = rnn.initHidden().to(device)
    for idx in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[idx], hidden)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return output, loss.item()


start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000


# Just return an output given a line
def evaluate(line_tensor):
    with torch.no_grad():
        hidden = rnn.initHidden().to(device)
        line_tensor = line_tensor.to(device)
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        return output.cpu()


# Go through a bunch of examples and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()


def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])


predict('Dovesky')
predict('Jackson')
predict('Satoshi')
