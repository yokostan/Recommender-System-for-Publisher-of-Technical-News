
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torch.nn.functional as F

from pathlib import Path
from torchvision import transforms


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import os
import gc
import sys
import copy
import pickle
import logging
import argparse

if not os.path.isdir('./logs_3000epoch'):
    os.mkdir('./logs_3000epoch')
if not os.path.isdir('./models_3000epoch'):
    os.mkdir('./models_3000epoch')
if not os.path.isdir('./models_ml'):
    os.mkdir('./models_ml')
if not os.path.isdir('./training_data'):
    os.mkdir('./training_data')


training_data_path = '../../snowplow_processing/data_50_cluster/training_data.csv'
training_label_path = '../../snowplow_processing/data_50_cluster/training_labels.csv'

#gsutil_medium_path = sys.argv[1]
#gsutil_medium_path = 'digital-trend-data-unzipped/training_data/complete_data'

if not os.path.isfile(training_data_path):
    #!gsutil -m cp gs://{gsutil_medium_path}/training_data.csv ./{training_data_path}
    print('training_data not exist')
        
if not os.path.isfile(training_label_path):
    #!gsutil -m cp gs://{gsutil_medium_path}/training_labels.csv ./{training_data_path}
    print('training_label not exist')

def set_logger(lr):
    logger = logging.getLogger('wide and deep')
    logging_name = './logs_3000epoch/lr_{}.txt'.format(lr)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)10s][%(levelname)s] %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(logging_name)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info('arguments:{}'.format(" ".join(sys.argv)))
    return logger


X_train = pd.read_csv(training_data_path, header=None)
Y_train = pd.read_csv(training_label_path, header=None)
print('X_train.shape:', X_train.shape, 'y_train.shape', Y_train.shape)

"""
#metric learning
print('----------------------------------------metric learning------------------------------------')

class simpleNet(nn.Module):

    def __init__(self, user_feature_size, news_feature_size):
        super(simpleNet, self).__init__()
        self.fc = nn.Linear(user_feature_size, news_feature_size)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, users, posts):
        users_post = self.fc(users)
        h = self.cos(users_post, posts)
        return h

def test_simpleNet():
    dtype = torch.float32 
    x = torch.randn((176359,65), dtype=dtype)  
    y = torch.randn((176359,32), dtype=dtype) 
    model = simpleNet(x.shape[1], y.shape[1])
    scores = model(x, y)
    print("test simpleNet:", scores.size())  
test_simpleNet() 


def train_model(model, scheduler, optimizer, epochs, logger):
    l = [0]*epochs
    for epoch in range(epochs):
        optimizer.zero_grad()    
        outputs = model(users_train, posts_train)
        loss = nn.functional.binary_cross_entropy((outputs + 1) / 2, y_train)
        loss.backward()
        optimizer.step()
        l[epoch] = loss.data
        if epoch % 20 == 0:
            logger.info(f"Training Results - Epoch: {epoch} loss: {loss:.4f}")
        if epoch % 100 == 0:
            torch.save({
        'model': model,
        'loss': loss}, "./models_ml/model_{0:04d}_.pwf".format(epoch))   
            scheduler.step(loss)
            
    return l


print('X_train.shape:', X_train.shape, 'y_train.shape', Y_train.shape)
user_dimension = X_train.shape[1] - 2 - 32
post_dimension = 32
output_dimension = 1
users_train = torch.Tensor(X_train.drop(X_train.columns[0:2], axis=1).drop(X_train.columns[user_dimension + 2:X_train.shape[1]], axis=1).values)
posts_train = torch.Tensor(X_train.drop(X_train.columns[0:2 + user_dimension], axis=1).values)
y_train = torch.Tensor(Y_train.values)

print('users_train.shape', users_train.shape, 'posts_train.shape', posts_train.shape,'y_train.shape', y_train.shape)

_, user_features = users_train.shape #size user features
_, post_features = posts_train.shape #size M


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device:', device)

loss_all = []

lr = 1e-3
logger = set_logger(lr)
model = simpleNet(user_features, post_features)
if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        print('multi gpu')
        model = torch.nn.DataParallel(model)
model.to(device)
    
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
loss = train_model(model, scheduler, optimizer, 3000, logger)
torch.save({
        'model': model,
        'loss': loss}, "./models_ml/model_metric_learning_{0:03f}.pwf".format(lr))  

np.savetxt("train_loss.csv", loss, delimiter=",")

fig = plt.figure()
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.plot(loss)
plt.savefig('./models_ml/model_loss.png')
"""
#wide and deep
print('----------------------------------------wide and deep------------------------------------')
x = torch.Tensor(X_train.drop(X_train.columns[0:2], axis=1).values)
y = torch.Tensor(Y_train.values)
print(x.shape, y.shape)


Din = X_train.shape[1] - 2
Dout = 1


class WideAndDeep(nn.Module):

    def __init__(self, input_dimension, output_dimension):
        super(WideAndDeep, self).__init__()
        self.batchnorm_1 = nn.BatchNorm1d(num_features = input_dimension)
        self.batchnorm_2 = nn.BatchNorm1d(num_features = 64)
        self.batchnorm_3 = nn.BatchNorm1d(num_features = 32)
        self.fc_wide = nn.Linear(input_dimension, output_dimension)
        self.fc_deep_1 = nn.Linear(input_dimension, 64)
        self.fc_deep_2 = nn.Linear(64, 32)
        self.fc_deep_3 = nn.Linear(32, output_dimension)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x_train):
        x_norm_1 = self.batchnorm_1(x_train)
        f_wide = self.fc_wide(x_norm_1)
        
        out_layer1_fc = self.fc_deep_1(x_norm_1)
        out_layer1_relu = self.relu(out_layer1_fc)
        
        out_layer2_bn = self.batchnorm_2(out_layer1_relu)
        out_layer2_fc = self.fc_deep_2(out_layer2_bn)
        out_layer2_relu = self.relu(out_layer2_fc)
        
        out_layer3_bn = self.batchnorm_3(out_layer2_relu)
        out_layer3_fc = self.fc_deep_3(out_layer3_bn)
        f_deep = self.relu(out_layer3_fc)
        
        scores = self.sigmoid(f_wide + f_deep)
        return scores
    
def test_WideAndDeep():
    dtype = torch.float32 
    x = torch.randn((310288, 97), dtype=dtype)  
    y = torch.randn((310288, 1), dtype=dtype) 
    net = WideAndDeep(x.shape[1], y.shape[1])
    scores = net(x)
    print("test Wide and Deep:", scores.size())  
test_WideAndDeep()


def train_WDNet(net, optimizer, epochs, logger): 
    print("start training")
    l = [0]*epochs
    for e in range(epochs):
        optimizer.zero_grad()    
        scores = net(x)
        loss = nn.functional.binary_cross_entropy(scores, y)
        loss.backward()
        optimizer.step()
        l[e] = loss.data
        logger.info(f"Training Results - Epoch: {e} loss: {loss:.4f}")
        print(f"Training Results - Epoch: {e} loss: {loss:.4f}")

        if e % 100 == 0:
            torch.save({
            'model': net,
            'loss': loss}, "./models_3000epoch/epoch_{0:04d}.pwf".format(e))  
    return l, scores

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device:', device)

learning_rate = [1e-2]
loss_all = []
for i in range(len(learning_rate)):
    lr = learning_rate[i]
    logger = set_logger(lr)
    net = WideAndDeep(Din, Dout)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print('multi gpu')
            net = torch.nn.DataParallel(net)
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    loss, wide_scores = train_WDNet(net, optimizer, 1000, logger)
    loss_all.append(loss)


for i in range(len(loss_all)):
    fig = plt.figure()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    loss = loss_all[i]
    plt.plot(loss)
    plt.savefig('./models_3000epoch/model_{0:02d}.png'.format(i))
    #plt.show()
