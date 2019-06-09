import numpy as np
import pandas as pd
import os
import gc
import sys
import copy
import pickle
import logging
import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms
from matplotlib import pyplot as plt
from scipy.spatial import distance


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

def generate_candidate_post(user_feature, post_feature_dict, K):
    clustering = user_feature[-K:]
    user_cluster = [i for i in range(K) if clustering[i] > 0]
    candidate_post_feature = []
    candidate_post_idx_dict = {}
    count = 0
    for c in user_cluster:
        posts = post_feature_dict[str(c)]
        posts_ids = posts.keys() 
        for post_id in posts_ids:
            candidate_post_idx_dict[count] = post_id
            #print(posts[post_id]['feature'])
            candidate_post_feature.append([float(i) for i in posts[post_id]['feature']])
            count += 1
    return np.array(candidate_post_feature), candidate_post_idx_dict, count

def create_feature(user_feature, post_feature, K):
    user_feature = np.array(user_feature).reshape([1,15 + K]).repeat(post_feature.shape[0], axis = 0)
    mask = np.zeros(user_feature.shape)
    mask[:, 0] = post_feature[:, -1]
    user_feature = user_feature - mask
    return torch.from_numpy(np.concatenate((user_feature,post_feature[:,:-2]),axis = 1))

def find_best_rank(score_idx, candidate_post_idx_dict, threshold, post_cluster_dict, post_feature_dict, user_clicks_in_test):
    best_rank = None
    for i in range(len(score_idx)):
        idx = int(score_idx[i])
        rec_post_id = candidate_post_idx_dict[idx]
        rec_post_cluster = post_cluster_dict[rec_post_id]
        rec_post_feature = post_feature_dict[rec_post_cluster][rec_post_id]['feature']
        for post in user_clicks_in_test:
            test_post_id = post
            test_post_cluster = post_cluster_dict[test_post_id]
            test_post_feature = post_feature_dict[test_post_cluster][test_post_id]['feature']
            if test_post_cluster == rec_post_cluster and distance.cosine(test_post_feature[:-2], rec_post_feature[:-2]) < 1 - threshold:
                best_rank = i
                return best_rank
    return best_rank
        
def get_topK_post_title(score_idx, candidate_post_idx_dict, post_cluster_dict, post_feature_dict, K):
    output = []
    for i in range(K):
        idx = int(score_idx[i])
        post_id = candidate_post_idx_dict[idx]
        post_cluster = post_cluster_dict[post_id]
        post_title = post_feature_dict[post_cluster][post_id]['title']
        output.append(post_title)
    return output
        
def check_accuracy(user_nums, user_clicklist_test, user_idx_dict, users_feature, post_feature_dict, post_cluster_dict, model, threshold, K, post_topk): 
    with torch.no_grad():
        best_ranks = np.zeros(user_nums)
        rec_posts = []
        i = 0
        for user_id in user_clicklist_test.keys():
            user_clicks_in_test = user_clicklist_test[user_id].split(',')
            user_idx = user_idx_dict[user_id]
            
            #user_clicks_in_train = user_clicklist_train[user_id].split(',')
            
            candidate_post_feature, candidate_post_idx_dict, num_candidate_post  = generate_candidate_post(users_feature[user_idx], post_feature_dict, K)

            x_train = create_feature(users_feature[user_idx], candidate_post_feature, K)
            
            scores = model(x_train.float())
            scores = scores.numpy()

            score_idx = np.argsort(-scores, axis = 0)
        
            rank = find_best_rank(score_idx, candidate_post_idx_dict, threshold, post_cluster_dict, post_feature_dict, user_clicks_in_test)
            top_K_posts = get_topK_post_title(score_idx, candidate_post_idx_dict, post_cluster_dict, post_feature_dict, post_topk)
            rec_posts.append(top_K_posts)
            if rank == 1:
                print(user_id)
            if rank != None:
                best_ranks[i] = rank + 1
            else:
                best_ranks[i] = 70000
            if i % 100 == 0:
                print(i, ' has been processed')
            i += 1
        mrr = np.sum(1 / best_ranks) / user_nums           
    return mrr, best_ranks, rec_posts


def plot_result(best_r, num_bins_width):
    fig1, plot1 = plt.subplots()
    plt.plot(best_r, 'ro', markersize=1)
    plot1.set_xlabel('user')
    plot1.set_ylabel('post rank')  
    plot1.set_xlim([0, 3500])
    plot1.set_ylim([0, 10000])
    plt.show()

    fig2, plot2 = plt.subplots()
    n,bins,patches = plot2.hist(best_r, num_bins_width, density=0, histtype='barstacked')
    plot2.set_xlabel('rank') 
    plot2.set_ylabel('number')   
    plot2.set_title('equal width Histogram')  
    plot2.set_xlim([0, 500])
    fig2.tight_layout()
    plt.show()
    
def print_result(best_ranks, user_nums, n1, n2, n3, n4):
    best_ranks = np.array(best_ranks)
    mrr = np.sum(1 / best_ranks) / user_nums
    print("result: ", mrr, best_ranks, best_ranks.min(), best_ranks.max())
    print(np.sum(best_ranks <= n1))
    print(np.sum(best_ranks <= n2))
    print(np.sum(best_ranks <= n3))
    print(np.sum(best_ranks <= n4))
    
def print_rec_post_title(user_id, user_idx_dict, rec_posts, user_clicklist_train, post_cluster_dict, post_feature_dict, K):
    """
    clicklist_train = user_clicklist_train[user_id].split(',')
    print("===================training post=======================")
    for click in clicklist_train:
        cluster = post_cluster_dict[click]
        title = post_feature_dict[cluster][click]['title']
        print(title)
    print("===================recommended post=======================")
    idx = user_idx_dict[user_id]
    rec_posts = rec_posts[idx]
    for title in rec_posts:
        print(title)
        
    """
    clicklist_train = user_clicklist_train[user_id].split(',')
    idx = user_idx_dict[user_id]
    rec_posts = rec_posts[idx]
    cluster_val = [0] * K
    cluster_train = [0] * K
    for click in clicklist_train:
        cluster = post_cluster_dict[click]
        cluster_train[int(cluster)] += 1
    for title in rec_posts:
        for cluster_id in post_feature_dict.keys():
            posts = post_feature_dict[cluster_id]
            for post_id in posts.keys():
                post = posts[post_id]
                if post['title'] == title:
                    cluster_val[int(cluster_id)] += 1
    print(cluster_val)
    print(cluster_train)
    return cluster_train, cluster_val
    
  