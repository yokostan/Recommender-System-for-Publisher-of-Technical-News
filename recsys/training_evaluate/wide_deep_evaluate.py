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

from wide_deep_evaluate_tools import *
import pickle


for K in [100]:
    print("running k = {}".format(K))
    threshold = 0.95
    post_topk = 10
    user_click_train_path = './evaluate_data/user_clicklist_train.csv'
    user_click_val_path = './evaluate_data/user_clicklist_val.csv'
    user_click_test_path = './evaluate_data/user_clicklist_test.csv'

    user_feature_path = '../../snowplow_processing/data_{}_cluster/user_feature_{}.json'.format(K, K)
    post_feature_path = '../../snowplow_processing/data_{}_cluster/clustered_post_{}.json'.format(K, K)

    post_cluster_path = '../../snowplow_processing/data_{}_cluster/post_feature_{}.json'.format(K, K)

    model_path = '../K{}/models/epoch_0900.pwf'.format(K)

    if not os.path.isfile(user_click_train_path):
        raise Exception("no training user clicklist")    
    if not os.path.isfile(user_click_val_path):
        raise Exception("no validation user clicklist")
    if not os.path.isfile(user_click_test_path):
        raise Exception("no test user clicklist")        
    if not os.path.isfile(user_feature_path):
        raise Exception("no user feature")        
    if not os.path.isfile(post_feature_path):
        raise Exception("no post feature")  
    if not os.path.isfile(model_path):
        raise Exception("no model")
    if not os.path.isfile(post_cluster_path):
        raise Exception("post cluster")

    # load data
    user_click_val = pd.read_csv(user_click_val_path, header=None)
    user_click_test = pd.read_csv(user_click_test_path, header=None)
    user_click_train = pd.read_csv(user_click_train_path, header=None)
    with open(user_feature_path) as json_file:  
        user_feature_dict = json.load(json_file)
    with open(post_feature_path) as json_file:  
        post_feature_dict = json.load(json_file)
    with open(post_cluster_path) as json_file:  
        post_cluster_dict = json.load(json_file)
    for post_id in post_cluster_dict.keys():
        feature = post_cluster_dict[post_id]
        post_cluster_dict[post_id] = str(int(feature[-2]))

    # Model class must be defined somewhere
    model = torch.load(model_path)['model']
    model.eval()

    # build user idx dict
    user_idx_dict = {}
    user_feature = []
    i = 0
    for user_id in user_feature_dict.keys():
        user_idx_dict[user_id] = i
        i += 1
        user_feature.append([1553990399] + user_feature_dict[user_id])
    user_feature = np.array(user_feature) 
    print(user_feature.shape)

    # user clicklist in training set
    user_clicklist_train = {}
    for i in range(user_click_train.shape[0]):
        user_clicklist_train[user_click_train.loc[i, 0]] = user_click_train.loc[i, 1]
    print(len(user_clicklist_train))

    # user clicklist in val set
    user_clicklist_val = {}
    for i in range(user_click_val.shape[0]):
        user_clicklist_val[user_click_val.loc[i, 0]] = user_click_val.loc[i, 1]
    print(len(user_clicklist_val))
    user_num_val = len(user_clicklist_val)

    # user clicklist in test set
    user_clicklist_test = {}
    for i in range(user_click_test.shape[0]):
        user_clicklist_test[user_click_test.loc[i, 0]] = user_click_test.loc[i, 1]
    print(len(user_clicklist_test))
    user_num_test = len(user_clicklist_test)

    print("evaluate validation")
    mrr_val, best_ranks_val, rec_posts_val = check_accuracy(user_num_val, user_clicklist_val, user_idx_dict,\
                                                user_feature, post_feature_dict, post_cluster_dict, model, threshold, K, post_topk)
    
    save_path_best_ranks_val = 'best_ranks_val_{}'.format(K)
    save_path_rec_posts_val = 'rec_posts_val_{}'.format(K)
    np.savetxt(save_path_best_ranks_val, best_ranks_val)
    with open(save_path_rec_posts_val, 'wb') as fp:
        pickle.dump(rec_posts_val, fp)
    
    print("evaluate test")
    mrr_test, best_ranks_test, rec_posts_test = check_accuracy(user_num_test, user_clicklist_test, user_idx_dict,\
                                                user_feature, post_feature_dict, post_cluster_dict, model, threshold, K, post_topk)

    
    save_path_best_ranks_test = 'best_ranks_test_{}'.format(K)
    save_path_rec_posts_test = 'rec_posts_test_{}'.format(K)
    np.savetxt(save_path_best_ranks_test, best_ranks_test)
    with open(save_path_rec_posts_test, 'wb') as fp:
        pickle.dump(rec_posts_test, fp)
