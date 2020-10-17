# -*- coding: utf-8 -*-
"""Run.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15Cg7uoa70mKgx1pcmj8XGHAoRjrYhhCo
"""

#!unzip images.zip

#!rm -rf /content/images

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from sklearn.model_selection import train_test_split
from torchvision import datasets, models
import matplotlib.pyplot as plt
from GroceriesDataset import GroceriesDataset
from BaseModel import *
from query_strategy import *
from Evaluation import *

class_name_map = {'BEANS': 22,
 'CAKE': 6,
 'CANDY': 2,
 'CEREAL': 20,
 'CHIPS': 4,
 'CHOCOLATE': 0,
 'COFFEE': 3,
 'CORN': 14,
 'FISH': 12,
 'FLOUR': 5,
 'HONEY': 18,
 'JAM': 21,
 'JUICE': 24,
 'MILK': 16,
 'NUTS': 7,
 'OIL': 13,
 'PASTA': 8,
 'RICE': 23,
 'SODA': 15,
 'SPICES': 9,
 'SUGAR': 10,
 'TEA': 19,
 'TOMATO_SAUCE': 17,
 'VINEGAR': 11,
 'WATER': 1}

#!nvidia-smi

NUM_INITIAL_LAB = 1000
TEST_SET_RATIO = 0.2
NUM_ROUND = 5
NUM_LABEL_PER_ROUND = 200
BATCH_SIZE = 32
FIT_EPOCH = 5
strategy = 'entropy'


configs = {'transforms': [transforms.Compose([
                                            transforms.Resize(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                                transforms.Compose([
                                            transforms.Resize(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])],
          'batch_size': BATCH_SIZE,
          'epoch': FIT_EPOCH,
          'num_ft_layers': 5,
          'loss_function': nn.CrossEntropyLoss(),
          'num_class': 25,
           }

train_path = '/content/train_split_all.txt'
test_path =  '/content/test_split_all.txt'
image_dir = '/content/images'
train_ds = GroceriesDataset(train_path,image_dir,class_name_map,configs['transforms'])

test_ds = GroceriesDataset(test_path,image_dir,class_name_map,configs['transforms'])

def get_initial_label(num):
  tmp = np.arange(len(train_ds))
  np.random.shuffle(tmp)
  return tmp[:num]

testloader = torch.utils.data.DataLoader(test_ds, batch_size=32)
test_target = test_ds.target_list
test_length = len(test_target)
test_ds.set_mode(1)



print('Initial Status')
print('Number of labeled images: {}'.format(NUM_INITIAL_LAB))
print('Number of unlabeled images: {}'.format(len(train_ds) - NUM_INITIAL_LAB))
print('Number of testing images: {}'.format(len(test_ds)))

print('')
print('Begin Train')

strategies = ['random', 'uncertain', 'entropy','margin', 'k_means', 'k_center_greedy']
allacc = []
allssim = []
allcost = []

for s in strategies:
  print(s)
  accuracy = []
  ssim_list = []
  cost_list = []

  label_idx = get_initial_label(NUM_INITIAL_LAB)
  unlabel_idx = np.setdiff1d(np.arange(len(train_ds)), label_idx)
  class_weight = [1]*len(class_name_map)

  for i in range(NUM_ROUND+1):
    print('Round ',i)
    Model = BaseModel(train_ds,'resnet18',label_idx,configs)
    Model.fit()
    if i == 0:
      class_weight = Model.weights

    print('Fit Finished')

    _, pred = torch.max(Model.predict(testloader), 1)
    cur_acc = classification_evaluation(pred, test_target, 'f1', 'macro')
    cate_acc = classification_evaluation(pred, test_target, 'f1', None)
    accuracy.append(cur_acc)

    print('F1 score: ',cur_acc)

    query_time, query_this_round = query(strategy, Model, NUM_LABEL_PER_ROUND)
    images_queried = np.array([train_ds[idx][0].numpy().transpose(1,2,0) for idx in query_this_round])
    ssim = av_SSIM(images_queried, pairs=300)

    del images_queried

    print('SSIM this round ', ssim)
    ssim_list.append(ssim)

    label_idx = np.concatenate((label_idx, query_this_round), axis=None)
    cost = Model.query_cost(query_this_round, class_weight)
    cost_list.append(cost)
    print('')
    print('')
    print('For next Round')
    print('query cost: {}'.format(cost))
    print("New {} unlabel images, Total: {} images, Spend time: {}".format(NUM_LABEL_PER_ROUND, len(label_idx), query_time))

    print('')
  allacc.append(accuracy)
  allssim.append(ssim_list)
  allcost.append(cost_list)

plot_result(allacc, strategies, 'f1 score')

plot_result(allssim, strategies, 'SSIM')
