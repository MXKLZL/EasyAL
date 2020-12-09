import sys
import argparse
sys.path.insert(1, '/Users/zhangzhanming/Desktop/KPMG/Capstone-Project')
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from sklearn.model_selection import train_test_split
from torchvision import datasets, models
import matplotlib.pyplot as plt
from GroceriesDataset import *
from BaseModel import *
from query_strategy import *
from Evaluation import *
from LossPredictBaseModel import *
import pickle
from utils import *
from ModelConstructor import get_model_class


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

NUM_INITIAL_LAB = 200
TEST_SET_RATIO = 0.2
NUM_ROUND = 15
NUM_LABEL_PER_ROUND = 100
BATCH_SIZE = 32
DISTANCE = 'cosine'
STANDARDIZE = True


configs = {'transforms': [transforms.Compose([
                                            transforms.Resize(224),
                                            transforms.RandomAffine(30, scale = (0.8,1.5)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                            #AddGaussianNoise(0., 1.)
                                            ]
                                            ),
                                transforms.Compose([
                                            transforms.Resize(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ]),
                                TransformTwice(transforms.Compose([
                                            transforms.Resize(224),
                                            transforms.RandomAffine(30, scale = (0.8,1.5)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                            #AddGaussianNoise(0., 1.)
                                            ]
                                            )),
],
          'batch_size': BATCH_SIZE,
          'num_ft_layers': 5,
          'loss_function': nn.CrossEntropyLoss,
          'num_class': 25,
          'weighted_loss': True,
          #'epoch_loss':6,
          #'margin':1.0,
          #'lambda':1,
          'pretrained':False

           }

image_dir = 'images'

def get_initial_label(num):
    tmp = np.arange(len(train_ds))
    np.random.seed(41)
    np.random.shuffle(tmp)
    return tmp[:num]


configs['alpha'] = 0.6
configs['ramp_length'] = 2
configs['epoch'] = 1

configs['labeled_batch_size'] = 8
configs['unlabeled_batch_size'] = 24

img_root = 'images'
target_list = []
path_list = []

for target in os.listdir(img_root):
    if target.startswith('.'):
        continue
    for f in os.listdir(os.path.join(img_root, target)):
        path_list.append(os.path.join(img_root, target,f))
        target_list.append(target)

dataset= MultiTransformDataset(path_list, class_name_map=class_name_map,classes=list(class_name_map.keys()), transform =configs['transforms']) 
idx2base, base2idx = get_mapping(dataset)

response = ''

while response != 'end':
    print('Parsing annotations..')
    index_list, target_list = read_from_oracle('/Users/zhangzhanming/Desktop/KPMG/local/my-project/completions', idx2base, base2idx)
    target_list = [class_name_map[t] for t in target_list]
    print(f'Got {len(target_list)} labeled samples' )
    dataset.update_target(index_list, target_list)
    print('Fitting model...')
    model = get_model_class(dataset, 'mobilenet', configs)
    model.fit()
    print('Quering samples...')
    query_time, queried_index = query('margin',model,20)
    update_json('/Users/zhangzhanming/Desktop/KPMG/local/my-project/tasks.json',queried_index,idx2base, base2idx, model, dataset, class_name_map)
    input('Input "end" to stop or "next" to go to next round')

