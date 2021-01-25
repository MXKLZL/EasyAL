import sys
import argparse
sys.path.insert(1, '..')
import os
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision import datasets, models
import matplotlib.pyplot as plt
from example.GroceriesDataset import *
from BaseModel import *
from query_strategy import *
from Evaluation import *
from LossPredictBaseModel import *
import pickle
from utils import *
from ModelConstructor import get_model
from MultiTransformDataset import TransformTwice

'''
The dataset used in this example is Freiburg Groceries Dataset (http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/)
The uncompressed file is in this structure: 'images/[class_name]/[image_name].png'
'''

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
 'WATER': 1} # This is a fixed class_name to integer target conversion map


NUM_LABEL_PER_ROUND = 20


configs = {'transforms': [# data transformation object for training set 
                                transforms.Compose([
                                            transforms.Resize(224),
                                            transforms.RandomAffine(30, scale = (0.8,1.5)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                            #AddGaussianNoise(0., 1.)
                                            ]
                                            ),
                         # data transformation object for test set
                                transforms.Compose([
                                            transforms.Resize(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ]),
                         # double data transformation object for training set needed for Mean Teacher model
                                TransformTwice(transforms.Compose([
                                            transforms.Resize(224),
                                            transforms.RandomAffine(30, scale = (0.8,1.5)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                            #AddGaussianNoise(0., 1.)
                                            ]
                                            )),
],
          'num_ft_layers': 5,
          'loss_function': nn.CrossEntropyLoss,
          'num_class': 25,
          'weighted_loss': True,
          #'epoch_loss':6,
          #'margin':1.0,
          #'lambda':1,
          'pretrained':False,
          'alpha' :0.6,
          'ramp_length':2,
          'epoch':3,
          'labeled_batch_size':32,
          'unlabeled_batch_size':24
           }


img_root = 'example/images' # The path of the folder of your images
target_list = []
path_list = []

# parse image folder to get a list of image paths and corresponding class_names
for target in os.listdir(img_root):
    if target.startswith('.'):
        continue
    for f in os.listdir(os.path.join(img_root, target)):
        path_list.append(os.path.join(img_root, target,f))
        target_list.append(target)

# use MultiTransformDataset class to enable different modes of data transformation in a single dataset object
dataset= MultiTransformDataset(path_list, class_name_map=class_name_map,classes=list(class_name_map.keys()), transform =configs['transforms']) 

# get image idex and image basename conversion mapping for label-studio use
idx2base, base2idx = get_mapping(dataset)

response = ''

# get corresponding model wrapper object for active learning
model = get_model(dataset, 'mobilenet', configs, model_type = 'Basic')

while response != 'end':
    print('Parsing annotations..')

    # parse human annotations from label-studio generated files
    index_list, target_list = read_from_oracle('example/my-project/completions', idx2base, base2idx)
    target_list = [class_name_map[t] for t in target_list]
    print(f'Got {len(target_list)} labeled samples' )

    # update the targets with human annotations to dataset object
    dataset.update_target(index_list, target_list)
    print('Fitting model...')

    # update model
    model.update()

    # fit model
    model.fit()

    # query image index for next round's labeling, here we used strategy, 'margin', stands for margin sampling
    print(f'Quering {NUM_LABEL_PER_ROUND} samples...')
    query_time, queried_index = query('margin',model,NUM_LABEL_PER_ROUND)

    # update json file to make label-studio show images queried by active learning strategy
    update_json('example/my-project/tasks.json',queried_index,idx2base, base2idx, model, dataset, class_name_map)
    respones = input('Input "end" to stop or "next" to go to next round')

