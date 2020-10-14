# -*- coding: utf-8 -*-
"""query_strategy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_G_vtAh9NwFoSrDjE_ZsY_2xTij10rxy
"""
import numpy as np
import torch

def query(strategy,model_class,label_per_round):
  
  if strategy == 'random':
    unlabel_index = model_class.get_unlabeled_index()
    np.random.shuffle(unlabel_index)
    return unlabel_index[:label_per_round]
  
  if strategy == 'uncertain':
    unlabel_index = model_class.get_unlabeled_index()
    p, _ = torch.max(model_class.predict_unlabeled(), 1)
    selected_index = p.numpy().argsort()[:label_per_round]
    return unlabel_index[selected_index]
  
  if strategy =='margin':
    unlabel_index = model_class.get_unlabeled_index()
    p = model_class.predict_unlabeled()
    p = -np.sort(-p,axis = 1)
    #sort every row by descending
    diff = p[:,0] - p[:,1]
    #get difference between first class and second class
    selected_index = np.argsort(diff)[:label_per_round]
    return unlabel_index[selected_index]
  
  if strategy == 'entropy':
    unlabel_index = model_class.get_unlabeled_index()
    p = model_class.predict_unlabeled()
    entropy = (-p*torch.log(p)).sum(1)
    #calculate entropy for each image
    selected_index = np.argsort(entropy.numpy())[::-1][:label_per_round]
    return unlabel_index[selected_index]











  

