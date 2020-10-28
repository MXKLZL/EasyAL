# -*- coding: utf-8 -*-
"""query_strategy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_G_vtAh9NwFoSrDjE_ZsY_2xTij10rxy
"""
import numpy as np
import torch
import time
import random
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm
import numpy as np

def query(strategy, model_class, label_per_round,alpha = 0.5,add_uncertainty = False):
  start = time.time()

  if strategy == 'random':
    unlabel_index = model_class.get_unlabeled_index()
    np.random.shuffle(unlabel_index)

    end = time.time()
    duration = end - start
    return duration, unlabel_index[:label_per_round]

  if strategy == 'uncertain':
    unlabel_index = model_class.get_unlabeled_index()
    p, _ = torch.max(model_class.predict_unlabeled(), 1)
    selected_index = p.numpy().argsort()[:label_per_round]

    end = time.time()
    duration = end - start
    return duration, unlabel_index[selected_index]

  if strategy == 'margin':
    unlabel_index = model_class.get_unlabeled_index()
    p = model_class.predict_unlabeled()
    p = -np.sort(-p, axis=1)
    # sort every row by descending
    diff = p[:, 0] - p[:, 1]
    # get difference between first class and second class
    selected_index = np.argsort(diff)[:label_per_round]

    end = time.time()
    duration = end - start
    return duration, unlabel_index[selected_index]

  if strategy == 'entropy':
    unlabel_index = model_class.get_unlabeled_index()
    p = model_class.predict_unlabeled()
    entropy = (-p * torch.log(p)).sum(1)
    # calculate entropy for each image
    selected_index = np.argsort(entropy.numpy())[::-1][:label_per_round]

    end = time.time()
    duration = end - start
    return duration, unlabel_index[selected_index]

  if strategy == 'k_means':

    unlabel_loss = None
    if add_uncertainty != False:
      unlabel_loss = get_loss(add_uncertainty,model_class)
    

    unlabel_index = model_class.get_unlabeled_index()
    embedding = np.array(model_class.get_embedding_unlabeled())
    cluster_ = KMeans(n_clusters=label_per_round)
    cluster_.fit(embedding)
    cluster_index = cluster_.predict(embedding)
    centers = cluster_.cluster_centers_[cluster_index]
    dis = np.sum(np.array(pow((embedding - centers), 2)), axis=1)

    centerlabels = []
    for i in range(label_per_round):
      clusterlabel = np.where(cluster_index == i)[0]

      if unlabel_loss is not None:
        dis_tmp = np.power(dis[clusterlabel],alpha)
        uncertainty = np.power(unlabel_loss[clusterlabel],1 - alpha)
        combine = dis_tmp/uncertainty
        centerlabels.append(clusterlabel[combine.argsort()[0]])
      else:
        centerlabels.append(clusterlabel[dis[clusterlabel].argsort()[0]])


    end = time.time()
    duration = end - start
    return duration, unlabel_index[centerlabels]

  if strategy == 'k_means++':

    unlabel_index = model_class.get_unlabeled_index()
    unlabel_embedding = np.array(model_class.get_embedding_unlabeled())
    index = np.random.randint(len(unlabel_index))
    batch = [unlabel_index[index]]
    label_embedding = unlabel_embedding[[index]]
    unlabel_embedding = np.delete(unlabel_embedding, index, 0)
    unlabel_index = np.delete(unlabel_index, index, 0)

    #pick items based on a distribution
    def random_pick(some_list, probabilities):
      x = random.uniform(0, 1)
      cumulative_probability = 0.0
      for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
          break
      return item

    for j in tqdm(range(label_per_round - 1)):
      mindis = []
      for i in range(len(unlabel_index)):
        unlabel_loss = None

        if add_uncertainty:
          unlabel_loss = get_loss(add_uncertainty, model_class)

        dists = get_distance(unlabel_embedding[i], label_embedding, 'euclidean_distance')  # or cosine_distance

        mindis.append(min(dists))

      if add_uncertainty:
        mindis = np.power(unlabel_loss, 1 - alpha) * np.power(mindis, alpha)

      mindis = mindis / sum(mindis)

      picked_index = random_pick(range(0, len(unlabel_index)), mindis)
      batch.append(unlabel_index[picked_index])
      label_embedding = np.append(label_embedding, [unlabel_embedding[picked_index]], 0)
      unlabel_embedding = np.delete(unlabel_embedding, picked_index, 0)
      unlabel_index = np.delete(unlabel_index, picked_index, 0)

    end = time.time()
    duration = end - start
    return duration, np.array(batch)

  if strategy == 'k_center_greedy':
    # unlabel_index = model_class.get_unlabeled_index()
    # unlabel_embedding = np.array(model_class.get_embedding_unlabeled())
    # label_index = model_class.labeled_index
    # label_embedding = np.array(model_class.get_embedding(model_class.data_loader_labeled))

    # # for each unlabeled dataset i, compute its distance with all labeled j, 
    # # and find the smallest distance
    # min_dists = []
    # for i in range(len(unlabel_embedding)):
    #   l2_dists = np.linalg.norm(unlabel_embedding[i] - label_embedding, axis=1)
    #   min_dists.append(l2_dists.min())

    # selected_index = np.argsort(min_dists)[::-1][:label_per_round]
    # # find the unlabeled dataset i with largest minimal distance 
    # end = time.time()
    # duration = end - start
    # return duration, unlabel_index[selected_index]

    unlabel_index = model_class.get_unlabeled_index()
    unlabel_embedding = np.array(model_class.get_embedding_unlabeled())
    label_embedding = np.array(model_class.get_embedding(model_class.data_loader_labeled))
    batch = []
    for j in tqdm(range(label_per_round)):
      min_dists = []
      for i in range(len(unlabel_embedding)):
        #print(unlabel_embedding[i] - label_embedding)
        #l2_dists = np.linalg.norm(unlabel_embedding[i] - label_embedding, axis=1)
        dists = get_distance(unlabel_embedding[i], label_embedding, 'euclidean_distance')  # or cosine_distance

        min_dists.append(dists.min())

      #get index of data we choose in unlabel idx array
      label_greedy = np.argsort(min_dists)[::-1][0]

      #update embedding of label and unlabel data
      label_greedy_embedding = unlabel_embedding[label_greedy]
      unlabel_embedding = np.delete(unlabel_embedding, label_greedy,0)
      label_embedding = np.append(label_embedding, [label_greedy_embedding],axis = 0) 

      #update index of label and unlabel data
      label_greedy_idx = unlabel_index[label_greedy]
      unlabel_index = np.delete(unlabel_index, label_greedy)

      #update result
      batch.append(label_greedy_idx)

    end = time.time()
    duration = end - start
    return duration, np.array(batch)

    
  
  if strategy == 'confident_coreset':

    unlabel_index = model_class.get_unlabeled_index()
    unlabel_embedding = np.array(model_class.get_embedding_unlabeled())
    label_embedding = np.array(model_class.get_embedding(model_class.data_loader_labeled))
    batch = []
    

    if add_uncertainty == False:
      unlabel_loss = get_loss('loss',model_class)
    else:
      unlabel_loss = get_loss(add_uncertainty,model_class)


    for j in tqdm(range(label_per_round)):
      min_dists = []
      for i in range(len(unlabel_embedding)):
        #print(unlabel_embedding[i] - label_embedding)
        #l2_dists = np.linalg.norm(unlabel_embedding[i] - label_embedding, axis=1)
        dists = get_distance(unlabel_embedding[i], label_embedding, 'euclidean_distance') # or cosine_distance
        min_dists.append(dists.min())

      distance = np.power(np.array(min_dists),alpha)
      uncertainty = np.power(unlabel_loss,1-alpha)
      #get index of data we choose in unlabel idx array
      label_greedy = np.argsort(distance*uncertainty)[::-1][0]

      #update embedding of label and unlabel data
      label_greedy_embedding = unlabel_embedding[label_greedy]
      unlabel_embedding = np.delete(unlabel_embedding, label_greedy,0)
      unlabel_loss = np.delete(unlabel_loss, label_greedy,1)
      label_embedding = np.append(label_embedding, [label_greedy_embedding],axis = 0) 

      #update index of label and unlabel data
      label_greedy_idx = unlabel_index[label_greedy]
      unlabel_index = np.delete(unlabel_index, label_greedy)

      #update result
      batch.append(label_greedy_idx)

    end = time.time()
    duration = end - start
    return duration, np.array(batch)

def get_loss(strategy,model_class):
  p = None
  if strategy == 'uncertain':
    p, _ = torch.max(model_class.predict_unlabeled(), 1)
    p = 1-p

  elif strategy =='margin':
    p = model_class.predict_unlabeled()
    p = -np.sort(-p, axis=1)
    p = p[:, 0] - p[:, 1]
    p = 1-p
  elif strategy =='entropy':
    p = model_class.predict_unlabeled()
    p = (-p * torch.log(p)).sum(1)
  elif strategy == 'loss':
    p = model_class.predict_unlabeled_loss().view(1,-1).squeeze()


  if p != None:
    return np.array(p)
  else:
    print('Please Enter a Valid Loss Strategy')
    return p
    
def get_distance(unlabel, label_embedding, strategy):

  if strategy == 'cosine_distance':
    return (np.sum((unlabel * label_embedding), axis=1)) / (np.linalg.norm(unlabel) * np.linalg.norm(label_embedding, axis=1))

  elif strategy == 'euclidean_distance':
    return np.linalg.norm(unlabel - label_embedding, axis=1)

  


