import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.text import TextPath
from random import sample
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy.spatial import distance
from BaseModel import BaseModel
from torchvision import datasets, models
import torch.nn.functional as F
import torch.nn as nn
import random
import torch
import itertools

def tsne_vis_each_iter(train_ds, Model, strategy_queries):
    num_iter = len(list(strategy_queries.values())[0])

    for iter_ in range(num_iter):
        strategy_queries_iter = {}
        for s, qs in strategy_queries.items():
            strategy_queries_iter[s] = {}
            for i, q in qs.items():
                if i == iter_:
                    strategy_queries_iter[s][i] = q
        
        tsne_vis(train_ds, Model, strategy_queries_iter, f"iteration_{iter_}")

def tsne_vis(train_ds, Model, strategy_queries, name):
    full_idx = np.arange(len(train_ds))
    dataset_query = torch.utils.data.Subset(train_ds, full_idx)
    data_loader_query = torch.utils.data.DataLoader(dataset_query, batch_size = 32)
    embeddings = Model.get_embedding(data_loader_query)

    colors = sample(list(mcolors.cnames.keys()), len(strategy_queries))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    color_map = {}
    for idx, s in enumerate(strategy_queries):
        color_map[s] = colors[idx]

    tsne = TSNE(n_components=2).fit_transform(embeddings.numpy())

    plt.figure(figsize=(20, 20), dpi=80)
    ax = plt.gca()
    for s, qs in strategy_queries.items():
        for i, q in qs.items():
            for p in q:
                x, y = tsne[p][0], tsne[p][1]
                plt.scatter(x, y, marker=TextPath((0, 0), str(i)), s=250, color=color_map[s])
    h = []
    for idx, s in enumerate(strategy_queries):
        h.append(mpatches.Patch(color=color_map[s], label=s))

    plt.legend(handles=h)
    plt.savefig(f"{name}.jpg")


def vis(query_each_round, dataset, class_name_map, strategy, random_sample=10):
    num_rounds = len(query_each_round)
    query_round_by_class = {}
    for k, v in class_name_map.items():
        query_round_by_class[v] = defaultdict(list)

    for idx, query in query_each_round.items():

        for x in query:
            img, label = dataset[x]
            img = img  / 2 + 0.5
            query_round_by_class[label][idx].append(np.transpose(img.numpy(), (1, 2, 0)))
    
    for k, v in class_name_map.items():
        f, axarr = plt.subplots(num_rounds, random_sample, figsize=(10, 4)) 
        plt.rcParams.update({'figure.max_open_warning': 0})
        for round_idx in range(num_rounds):
            samples = random.sample(query_round_by_class[v][round_idx], min(random_sample, len(query_round_by_class[v][round_idx]))) 
            for sample_idx, sample in enumerate(samples):
                ax = axarr[round_idx, sample_idx]
                ax.imshow(sample)
                # ax.set_title(f"{round_idx}")
        for ax in axarr.reshape(-1):
            ax.axis('off')
        plt.ylabel(f"rounds")
        plt.xlabel(f"samples from {k}")
        plt.axis('on')
        plt.xlim((1, random_sample))
        plt.ylim((num_rounds, 1))
        xticks = plt.gca().xaxis.get_major_ticks()
        for item in xticks:
            item.label1.set_visible(False)
        # xticks[0].label1.set_visible(True)
        # xticks[-1].label1.set_visible(True)
        yticks = plt.gca().yaxis.get_major_ticks()
        for item in yticks:
            item.label1.set_visible(False)
        yticks[0].label1.set_visible(True)
        yticks[-1].label1.set_visible(True)
        plt.gca().xaxis.set_ticks_position('none') 
        plt.gca().yaxis.set_ticks_position('none') 
        plt.savefig(f"visualization/{strategy}_class_{k}.jpg", bbox_inches='tight')