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
import torch.nn.functional as F
import torch.nn as nn
import random
import torch
import itertools

############## Code of integrating with label studio ##################
import os
import json

def get_mapping(ds):

    idx2base = []
    base2idx = {}
    for idx, path in enumerate(ds.path_list):
        image_path = os.path.join(ds.root_dir, path)
        base_name = os.path.basename(image_path)
        idx2base.append(base_name)
        base2idx[base_name] = idx
    return idx2base, base2idx

def read_from_oracle(completion_path, idx2base, base2idx):

    res = []
    for filename in os.listdir(completion_path):
        if filename.endswith(".json"): 
            json_path = os.path.join(completion_path, filename)
        
            with open(json_path) as f:
                data = json.load(f)
            choice = data['completions'][0]['result'][0]['value']['choices'][0]
            base_name = os.path.basename(data['task_path'])

            res.append((base2idx[base_name], choice))
    
    index_list, target_list = zip(*res)

    #Target_list is a list of raw class name, as in label studio, need to be transformed to numbers in dataset
    return index_list, target_list

def update_json(task_json_path, query_indices, idx2base, base2idx, model, ds, class_name_map):
    target_map = {}
    for key in class_name_map:
        target_map[class_name_map[key]] = key

    with open(task_json_path) as f:
        data = json.load(f)

    dataset_query = torch.utils.data.Subset(ds, query_indices)
    data_loader_query = torch.utils.data.DataLoader(dataset_query, batch_size = 32)
    preds = model.predict(data_loader_query)  # vector of probability

    update_set = set()      # store all the task_path that need to set score to 1
    base2label = {}         # store the mapping of those task_path to its predicted label

    for idx, y in zip(query_indices, preds):
        base_name = idx2base[idx]
        update_set.add(base_name)
        base2label[base_name] = target_map[np.argmax(y).item()]
        
    for id_ in data:
        id_base_name = os.path.basename(data[id_]['task_path'])
        if id_base_name in update_set:
            data[id_]['predictions'] = [{
                'result': [{
                    'from_name': 'choice',
                    'to_name': 'image',
                    'type': 'choices',
                    'value': {
                        'choices': [
                            base2label[id_base_name]
                        ]
                    }
                }],
                'score' : 1
            }]

    json_object = json.dumps(data)

    with open(task_json_path, 'w') as json_file:
        json_file.write(json_object)

############## Code of integrating with label studio ##################


def tsne_vis_each_iter(train_ds, Model, strategy_queries, opacity=None, models=None, tsne_precompute=None):
    num_iter = len(list(strategy_queries.values())[0])
    for iter_ in range(num_iter):
        print(f"drawing for iteration {iter_}")
        strategy_queries_iter = {}
        for s, qs in strategy_queries.items():
            strategy_queries_iter[s] = {}
            # for i, q in qs.items():
            #     if i == iter_:
            strategy_queries_iter[s][iter_] = qs[iter_]
        file_name = f"iteration_{iter_}_opacity" if opacity else f"iteration_{iter_}"
        tsne_vis(train_ds, Model, strategy_queries_iter, file_name, opacity=opacity, models=models, tsne_precompute=tsne_precompute)

def tsne_vis_each_strategy(train_ds, Model, strategy_queries, strategy_name, opacity=None, tsne_precompute=None):
    strategy_queries_strategy = {}
    strategy_queries_strategy[strategy_name] = strategy_queries
    file_name = f"{strategy_name}_all_iters_opacity" if opacity else f"{strategy_name}_all_iters"
    tsne_vis(train_ds, Model, strategy_queries_strategy, file_name, iterationLegend=True, opacity=opacity, tsne_precompute=tsne_precompute)

def tsne_vis_all_seperated(train_ds, Model, strategy_queries, name, iterationLegend=False, opacity=None, models=None, tsne_precompute=None, by=4):
    # plot every 4 iterations for all strategies
    num_iter = len(list(strategy_queries.values())[0])
    curr_iter = 0
    while curr_iter < num_iter:
        strategy_queries_iter = {}
        for s, qs in strategy_queries.items():
            strategy_queries_iter[s] = {}
            for iter_ in range(curr_iter, curr_iter+by):
                if iter_ in qs:
                    strategy_queries_iter[s][iter_] = qs[iter_]
        print(f"drawing {curr_iter} to {curr_iter+by-1}")
        file_name = f"{name}_iteration_{curr_iter}_{curr_iter+by-1}_opacity" if opacity else f"{name}_iteration_{curr_iter}_{curr_iter+by-1}"
        tsne_vis(train_ds, Model, strategy_queries_iter, file_name, opacity=opacity, models=models, tsne_precompute=tsne_precompute)
        curr_iter += by

def tsne_vis(train_ds, Model, strategy_queries, name, iterationLegend=False, opacity=None, models=None, tsne_precompute=None,save_path='.'):

    if tsne_precompute is not None:
        tsne = tsne_precompute
    else:
        full_idx = np.arange(len(train_ds))
        dataset_query = torch.utils.data.Subset(train_ds, full_idx)
        data_loader_query = torch.utils.data.DataLoader(dataset_query, batch_size = 32)
        embeddings = Model.get_embedding(data_loader_query)
        tsne = TSNE(n_components=2).fit_transform(embeddings.numpy())
    
    num_iter = len(list(strategy_queries.values())[0])
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
    # colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    color_map = {}
    if not iterationLegend:
        for idx, s in enumerate(strategy_queries):
            color_map[s] = colors[idx]
    else:
        if len(colors) < num_iter:
            colors = colors + list(sample(list(mcolors.cnames.keys()), num_iter-len(colors)))
        for idx in range(num_iter):
            color_map[idx] = colors[idx]

    plt.figure(figsize=(15, 15), dpi=150)
    ax = plt.gca()
    s_name = ""
    opacities = list((np.arange(0, num_iter) + 1  ) / num_iter)[::-1]
    opacity_map = {}
    for idx, key in enumerate(list(strategy_queries.values())[0]):
        opacity_map[key] = opacities[idx]
    for s, qs in strategy_queries.items():
        if tsne_precompute is None and models:
            embeddings = models[s].get_embedding(data_loader_query)
            tsne = TSNE(n_components=2).fit_transform(embeddings.numpy())
        s_name = s
        for i, q in qs.items():
            for p in q:
                x, y = tsne[p][0], tsne[p][1]
                if not iterationLegend:
                    plt.scatter(x, y, marker=TextPath((0, 0), str(i)), 
                                s=250, color=color_map[s], alpha=opacity_map[i] if opacity is not None else None)
             
                else:
                    plt.scatter(x, y, marker=TextPath((0, 0), str(i)), 
                                s=250, color=color_map[i], alpha=opacity_map[i] if opacity is not None else None)
    h = []

    if not iterationLegend:
        for idx, s in enumerate(strategy_queries):
            h.append(mpatches.Patch(color=color_map[s], label=s))
        plt.legend(handles=h)
        plt.title("embedding tsne by strategy")
    else:
        plt.title(f"{s_name} strategy embedding tsne by iteration")
    #     for idx in range(num_iter):
    #         h.append(mpatches.Patch(color=color_map[idx], label=s))
    # plt.savefig(f"visualization/{name}.jpg")
    plt.savefig(f"{save_path}/{name}.jpg")


def vis(query_each_round, dataset, class_name_map, strategy, random_sample=10,save_path='.'):
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
        # plt.savefig(f"visualization/{strategy}_class_{k}.jpg", bbox_inches='tight')
        plt.savefig(f"{save_path}/{strategy}_class_{k}.jpg", bbox_inches='tight')