import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random

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