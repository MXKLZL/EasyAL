from skimage.metrics import structural_similarity as ssim
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy.spatial import distance
from torchvision import datasets, models
import torch.nn.functional as F
import random
import torch

def av_SSIM(images, other=None, pairs=1000):
    l = np.zeros(pairs)
    if other:
        ind_a = np.random.choice(list(range(images.shape[0])), size = pairs)
        ind_b = np.random.choice(list(range(other.shape[0])), size = pairs)
    else:
        ind_a = np.random.choice(list(range(images.shape[0])), size = pairs)
        ind_b = np.zeros(pairs, dtype=int)
        count = 0
        while count < pairs:
            ind_b[count] = np.random.choice(list(range(images.shape[0])), size = 1)[0]
            if ind_a[count] != ind_b[count]:
                count += 1
    
    for i in range(pairs):
        if other:
            l[i] = ssim(images[ind_a[i]], other[ind_b[i]], data_range=1, multichannel=True)
        else:
            l[i] = ssim(images[ind_a[i]], images[ind_b[i]], data_range=1, multichannel=True)
    
    return l.mean()

def average_embed_dis(dataset,batch_idx,model,configs,pairs = 1000):
    #prepare dataset
    dataset.set_mode(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    dataset_query = torch.utils.data.Subset(dataset, batch_idx)
    data_loader_query = torch.utils.data.DataLoader(dataset_query, batch_size = configs['batch_size'])

    # backup_layer = model.fc
    # model.fc = nn.Sequential()
    

    preds = None
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader_query:
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            output = F.softmax(output, dim=1)
            output = output.cpu()
            if preds is not None:
                preds = torch.cat((preds, output))
            else:
                preds = output

    # model.fc = backup_layer
    
    embed_dis = []

    for i in range(pairs):
        cur_pair = random.sample(range(0, len(batch_idx)),2)
        embed1 = preds[cur_pair[0]]
        embed2 = preds[cur_pair[1]]

        embed_dis.append(distance.euclidean(embed1.numpy(), embed2.numpy()))
    
    return sum(embed_dis)/len(embed_dis)



def classification_evaluation(pred, test_target, strategy, search_category):
    if strategy == 'precision':
        return metrics.precision_score(test_target, pred, average=search_category)

    if strategy == 'recall':
        return metrics.recall_score(test_target, pred, average=search_category)

    if strategy == 'f1':
        return metrics.f1_score(test_target, pred, average=search_category)

def plot_result(evaluation, strategies, index):
  plt.figure(figsize=(15, 10))
  for i in range(len(evaluation)):
    plt.plot(np.arange(len(evaluation[i])), evaluation[i], marker='o', linestyle='dashed',linewidth=1, markersize=5, label = strategies[i])

  x_major_locator = MultipleLocator(1)
  ax = plt.gca()
  ax.xaxis.set_major_locator(x_major_locator)
  plt.xlabel('round', fontsize=14)
  plt.ylabel(index, fontsize=14)
  plt.legend()
  plt.title(index + ' under different active learning strategies')


def get_auc(evaluation, strategies):
  auc = []
  for eachs in range(len(strategies)):
    roc = 0
    if len(evaluation[eachs]) < 2:
        auc.append(None)
    for i in range(len(evaluation[eachs]) - 1):
      roc += (evaluation[eachs][i] + evaluation[eachs][i + 1])
    auc.append(roc / (2 * len(evaluation[eachs]) - 1))

  return auc

