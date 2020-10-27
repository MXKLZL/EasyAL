from skimage.metrics import structural_similarity as ssim
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

def average_embed_dis(dataset,batch_idx,configs,model = None,pairs = 1000):
    #prepare dataset
    dataset.set_mode(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    dataset_query = torch.utils.data.Subset(dataset, batch_idx)
    data_loader_query = torch.utils.data.DataLoader(dataset_query, batch_size = configs['batch_size'])

    # backup_layer = model.fc
    # model.fc = nn.Sequential()
    
    
    preds = None

    if isinstance(model, BaseModel):
        preds = model.get_embedding(data_loader_query)

    else:
        #prepare model
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(dataset.classes))
        backup_layer = model.fc
        model.fc = nn.Sequential()
        model = model.to(device)


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

        model.fc = backup_layer
    
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

def get_confusion_matrix(pred, test_target, labels = None):
    if labels is None:
        labels = list(range(25))
    cf_m = metrics.confusion_matrix(test_target, pred, labels=labels)
    plot_confusion_matrix(cf_m, target_names=labels, normalize=True)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, nan=0.0)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()