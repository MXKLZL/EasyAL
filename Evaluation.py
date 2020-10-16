
from skimage.metrics import structural_similarity as ssim
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

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


def classification_evaluation(pred, test_target, strategy, search_category):
    if strategy == 'precision':
        return metrics.precision_score(pred, test_target, average=search_category)

    if strategy == 'recall':
        return metrics.recall_score(pred, test_target, average=search_category)

    if strategy == 'f1':
        return metrics.f1_score(pred, test_target, average=search_category)

def plot_result(evaluation, strategies, index):
  for i in range(len(evaluation)):
    plt.plot(np.arange(len(evaluation[i])), evaluation[i], marker='o', linestyle='dashed',linewidth=1, markersize=5, label = strategies[i])

  x_major_locator = MultipleLocator(1)
  ax = plt.gca()
  ax.xaxis.set_major_locator(x_major_locator)
  plt.xlabel('epoch', fontsize=14)
  plt.ylabel(index, fontsize=14)
  plt.legend()
  plt.title(index + ' under different active learning strategies')
  plt.show()