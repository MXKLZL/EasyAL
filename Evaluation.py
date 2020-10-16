
from skimage.metrics import structural_similarity as ssim
import numpy as np

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