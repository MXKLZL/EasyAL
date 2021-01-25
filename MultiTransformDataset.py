from PIL import Image
import torch
from torch.utils.data import Dataset
import os



class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

class MultiTransformDataset(Dataset):

    def __init__(self, path_list, target_list=None, classes=None, class_name_map=None, root_dir='', transform=None):
        """
        Args:
            path_list (list of string): list of path to image files
            target_list (list of string): list targets of samples
            root_dir (string): root directory to paths
            classes (list of string): class names corresponding to targets
            class_name_map(dictionary): mapping between targets and classes
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """


        self.path_list = path_list
        if target_list is not None:
            self.target_list = target_list
        else:
            self.target_list = [-1 for i in path_list]
        self.classes = classes
        self.root_dir = root_dir
        self.transform = transform
        self.class_name_map = class_name_map
        self.mode = 0

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.root_dir, self.path_list[idx])
        image = Image.open(image_path)
        if self.transform:
            image = self.transform[self.mode](image)

        return image, self.target_list[idx]

    def set_mode(self, mode):
        self.mode = mode

    def update_target(self, index, new_target):
        for i in range(len(index)):
            self.target_list[index[i]] = new_target[i]