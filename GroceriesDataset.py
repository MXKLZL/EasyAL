from PIL import Image
import torch
from torch.utils.data import Dataset
import os

class GroceriesDataset(Dataset):

    def __init__(self, image_path_file, root_dir, class_name_map, transform=None):
        """
        Args:
            image_path_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        with open(os.path.join(image_path_file),'r') as fh:
            path_list, target_list = zip(*[(line.split(' ')[0], line.split(' ')[1].strip()) for line in fh])

        self.path_list = path_list
        self.target_list = list(target_list)
        self.classes = list(class_name_map.keys())
        for i in range(len(target_list)):
            self.target_list[i] = class_name_map[self.target_list[i]]
        self.class_name_map = class_name_map
        self.root_dir = root_dir
        self.transform = transform
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