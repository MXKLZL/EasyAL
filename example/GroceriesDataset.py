from os import path
from PIL import Image
from MultiTransformDataset import *
import os

class GroceriesDataset(MultiTransformDataset):

    def __init__(self, image_path_file, root_dir, class_name_map, transform=None):
        
        with open(os.path.join(image_path_file),'r') as fh:
            path_list, target_list = zip(*[(line.split(' ')[0], line.split(' ')[1].strip()) for line in fh])

        target_list = list(target_list)
        classes = list(class_name_map.keys())
        for i in range(len(target_list)):
            target_list[i] = class_name_map[target_list[i]]
        super().__init__(path_list, target_list, classes, class_name_map, root_dir, transform)