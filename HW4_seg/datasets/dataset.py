import torch
import torch.nn as nn
from torchvision import transforms
import os
import cv2
from glob import glob
from PIL import Image
import tiffile
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset, Dataset

# Split datasets for KFold (For Each patient ID)
class Split_Datasets():
    def __init__(self, path):
        super(Split_Datasets, self).__init__()
        self.image_list = glob(path + "/**/*.tif", recursive=True)
        self.id_list, self.group = self.split(self.image_list)     
        
    def __getitem__(self, index):
        return self.group[index]
    
    def __len__(self):
        return len(self.id_list)
    
    def read_id(self, text):
        parts = text.split("_")
        return parts[-2]

    def split(self, lists):
        ID_list = []
        for text in lists:
            if text.split("_")[-1] != "mask.tif":
                ID = text.split("_")[-2]
                if ID not in ID_list:
                    ID_list.append(ID)
        group = {}
        encoder = LabelEncoder()
        encoded_id = encoder.fit_transform(ID_list)

        for ID in encoded_id:
            group[ID] = []
        for text in lists:
            for key, _ in group.items():
                if self.read_id(text) == encoder.inverse_transform(encoded_id)[key]:
                    group[key].append(text)
        return ID_list, group

class Dataset():
    def __init__(self, path, mode, transform):
        super(Dataset, self).__init__()
        if mode == 'train' or mode == 'valid':
            self.file_list = path  
            self.mask_list = self.make_mask_list(self.file_list)
            # print(self.file_list)
        else:
            self.test_name = []
            self.file_list = glob(path + "/*.tif", recursive=True)
            for file in self.file_list:
                parts = file.split("/")
                name = parts[-1].split(".")[0] + "_mask.tif"
                self.test_name.append(name)

        self.transform = transform
        self.mode = mode
        
    def __getitem__(self, index):
        image_file_path = self.file_list[index]
        image = tiffile.imread(image_file_path)
        image = Image.fromarray(image)
        image = self.transform(image)
            
        if self.transform and self.mode == 'train':
            gt_file_path = self.mask_list[index]
            gt = tiffile.imread(gt_file_path)
            gt = Image.fromarray(gt)
            gt = self.transform(gt)
            return image, gt
        else:
            return image, self.test_name[index]
    
    def __len__(self):
        return len(self.file_list)

    # Make mask image path lists from input image path lists
    def make_mask_list(self, file_list):
        new_mask_path_list = []
        for file in file_list:
            parts = file.split("_")
            last_part = parts[-1]
            number = last_part.split(".")[0]  
            new_last_part = f"{int(number)}_mask.tif"
            mask_path = ""
            for part in parts[:-1]:
                mask_path = mask_path + part + "_"
            new_mask_path = mask_path + new_last_part
            new_mask_path_list.append(new_mask_path)
        return new_mask_path_list
    
