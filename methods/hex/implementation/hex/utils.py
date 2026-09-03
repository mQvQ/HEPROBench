
import os
import random
import torch
import torch.nn as nn
import numpy as np
from os.path import join
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import logging
import pandas as pd

def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class PatchDataset(Dataset):
    def __init__(self, csv,label_columns, transform=None):
        self.images = csv['images'].values
        self.labels = csv[label_columns].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx, :]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label, image_path


class CustomDataset(Dataset):

    def __init__(self, dataroot, phase, panel_key, transform=None, csv_path=None):
        self.dataroot = dataroot
        self.phase = phase
        self.panel_key = panel_key
        if csv_path:
            self.df = pd.read_csv(csv_path)
        elif self.panel_key == 'panel-1' or self.panel_key == 'panel-2':
            if self.phase == "train":
                self.df = pd.read_csv(os.path.join(self.dataroot, f"train_filter_dapi_std_11_inv_red_nmi_003_patch_meta_{self.panel_key}.csv"))
            elif self.phase == "valid":
                self.df = pd.read_csv(os.path.join(self.dataroot, f"valid_filter_dapi_std_11_inv_red_nmi_003_patch_meta_{self.panel_key}.csv"))
            elif self.phase == "test":
                self.df = pd.read_csv(os.path.join(self.dataroot, f"test_filter_dapi_std_11_inv_red_nmi_003_patch_meta_{self.panel_key}.csv"))
            elif self.phase == "infer_valid_test":
                df1 = pd.read_csv(os.path.join(self.dataroot, f"test_filter_dapi_std_11_inv_red_nmi_003_patch_meta_{self.panel_key}.csv"))
                df2 = pd.read_csv(os.path.join(self.dataroot, f"valid_filter_dapi_std_11_inv_red_nmi_003_patch_meta_{self.panel_key}.csv"))
                self.df = pd.concat([df1, df2])
        else:
            if self.phase == "train":
                self.df = pd.read_csv(os.path.join(self.dataroot, f"train_filter_dapi_std_11_inv_red_nmi_003_patch_meta.csv"))
            elif self.phase == "valid":
                self.df = pd.read_csv(os.path.join(self.dataroot, f"valid_filter_dapi_std_11_inv_red_nmi_003_patch_meta.csv"))
            elif self.phase == "test":
                self.df = pd.read_csv(os.path.join(self.dataroot, f"test_filter_dapi_std_11_inv_red_nmi_003_patch_meta.csv"))
            elif self.phase == "infer_valid_test":
                df1 = pd.read_csv(os.path.join(self.dataroot, f"test_filter_dapi_std_11_inv_red_nmi_003_patch_meta.csv"))
                df2 = pd.read_csv(os.path.join(self.dataroot, f"valid_filter_dapi_std_11_inv_red_nmi_003_patch_meta.csv"))
                self.df = pd.concat([df1, df2])

        
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.panel_key == 'panel-1' or self.panel_key == 'panel-2':
            he_path = os.path.join(self.dataroot, 'Series-14-After-Registration-High-Quality-Region-Cropped-Patch', self.df.iloc[idx]['image_path'])
            if_path = os.path.join(self.dataroot, 'Series-14-After-Registration-High-Quality-Region-Cropped-Patch', self.df.iloc[idx]['target_path'])
        else:
            he_path = os.path.join(self.dataroot, self.df.iloc[idx]['image_path'])
            if_path = os.path.join(self.dataroot, self.df.iloc[idx]['target_path'])

        he_img = Image.open(he_path)
        if self.transform:
            he_img = self.transform(he_img)

        if_img = np.load(if_path)
        # crop 14*14 ceter region and calc average value from if_img
        if_width, if_height = if_img.shape[0], if_img.shape[1]
        x = if_width // 2
        y = if_height // 2
        if_patch = if_img[x-7:x+7, y-7:y+7, :]
        exp_value = np.mean(if_patch, axis=(0, 1))

        return he_img, exp_value, he_path, if_path



def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)

    print("\nTrainable parameters:")
    for name, param in net.named_parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
            print(f"{name}, Shape: {param.shape}")

    print('\nTotal number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)
