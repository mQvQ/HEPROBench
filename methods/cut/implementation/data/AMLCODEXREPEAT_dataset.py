import os
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import random
import pyvips
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms


def get_augmentations(width, height, return_nuclei=False, training=True):
    '''
    miphei-orion: infer: 333x333 -> centercrop 256x256
    '''
    additional_targets = {'image_target': 'image'}
    if return_nuclei:
        additional_targets["nuclei"] = "image"
    if training:
        spatial_augmentations = A.Compose([
            A.RandomCrop(width=width, height=height),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.CoarseDropout(p=0.1, num_holes_range=[1, 1], hole_height_range=[0., 0.3], hole_width_range=[0., 0.3])
        ], additional_targets=additional_targets)

        # color_augmentations = A.Compose([
        #     HedColorAugmentor(thresh=0.015, p=0.25),
        #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), ##
        #     A.GaussianBlur(blur_limit=(7, 7), sigma_limit=(0.1, 1.5), p=0.1),
        #     A.GaussNoise(std_range=(0.05, 0.1), p=0.1)
        # ])
        color_augmentations = None
    else:
        spatial_augmentations = A.Compose([
            A.CenterCrop(width=width, height=height),
        ], additional_targets=additional_targets)

        color_augmentations = None

    return spatial_augmentations, color_augmentations


def get_orion_transform(opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    transform_list.append(transforms.CenterCrop(opt.crop_size))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def get_smumihc_transform(opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    transform_list.append(transforms.CenterCrop(opt.crop_size))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class AMLCODEXREPEATDataset(BaseDataset):
    """
    aligned dataset for crc-orion
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.panel_key = opt.panel_key  # 'panel-1 or panel-2'
        if opt.phase == "train":
            self.df = pd.read_csv(os.path.join(opt.dataroot, f"train_filter_dapi_std_th_11_patch_meta.csv"))
        elif opt.phase == "valid":
            self.df = pd.read_csv(os.path.join(opt.dataroot, f"val_filter_dapi_std_th_11_patch_meta.csv"))
        elif opt.phase == 'infer':
            self.df = pd.read_csv(os.path.join(opt.dataroot, f"test_filter_dapi_std_th_11_patch_meta.csv"))
        elif opt.phase == 'infer_valid_test':
            df1 = pd.read_csv(os.path.join(opt.dataroot, f"test_filter_dapi_std_th_11_patch_meta.csv"))
            df2 = pd.read_csv(os.path.join(opt.dataroot, f"val_filter_dapi_std_th_11_patch_meta.csv"))
            self.df = pd.concat([df1, df2])
        else:
            raise ValueError(f"Invalid split: {opt.phase}")
        self.dataset_folder = opt.dataroot
        self.A_size = len(self.df)  # get the size of dataset A
        self.B_size = len(self.df)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_smumihc_transform(opt, grayscale=(input_nc == 1))
        self.transform_B = get_smumihc_transform(opt, grayscale=(output_nc != 3))
        print(self.transform_A, self.transform_B)

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = os.path.join(self.dataset_folder,
                              self.df.iloc[index]['image_path'])
        B_path = os.path.join(self.dataset_folder,
                              self.df.iloc[index]['target_path'])

        A_img = Image.open(A_path).convert('RGB')
        # b: .npy
        B_img = np.load(B_path)  # hwc

        # apply image transformation
        A = self.transform_A(A_img)
        B = []
        for i in range(B_img.shape[-1]):
            frame = Image.fromarray(B_img[:, :, i])
            B.append(self.transform_B(frame))
        B = torch.stack(B).squeeze(1)  # chw
        # repeat A: 3 channel -> 54 channels
        A = torch.cat([A] * 18, dim=0)
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

