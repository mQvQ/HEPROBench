"""
Training script for H&E to multiplex protein prediction model.

This script trains a deep learning model to predict protein expression levels
from H&E images. It supports both training and evaluation modes.

Required directory structure:
ROOT_DIR/
    ├── data/                     # Contains training data
    │   └── cell_measurements.pqt # Parquet file with cell measurements
    ├── images/                   # H&E image data
    │   └── {uuid}/image.ome.zarr # Zarr formatted image files  
    ├── metadata/                 # Metadata files
    │   └── metadata_dict.pkl     # Dictionary with experiment metadata
    └── runs/                     # Training run outputs
"""

import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import wandb
from typing import Tuple, List, Dict, Optional
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from PIL import Image
import argparse
from sklearn.metrics import r2_score
import random

# Configure torch multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Default configuration constants (can be overridden by argparse)
DEFAULT_ROOT_DIR = "/path/to/project/root"
DEFAULT_BATCH_SIZE = 256 # per.gpu
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EVAL_INTERVAL = 5000
DEFAULT_NUM_WORKERS = 0
DEFAULT_PATCH_SIZE = 128
DEFAULT_OUTPUT_NC = 60
DEFAULT_CHECKPOINTS_DIR = "/path/to/benchmark/results/ROSIE/"
DEFAULT_TOTAL_ITERATION = 100000 
DEFAULT_SAVE_PER_ITERATION = 5000 
DEFAULT_SAMPLES_PER_IMAGE = 16
DEFAULT_SEED = 42
DEFAULT_PREPROCESS = "auto"  # auto|cpu|gpu


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def pad_patch(patch: np.ndarray, 
             original_size: Tuple[int, int], 
             x_center: int, 
             y_center: int, 
             patch_size: int = 128) -> np.ndarray:
    """
    Pads the given patch if its size is less than patch_size x patch_size pixels.

    Args:
        patch: NumPy array representing the patch image
        original_size: Tuple of (width, height) of the original image
        x_center: X coordinate of the center of the patch in the original image
        y_center: Y coordinate of the center of the patch in the original image
        patch_size: The target size of the patch

    Returns:
        Padded patch as a NumPy array
    """
    original_height, original_width = original_size
    current_height, current_width = patch.shape[:2]
    
    if current_height == patch_size and current_width == patch_size:
        return patch
        
    # Calculate padding needed
    pad_left = max(patch_size // 2 - x_center, 0)
    pad_right = max(x_center + patch_size // 2 - original_width, 0)
    pad_top = max(patch_size // 2 - y_center, 0)
    pad_bottom = max(y_center + patch_size // 2 - original_height, 0)

    # Apply padding
    pad_shape = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)) if patch.ndim == 3 else ((pad_top, pad_bottom), (pad_left, pad_right))
    padded_patch = np.pad(patch, pad_shape, mode='constant', constant_values=0)

    # Ensure the patch is exactly patch_size x patch_size
    padded_patch = padded_patch[:patch_size, :patch_size]

    return padded_patch

def masked_mse_loss(pred: torch.Tensor, 
                   target: torch.Tensor, 
                   mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean squared error loss with a mask.

    Args:
        pred: Predicted tensor
        target: Target tensor
        mask: Mask tensor with 1s for elements to include and 0s to exclude

    Returns:
        Loss value
    """
    mask = mask.bool()
    masked_pred = torch.masked_select(pred, mask)
    masked_target = torch.masked_select(target, mask)
    return F.mse_loss(masked_pred, masked_target, reduction='mean')

def get_model(num_outputs: Optional[int] = None,
             use_context: bool = False,
             use_mask: bool = False,
             pretrained: bool = True) -> nn.Module:
    """
    Creates and returns the model architecture.

    Args:
        num_outputs: Number of output features to predict
        use_context: Whether to use contextual features
        use_mask: Whether to use masking in the model

    Returns:
        PyTorch model instance
    """
    weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_small(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_outputs)
    return model


def _rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    # img: [B,3,H,W] in [0,1]
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    maxc, _ = img.max(dim=1)
    minc, _ = img.min(dim=1)
    v = maxc
    deltac = maxc - minc

    s = torch.where(maxc == 0, torch.zeros_like(deltac), deltac / (maxc + 1e-8))
    # Hue
    rc = (maxc - r) / (deltac + 1e-8)
    gc = (maxc - g) / (deltac + 1e-8)
    bc = (maxc - b) / (deltac + 1e-8)

    h = torch.zeros_like(maxc)
    h = torch.where((maxc == r) & (deltac != 0), (bc - gc), h)
    h = torch.where((maxc == g) & (deltac != 0), 2.0 + (rc - bc), h)
    h = torch.where((maxc == b) & (deltac != 0), 4.0 + (gc - rc), h)
    h = (h / 6.0) % 1.0

    return torch.stack((h, s, v), dim=1)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    # hsv: [B,3,H,W] with h in [0,1), s,v in [0,1]
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    i = torch.floor(h * 6.0).to(dtype=torch.int64)
    f = (h * 6.0) - i.to(dtype=h.dtype)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i_mod = i % 6
    r = torch.zeros_like(v)
    g = torch.zeros_like(v)
    b = torch.zeros_like(v)

    r = torch.where(i_mod == 0, v, r)
    g = torch.where(i_mod == 0, t, g)
    b = torch.where(i_mod == 0, p, b)

    r = torch.where(i_mod == 1, q, r)
    g = torch.where(i_mod == 1, v, g)
    b = torch.where(i_mod == 1, p, b)

    r = torch.where(i_mod == 2, p, r)
    g = torch.where(i_mod == 2, v, g)
    b = torch.where(i_mod == 2, t, b)

    r = torch.where(i_mod == 3, p, r)
    g = torch.where(i_mod == 3, q, g)
    b = torch.where(i_mod == 3, v, b)

    r = torch.where(i_mod == 4, t, r)
    g = torch.where(i_mod == 4, p, g)
    b = torch.where(i_mod == 4, v, b)

    r = torch.where(i_mod == 5, v, r)
    g = torch.where(i_mod == 5, p, g)
    b = torch.where(i_mod == 5, q, b)

    return torch.stack((r, g, b), dim=1)


def _color_jitter_batch(
    x: torch.Tensor,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
    generator: torch.Generator,
) -> torch.Tensor:
    # x: [B,3,H,W] float in [0,1]
    b = x.shape[0]
    device = x.device
    dtype = x.dtype

    if brightness > 0:
        br = (torch.rand((b, 1, 1, 1), device=device, dtype=dtype, generator=generator) * 2 - 1) * brightness
        x = x + br

    if contrast > 0:
        c = 1.0 + (torch.rand((b, 1, 1, 1), device=device, dtype=dtype, generator=generator) * 2 - 1) * contrast
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) * c + mean

    if saturation > 0:
        s = 1.0 + (torch.rand((b, 1, 1, 1), device=device, dtype=dtype, generator=generator) * 2 - 1) * saturation
        gray = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3])
        x = (x - gray) * s + gray

    if hue > 0:
        # Hue jitter via HSV
        delta = (torch.rand((b, 1, 1), device=device, dtype=dtype, generator=generator) * 2 - 1) * hue
        hsv = _rgb_to_hsv(x.clamp(0, 1))
        hsv[:, 0] = (hsv[:, 0] + delta).remainder(1.0)
        x = _hsv_to_rgb(hsv)

    return x.clamp(0, 1)


def _random_affine_batch(
    x: torch.Tensor,
    degrees: float,
    p_hflip: float,
    p_vflip: float,
    generator: torch.Generator,
) -> torch.Tensor:
    # x: [B,C,H,W]
    b, _, h, w = x.shape
    device = x.device
    dtype = x.dtype

    # Sample flips and rotation per sample
    do_hflip = torch.rand((b,), device=device, dtype=dtype, generator=generator) < p_hflip
    do_vflip = torch.rand((b,), device=device, dtype=dtype, generator=generator) < p_vflip
    sx = torch.where(do_hflip, x.new_full((b,), -1.0), x.new_full((b,), 1.0))
    sy = torch.where(do_vflip, x.new_full((b,), -1.0), x.new_full((b,), 1.0))

    if degrees > 0:
        angles = (torch.rand((b,), device=device, dtype=dtype, generator=generator) * 2 - 1) * degrees
        angles = angles * (torch.pi / 180.0)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
    else:
        cos = x.new_ones((b,))
        sin = x.new_zeros((b,))

    # R * S
    a00 = cos * sx
    a01 = -sin * sy
    a10 = sin * sx
    a11 = cos * sy

    theta = x.new_zeros((b, 2, 3))
    theta[:, 0, 0] = a00
    theta[:, 0, 1] = a01
    theta[:, 1, 0] = a10
    theta[:, 1, 1] = a11

    grid = F.affine_grid(theta, size=x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _imagenet_normalize_(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return x.sub_(mean).div_(std)


def preprocess_on_device(
    x_u8_chw: torch.Tensor,
    train: bool,
    generator: torch.Generator,
    out_size: int = 224,
    rotation_degrees: float = 10.0,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
) -> torch.Tensor:
    """
    x_u8_chw: [B,3,H,W] uint8 (0..255) or float in [0,1].
    Returns float tensor normalized for ImageNet.
    """
    if x_u8_chw.dtype == torch.uint8:
        x = x_u8_chw.to(dtype=torch.float32).div_(255.0)
    else:
        x = x_u8_chw.to(dtype=torch.float32)

    if x.shape[-1] != out_size or x.shape[-2] != out_size:
        x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)

    if train:
        x = _random_affine_batch(x, degrees=rotation_degrees, p_hflip=0.5, p_vflip=0.5, generator=generator)
        x = _color_jitter_batch(
            x,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            generator=generator,
        )

    return _imagenet_normalize_(x)


class ImageDataset(Dataset):
    """
    Dataset class for loading H&E image patches and their corresponding protein expression values.
    
    Args:
        data_df: DataFrame containing cell measurements
        root_dir: Root directory containing image data
        is_test: Whether this is a test dataset
        use_mask: Whether to use cell segmentation masks
        transform: Transforms to apply to images
        metadata_dict: Dictionary containing experiment metadata
        test_acq_ids: List of acquisition IDs to use for testing
        subset: Subset of coverslip IDs to use
        pred_only: Whether to only generate predictions (no ground truth)
    """
    def __init__(self,
                data_df: pd.DataFrame,
                root_dir: str,
                is_test: bool = False,
                use_mask: bool = False,
                transform: Optional[Dict] = None,
                metadata_dict: Optional[Dict] = None,
                test_acq_ids: Optional[List[str]] = None,
                subset: Optional[List[str]] = None,
                pred_only: bool = False,
                patch_size: int = 128,
                output_nc: int = 60,
                samples_per_image: int = 8):
        
        self.df = data_df
        self.root_dir = root_dir
        self.transform = transform
        self.patch_size = patch_size
        self.output_nc = output_nc
        self.samples_per_image = samples_per_image
        self.ps = self.patch_size//2
        self.ps_if = 4
        self.use_mask = use_mask
        self.invalid_acq_ids = set()
        self.zarr_cache = {}
        self.is_test = is_test
        self.pred_only = pred_only         
        self.df.reset_index(inplace=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Optional[Tuple]:
        """
        Get samples from a single image.
        
        Args:
            idx: Index of the item to get
            
        Returns:
            Tuple containing (patches_list, exp_values_list, masks_list)
            Each list contains samples_per_image samples from the same image
            Returns None if the item is invalid
        """
        row = self.df.iloc[idx]
        he_path = os.path.join(self.root_dir, self.df.iloc[idx]['image_path'])
        if_path = os.path.join(self.root_dir, self.df.iloc[idx]['target_path'])
        
        # Load images once
        he_img = Image.open(he_path).convert('RGB')
        he_width, he_height = he_img.size
        he_arr = np.asarray(he_img)  # HWC uint8; typically small (e.g., 256x256)

        # Target is expected to be a local .npy (HE=jpg, MIF=.npy)
        if if_path.lower().endswith(('.tif', '.tiff')):
            raise ValueError(f"TIFF targets are not supported (pyvips removed). Please convert to .npy: {if_path}")
        if_patch = np.load(if_path, mmap_mode="r")
        
        # Sample multiple points from the same image
        patches_list = []
        exp_values_list = []
        masks_list = []
        
        for _ in range(self.samples_per_image):
            # Random select a center point
            Y = np.random.randint(self.ps, if_patch.shape[0]-self.ps)
            X = np.random.randint(self.ps, if_patch.shape[1]-self.ps)
            
            # Extract 8x8 IF patch
            if_patch_sample = if_patch[Y-self.ps_if:Y+self.ps_if, X-self.ps_if:X+self.ps_if]
            
            # Prepare mask
            mask = np.ones(self.output_nc, dtype=np.float32)
            
            # Average expression value
            exp_value = (np.mean(if_patch_sample, axis=(0,1))/255.0).astype(np.float32)  # 0-1
            
            # Crop related HE patch from HWC uint8 array
            b = np.clip(Y-self.ps, 0, he_height)
            t = np.clip(Y+self.ps, 0, he_height)
            l = np.clip(X-self.ps, 0, he_width)
            r = np.clip(X+self.ps, 0, he_width)
            he_patch_cropped = he_arr[b:t, l:r]

            # Pad to exact patch_size if near edges
            he_patch_padded = pad_patch(he_patch_cropped, (he_height, he_width), X, Y, patch_size=self.patch_size)
            if he_patch_padded.shape != (self.patch_size, self.patch_size, 3):
                raise ValueError(f'H&E patch shape is {he_patch_padded.shape}, expected {(self.patch_size, self.patch_size, 3)}')

            if self.transform is None:
                # Return uint8 CHW patch; preprocessing/augmentation can happen on GPU later
                patch_u8 = torch.from_numpy(he_patch_padded).permute(2, 0, 1).contiguous()
                patches_list.append(patch_u8)
            else:
                if isinstance(self.transform, dict):
                    he_patch_pt = self.transform['all_channels'](Image.fromarray(he_patch_padded))
                    patch = self.transform['image_only'](he_patch_pt)
                else:
                    patch = self.transform(Image.fromarray(he_patch_padded))
                if patch.shape != (3, 224, 224):
                    raise ValueError(f'Patch shape is {patch.shape}, expected (3, 224, 224)')
                patches_list.append(patch)
            exp_values_list.append(exp_value)
            masks_list.append(mask)
        
        return patches_list, exp_values_list, masks_list

def evaluate(model: nn.Module,
            data_loader: DataLoader,
            device: torch.device,
            preprocess_mode: str = "cpu",
            seed: int = DEFAULT_SEED) -> float:
    """
    Evaluate the model on a dataset and return R2 score.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for the evaluation dataset
        device: Device to run evaluation on
        
    Returns:
        R2 score (float)
    """
    model.eval()
    
    all_outputs = []
    all_targets = []

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    
    with torch.no_grad():
        for inputs, exp_vec, mask in tqdm(data_loader, desc="Evaluating"):
            inputs = inputs.to(device, non_blocking=True)
            exp_vec = exp_vec.to(device)
            mask = mask.to(device)

            if preprocess_mode == "gpu":
                inputs = preprocess_on_device(inputs, train=False, generator=gen)

            outputs = model(inputs)
            
            # Apply mask if needed
            mask_bool = mask.bool()
            masked_outputs = torch.masked_select(outputs, mask_bool).detach().cpu().numpy()
            masked_targets = torch.masked_select(exp_vec, mask_bool).detach().cpu().numpy()
            
            all_outputs.append(masked_outputs)
            all_targets.append(masked_targets)
    
    # Concatenate all predictions and targets
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate R2 score
    val_r2 = r2_score(all_targets, all_outputs)
    
    return val_r2

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train H&E to multiplex protein prediction model')
    
    # Data paths
    parser.add_argument('--root_dir', type=str, default=DEFAULT_ROOT_DIR,
                        help='Base directory for project')
    parser.add_argument('--checkpoints_dir', type=str, default=DEFAULT_CHECKPOINTS_DIR,
                        help='Directory to save checkpoints')
    parser.add_argument('--panel', type=str, default=None,
                        help='Panel data name')
    parser.add_argument('--panel_dir', type=str, default=None,
                        help='Directory for panel data')
    parser.add_argument('--train_csv', type=str, default=None,
                        help='Explicit unified train CSV (image_path,target_path)')
    parser.add_argument('--val_csv', type=str, default=None,
                        help='Explicit unified validation CSV')
    
    # Model hyperparameters
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--patch_size', type=int, default=DEFAULT_PATCH_SIZE,
                        help='Patch size')
    parser.add_argument('--output_nc', type=int, default=DEFAULT_OUTPUT_NC,
                        help='Number of output channels')
    
    # Training hyperparameters
    parser.add_argument('--total_iteration', type=int, default=DEFAULT_TOTAL_ITERATION,
                        help='Total number of training iterations')
    parser.add_argument('--save_per_iteration', type=int, default=DEFAULT_SAVE_PER_ITERATION,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--eval_interval', type=int, default=DEFAULT_EVAL_INTERVAL,
                        help='Evaluate every N iterations')
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS,
                        help='Number of data loading workers')
    parser.add_argument('--samples_per_image', type=int, default=DEFAULT_SAMPLES_PER_IMAGE,
                        help='Number of samples to extract from each image')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='Random seed for training')
    parser.add_argument('--preprocess', type=str, default=DEFAULT_PREPROCESS,
                        choices=['auto', 'cpu', 'gpu'],
                        help='Where to run resize/augment/normalize (gpu reduces CPU load)')
    parser.add_argument('--device', type=str, default=None,
                        help='Explicit torch device; default is CUDA when available')
    parser.add_argument('--data_parallel', action='store_true',
                        help='Use every visible CUDA device through DataParallel')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Do not download/load ImageNet ConvNeXt initialization')
    
    # Wandb parameters
    parser.add_argument('--wandb_project', type=str, default='hande_to_codex',
                        help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default='model_training',
                        help='Wandb run name')
    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='offline',
                        help='Weights & Biases mode (offline is reproducible without login)')
    
    # Experiment name
    parser.add_argument('--name', type=str, default='experiment',
                        help='Experiment name for checkpoint directory')
    
    return parser.parse_args()

def main():
    """Main training and evaluation function."""
    
    # Parse arguments
    args = parse_args()

    # Reproducibility (avoid per-sample seeding inside Dataset)
    _seed_everything(args.seed)
    
    # Initialize wandb for experiment tracking
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args), mode=args.wandb_mode)

    # Set up data transforms
    transform_train = {
        'all_channels': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Resize(224, antialias=True),
            transforms.RandomRotation(degrees=(-10, 10)),
        ]),
        'image_only': transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }

    transform_eval = {
        'all_channels': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
        ]),
        'image_only': transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }

    if args.train_csv and args.val_csv:
        train_df = pd.read_csv(args.train_csv)
        val_df = pd.read_csv(args.val_csv)
    elif args.panel is not None:
        if not args.panel_dir:
            raise ValueError('--panel_dir is required when --panel is set')
        train_df = pd.read_csv(os.path.join(args.panel_dir, f'train_filter_dapi_std_11_inv_red_nmi_003_patch_meta_{args.panel}.csv'))
        val_df = pd.read_csv(os.path.join(args.panel_dir, f'valid_filter_dapi_std_11_inv_red_nmi_003_patch_meta_{args.panel}.csv'))
    else:
        train_df = pd.read_csv(os.path.join(args.root_dir, 'train_filter_dapi_std_11_inv_red_nmi_003_patch_meta.csv'))
        val_df = pd.read_csv(os.path.join(args.root_dir, 'valid_filter_dapi_std_11_inv_red_nmi_003_patch_meta.csv'))
    
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    preprocess_mode = args.preprocess
    if preprocess_mode == "auto":
        preprocess_mode = "gpu" if device.type == "cuda" else "cpu"

    # Create datasets
    train_dataset = ImageDataset(
        data_df=train_df,
        root_dir=args.root_dir,
        transform=None if preprocess_mode == "gpu" else transform_train,
        patch_size=args.patch_size,
        output_nc=args.output_nc,
        samples_per_image=args.samples_per_image,
    )

    val_dataset = ImageDataset(
        data_df=val_df,
        root_dir=args.root_dir,
        transform=None if preprocess_mode == "gpu" else transform_eval,
        is_test=True,
        patch_size=args.patch_size,
        output_nc=args.output_nc,
        samples_per_image=args.samples_per_image,
    )

    # Create data loaders
    def collate_fn(batch):
        """
        Collate function that handles multiple samples per image.
        
        Args:
            batch: List of tuples, each tuple contains (patches_list, exp_values_list, masks_list)
                  where each list has samples_per_image items
        
        Returns:
            Batched tensors: (inputs, labels, mask)
            - inputs: (batch_size * samples_per_image, 3, 224, 224)
            - labels: (batch_size * samples_per_image, output_nc)
            - mask: (batch_size * samples_per_image, output_nc)
        """
        batch = list(filter(lambda x: x is not None, batch))
        
        # Flatten: from N images × samples_per_image → (N * samples_per_image) samples
        all_patches = []
        all_exp_values = []
        all_masks = []
        
        for patches_list, exp_values_list, masks_list in batch:
            all_patches.extend(patches_list)
            all_exp_values.extend(exp_values_list)
            all_masks.extend(masks_list)
        
        # Use default_collate to batch the flattened samples
        return torch.utils.data.default_collate([
            (p, e, m) for p, e, m in zip(all_patches, all_exp_values, all_masks)
        ])

    train_loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
    )
    if args.num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)

    val_loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=_seed_worker,
    )
    if args.num_workers > 0:
        val_loader_kwargs["prefetch_factor"] = 2

    val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    # Set up model and training
    model = get_model(num_outputs=args.output_nc, pretrained=not args.no_pretrained)
    
    if args.data_parallel and device.type == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = masked_mse_loss

    # Create checkpoints directory with experiment name
    checkpoint_dir = os.path.join(args.checkpoints_dir, args.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Print training configuration
    print(f'\n{"="*60}')
    print(f'Training Configuration:')
    print(f'  Experiment name: {args.name}')
    print(f'  Checkpoint directory: {checkpoint_dir}')
    print(f'  Total iterations: {args.total_iteration}')
    print(f'  Batch size: {args.batch_size}')
    print(f'  Learning rate: {args.learning_rate}')
    print(f'  Eval interval: {args.eval_interval}')
    print(f'  Save interval: {args.save_per_iteration}')
    print(f'  Output channels: {args.output_nc}')
    print(f'  Samples per image: {args.samples_per_image}')
    print(f'  Effective batch size: {args.batch_size * args.samples_per_image}')
    print(f'{"="*60}\n')

    # Training loop
    iteration = 0
    best_val_r2 = float('-inf')
    best_iteration = 0
    current_val_r2 = None  # Track current val_r2
    
    # Create dataset iterator for continuous training
    train_iter = iter(train_loader)
    print('Starting training...\n')

    aug_gen = torch.Generator(device=device)
    aug_gen.manual_seed(args.seed + 12345)

    while iteration < args.total_iteration:
        model.train()
        
        # Get next batch (restart iterator if exhausted)
        try:
            inputs, labels, mask = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, labels, mask = next(train_iter)
        
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device).float()  # Ensure float32
        mask = mask.to(device).float()  # Ensure float32

        if preprocess_mode == "gpu":
            inputs = preprocess_on_device(inputs, train=True, generator=aug_gen)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels, mask)
        
        if torch.isnan(loss):
            print("Warning: Loss is NaN, skipping batch")
            iteration += 1
            continue
            
        loss.backward()
        optimizer.step()

        # Logging: print every 1k iterations
        if iteration % 1000 == 0:
            progress = (iteration / args.total_iteration) * 100
            print(f'[Iter {iteration}/{args.total_iteration}] ({progress:.2f}%) | '
                  f'Loss: {loss.item():.6f} | LR: {optimizer.param_groups[0]["lr"]:.2e}')
            wandb.log({'train_loss': loss.item(), 'iteration': iteration})

        # Validation
        if iteration % args.eval_interval == 0 and iteration > 0:
            print(f'[Iter {iteration}/{args.total_iteration}] Running validation...')
            current_val_r2 = evaluate(model, val_loader, device, preprocess_mode=preprocess_mode, seed=args.seed)
            
            progress = (iteration / args.total_iteration) * 100
            print(f'[Iter {iteration}/{args.total_iteration}] ({progress:.2f}%) | '
                  f'Val R2: {current_val_r2:.6f} | Best R2: {best_val_r2:.6f} (iter {best_iteration})')
            
            wandb.log({
                'val_r2': current_val_r2,
                'iteration': iteration
            })

            # Save best model based on val_r2
            if current_val_r2 > best_val_r2:
                best_val_r2 = current_val_r2
                best_iteration = iteration
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_r2': current_val_r2,
                }, best_model_path)
                print(f'✓ New best model saved at iteration {iteration} with val_r2: {current_val_r2:.6f}')

            scheduler.step(current_val_r2)

        # Save checkpoint periodically
        if iteration % args.save_per_iteration == 0 and iteration > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration}.pth')
            checkpoint_dict = {
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if current_val_r2 is not None:
                checkpoint_dict['val_r2'] = current_val_r2
            torch.save(checkpoint_dict, checkpoint_path)
            print(f'✓ Checkpoint saved at iteration {iteration} -> {checkpoint_path}')

        iteration += 1

    print(f'\n{"="*60}')
    print(f'Training completed!')
    print(f'Total iterations: {iteration}')
    latest_path = os.path.join(checkpoint_dir, 'latest_model.pth')
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, latest_path)
    if current_val_r2 is None:
        print('Validation was not scheduled during this short run.')
    else:
        print(f'Best model at iteration {best_iteration} with val_r2: {best_val_r2:.6f}')
        print(f'Best model saved at: {os.path.join(checkpoint_dir, "best_model.pth")}')
    print(f'Latest model saved at: {latest_path}')
    print(f'{"="*60}')
    wandb.finish()

if __name__ == '__main__':
    main()
