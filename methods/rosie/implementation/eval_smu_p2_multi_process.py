"""
Evaluation script for H&E to multiplex protein prediction model (Optimized).
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import tifffile
import cv2
from tqdm import tqdm
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from skimage.morphology import dilation, disk
from typing import Tuple, Optional
import torch.multiprocessing as mp
import warnings

# 忽略一些不必要的警告
warnings.filterwarnings("ignore")

# Configuration constants
# 注意：在多图并行模式下，单个Loader的worker数建议减少，以免线程爆炸
BATCH_SIZE = 512  # 增大Batch Size可以利用GPU并行性
PATCH_SIZE = 128
WHITE_THRESHOLD = 220


def pad_patch(patch: np.ndarray,
              original_size: Tuple[int, int],
              x_center: int,
              y_center: int,
              patch_size: int = PATCH_SIZE) -> np.ndarray:
    original_height, original_width = original_size
    current_height, current_width = patch.shape[:2]

    if current_height == patch_size and current_width == patch_size:
        return patch

    pad_left = max(patch_size // 2 - x_center, 0)
    pad_right = max(x_center + patch_size // 2 - original_width, 0)
    pad_top = max(patch_size // 2 - y_center, 0)
    pad_bottom = max(y_center + patch_size // 2 - original_height, 0)

    pad_shape = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)) if patch.ndim == 3 else (
        (pad_top, pad_bottom), (pad_left, pad_right))
    padded_patch = np.pad(patch, pad_shape, mode='constant', constant_values=0)
    return padded_patch[:patch_size, :patch_size]


def normalize_image(image, min_value, max_value):
    return ((image - min_value) * 255. / (max_value - min_value + 1e-8)).astype(np.uint8)


def get_model(num_outputs: int) -> nn.Module:
    model = models.convnext_small(weights='IMAGENET1K_V1')
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_outputs)
    return model


class ImageDataset(Dataset):
    def __init__(self, image_path: str, transform: Optional[dict] = None, stride_size: int = 8,
                 exclude_background: bool = True):
        self.image_path = image_path
        self.transform = transform
        self.patch_size = PATCH_SIZE
        self.ps = self.patch_size // 2
        self.stride_size = stride_size
        self.center_half = max(1, stride_size // 2)

        # 优化：ZARR 读取
        # 如果 ZARR 很大，这里 .compute() 会吃掉大量内存。
        # 建议生产环境中使用 dask 延迟加载，但为了保持逻辑一致性，此处保留逻辑，
        # 但要注意多进程时的内存消耗。
        if image_path.endswith('.zarr'):
            reader = Reader(parse_url(image_path, mode="r"))
            node = list(reader())[0]
            # 仅读取需要的层级，通常 level 0
            dask_data = node.data[0]
            self.he_zarr = [dask_data[i].compute() for i in range(3)]  # 依然强制加载到内存

            height, width = self.he_zarr[0].shape

            # Crop logic (保留原逻辑)
            center_y, center_x = height // 2, width // 2
            crop_size = 100  # 原逻辑只切了中间100x100? 这看起来像是个Bug或者测试代码，
            # 如果是全片预测，这里逻辑可能有误。但我暂时保留你的逻辑。
            # 修正：通常预测是全图，原代码这里的crop_size=100会导致只预测中心极小区域。
            # 如果你是做全片预测，请注释掉下面的 cropping 逻辑。
            # --- Cropping Logic Start ---
            # half_crop = crop_size // 2
            # y_start = max(0, center_y - half_crop)
            # y_end = min(height, center_y + half_crop)
            # x_start = max(0, center_x - half_crop)
            # x_end = min(width, center_x + half_crop)
            # self.he_zarr = [channel[y_start:y_end, x_start:x_end] for channel in self.he_zarr]
            # --- Cropping Logic End ---

            # 重新获取 shape (因为可能被 crop 了)
            height, width = self.he_zarr[0].shape

        else:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.he_zarr = [img[:, :, i] for i in range(3)]
            height, width = img.shape[:2]

        # 优化坐标生成：使用 numpy 向量化操作代替双重 for 循环 (可选，这里不是主要瓶颈)
        self.coords = []
        for y in range(0, height, stride_size):
            for x in range(0, width, stride_size):
                if exclude_background:
                    x_start = max(0, x - self.center_half)
                    x_end = min(width, x + self.center_half)
                    y_start = max(0, y - self.center_half)
                    y_end = min(height, y + self.center_half)

                    # 快速检查，避免全通道 mean
                    # 通常 Green 通道 (idx 1) 对背景判断足够
                    center_region = self.he_zarr[1][y_start:y_end, x_start:x_end]
                    if np.mean(center_region) < WHITE_THRESHOLD:
                        self.coords.append((x, y))
                else:
                    self.coords.append((x, y))

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        X, Y = self.coords[idx]

        # 边界检查
        h, w = self.he_zarr[0].shape
        b = np.clip(Y - self.ps, 0, h)
        t = np.clip(Y + self.ps, 0, h)
        l = np.clip(X - self.ps, 0, w)
        r = np.clip(X + self.ps, 0, w)

        he_patch = np.stack([channel[b:t, l:r] for channel in self.he_zarr], axis=-1)
        he_patch = pad_patch(he_patch, (h, w), X, Y)

        if isinstance(self.transform, dict):
            he_patch_pt = self.transform['all_channels'](he_patch)
            patch = self.transform['image_only'](he_patch_pt)
        else:
            patch = self.transform(he_patch)

        return patch, X, Y


def postprocess_predictions(predictions, tissue_mask, apply_border_threshold=False, bg_percentile=90,
                            max_percentile=99.9):
    """优化后的后处理：减少不必要的内存拷贝"""
    processed = np.zeros_like(predictions, dtype=np.uint8)

    # 预计算 border mask
    height, width = tissue_mask.shape
    pad = 50
    border_mask = np.zeros_like(tissue_mask, dtype=bool)
    border_mask[:pad, :] = True
    border_mask[-pad:, :] = True
    border_mask[:, :pad] = True
    border_mask[:, -pad:] = True

    # 可以考虑并行处理通道，但在Python中这里可能是瓶颈
    for i in range(predictions.shape[0]):
        channel = predictions[i]

        if apply_border_threshold:
            bg_values = channel[border_mask]
            bg_threshold = np.percentile(bg_values, bg_percentile) if bg_values.size > 0 else 0
        else:
            bg_threshold = np.percentile(channel, bg_percentile)

        vmax = np.percentile(channel, max_percentile)
        if vmax <= bg_threshold: bg_threshold = 0

        clipped = np.clip(channel, bg_threshold, vmax)
        normalized = normalize_image(clipped, bg_threshold, vmax)

        # Box blur 使用 scipy 的 convolve2d 比较慢，可以用 cv2 加速
        blurred = cv2.blur(normalized, (8, 8))  # 替换 box_blur 函数
        processed[i] = blurred

    return processed


def worker_inference(rank, image_list, args, model_state_dict, device_id):
    """
    单个进程的工作函数。
    rank: 进程ID
    image_list: 该进程需要处理的图片列表
    args: 命令行参数
    model_state_dict: 模型权重
    device_id: 指定 GPU ID
    """
    # 每个进程初始化自己的 Device
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    num_channels = 50
    model = get_model(num_outputs=num_channels)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    transform = {
        'all_channels': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
        ]),
        'image_only': transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }

    # Gaussian kernel 预计算
    stride_size = args.stride_size
    kernel_size = stride_size * 2
    y, x = np.mgrid[0:kernel_size, 0:kernel_size]
    center = kernel_size // 2
    weight_kernel = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * (kernel_size / 4) ** 2))
    # 转为 Tensor 方便 GPU 计算 (可选，目前瓶颈在 stitching 的 CPU 端，保持 numpy 即可)

    for idx, (img_name, output_name) in enumerate(image_list):
        try:
            full_image_path = os.path.join(args.input_dir, img_name)
            output_path = os.path.join(args.output_dir, f'{output_name}.tiff')

            if os.path.exists(output_path):
                print(f"[Process {rank}] Skipping {output_name}, already exists.")
                continue

            print(f"[Process {rank}] Processing {img_name} on {device}...")

            # 减少 workers 数，因为我们已经有多个进程在跑了
            overlap_stride = max(1, stride_size // 2)
            dataset = ImageDataset(full_image_path, transform=transform, stride_size=overlap_stride,
                                   exclude_background=args.exclude_background)

            # num_workers 设置为 2 或 4，不要太高，否则系统负载过重
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

            height, width = dataset.he_zarr[0].shape

            # 使用 float32 累加，最后再转
            raw_output = np.zeros((num_channels, height, width), dtype=np.float32)
            weight_map = np.zeros((height, width), dtype=np.float32)

            with torch.no_grad():
                for patches, X, Y in tqdm(dataloader, position=rank, desc=f"P{rank}-{output_name}", leave=False):
                    patches = patches.to(device)
                    # 批量预测
                    preds = model(patches).cpu().numpy()  # (B, C)

                    # --- 核心优化：向量化 Stitching ---
                    # 避免在 Python 中循环 C (通道数)
                    X = X.numpy()
                    Y = Y.numpy()

                    half_size = kernel_size // 2

                    # 对 Batch 中的每一个 patch
                    for i in range(len(preds)):
                        x_c, y_c = X[i], Y[i]
                        pred_vector = preds[i]  # Shape (C,)

                        # 边界计算
                        t = max(0, y_c - half_size)
                        b = min(height, y_c + half_size)
                        l = max(0, x_c - half_size)
                        r = min(width, x_c + half_size)

                        # Kernel 裁剪
                        kt = t - (y_c - half_size)
                        kb = b - (y_c - half_size)
                        kl = l - (x_c - half_size)
                        kr = r - (x_c - half_size)

                        # 获取对应的 weight kernel 切片
                        current_weight = weight_kernel[kt:kb, kl:kr]  # (H_k, W_k)

                        # Vectorized update:
                        # pred_vector: (C,) -> (C, 1, 1)
                        # current_weight: (H_k, W_k) -> (1, H_k, W_k)
                        # 广播相乘 -> (C, H_k, W_k)
                        weighted_patch = pred_vector[:, None, None] * current_weight[None, :, :]

                        raw_output[:, t:b, l:r] += weighted_patch
                        weight_map[t:b, l:r] += current_weight

            # Normalize
            weight_map = np.maximum(weight_map, 1e-8)
            raw_output /= weight_map
            if args.postprocess_image:
                # 重新生成 tissue mask (因为 dataset 不保存它)
                tissue_mask = np.any((np.array(dataset.he_zarr) < 220), axis=0)
                tissue_mask = dilation(tissue_mask, disk(9))

                processed_output = postprocess_predictions(raw_output, tissue_mask,
                                                           apply_border_threshold=args.apply_border_threshold)
                if args.smooth_sigma > 0:
                    # 对所有通道一次性做 gaussian filter 可能会慢，逐个做
                    for c in range(processed_output.shape[0]):
                        processed_output[c] = gaussian_filter(processed_output[c], sigma=args.smooth_sigma)

                tifffile.imwrite(output_path, processed_output, compression='zlib')
            else:
                tifffile.imwrite(output_path, raw_output.astype(np.float32), compression='zlib')

        except Exception as e:
            print(f"[Process {rank}] Error processing {img_name}: {e}")
            # 可以在这里记录到 error log
            continue


def main():
    parser = argparse.ArgumentParser(description='Run inference on H&E images (High Speed)')
    # ... (参数部分保持不变) ...
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument(
        '--metadata_csv',
        type=str,
        default=None,
        help=(
            'CSV containing an image_path column. Required for directory input '
            'unless <input_dir>/<split>_samples.csv exists.'
        ),
    )
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--stride_size', type=int, default=8)
    parser.add_argument('--exclude_background', action='store_true')
    parser.add_argument('--apply_border_threshold', action='store_true')
    parser.add_argument('--smooth_sigma', type=float, default=1.0)
    parser.add_argument('--postprocess_image', action='store_true')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--procs_per_gpu', type=int, default=2)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Prepare Tasks (保持不变)
    tasks = []
    if os.path.isfile(args.input_dir):
        output_name = os.path.splitext(os.path.basename(args.input_dir))[0]
        tasks.append((os.path.basename(args.input_dir), output_name))
        args.input_dir = os.path.dirname(args.input_dir)
    else:
        df_path = args.metadata_csv or os.path.join(args.input_dir, f'{args.split}_samples.csv')
        if not os.path.exists(df_path):
            parser.error(
                f'Metadata CSV not found: {df_path}. Pass --metadata_csv with an '
                'authorized local manifest containing an image_path column.'
            )
        df = pd.read_csv(df_path)
        if 'image_path' not in df.columns:
            parser.error(f'Metadata CSV must contain an image_path column: {df_path}')
        for img_name in df['image_path']:
            if img_name.endswith('.ome.zarr'):
                out = os.path.dirname(img_name)
                tasks.append((img_name, out))
            elif img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                out = os.path.splitext(os.path.basename(img_name))[0]
                tasks.append((img_name, out))

    print(f"Total images: {len(tasks)}")

    # 2. Load Model & Fix Keys (修复重点在这里)
    print("Loading weights...")
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    raw_state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # --- 修复逻辑开始 ---
    # 创建一个新的 state_dict，去掉 'module.' 前缀
    model_state_dict = {}
    for k, v in raw_state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # 去掉 'module.' (7个字符)
            model_state_dict[name] = v
        else:
            model_state_dict[k] = v
    print("Model weights loaded and sanitized (removed 'module.' prefix).")
    # --- 修复逻辑结束 ---

    # 3. Process Management
    num_devices = torch.cuda.device_count()
    if args.num_gpus > num_devices: args.num_gpus = num_devices

    total_procs = args.num_gpus * args.procs_per_gpu
    if len(tasks) < total_procs: total_procs = max(1, len(tasks))

    print(f"Running {total_procs} workers on {args.num_gpus} GPUs...")

    chunk_size = int(np.ceil(len(tasks) / total_procs))
    task_chunks = [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]

    mp.set_start_method('spawn', force=True)
    processes = []

    for i in range(len(task_chunks)):
        if i >= total_procs: break
        gpu_id = i % args.num_gpus if args.num_gpus > 0 else 0
        p = mp.Process(target=worker_inference, args=(i, task_chunks[i], args, model_state_dict, gpu_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Done.")


if __name__ == '__main__':
    main()
