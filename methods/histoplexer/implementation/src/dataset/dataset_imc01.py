import os
import random
from typing import List, Optional

import numpy as np
import scipy
import torch
from PIL import Image

from src.dataset.dataset import BaseDataset
from src.utils.data.transforms import HE_transforms, shared_transforms


class TuProDatasetIMC01(BaseDataset):
    """
    TuProDataset variant that scales IMC/IF patches to [0,1].
    This is separated from the original dataset for experiment isolation.
    """

    def __init__(
        self,
        split: str,
        mode: str,
        src_folder: str,
        tgt_folder: str,
        use_high_res: bool = True,
        p_flip_jitter_hed_affine: List[float] = [0.5, 0.0, 0.5, 0.5],
        patch_size: int = 256,
        channels: Optional[List[int]] = None,
        cohort: Optional[str] = None,
        use_fm_features: bool = False,
        fm_features_path: Optional[str] = None,
        imc_scale: float = 255.0,
        imc_clip: bool = True,
    ):
        super().__init__(split, mode, src_folder, tgt_folder)
        self.use_high_res = use_high_res
        self.he_transforms = HE_transforms
        self.shared_transforms = shared_transforms
        self.p_shared = p_flip_jitter_hed_affine[0]
        self.p_jitter = p_flip_jitter_hed_affine[1]
        self.p_hed = p_flip_jitter_hed_affine[2]
        self.p_affine = p_flip_jitter_hed_affine[3]
        self.patch_size = patch_size
        self.channels = channels
        self.cohort = cohort

        self.use_fm_features = use_fm_features
        self.fm_features_path = fm_features_path
        if self.use_fm_features:
            import h5py

            self.fm_features = h5py.File(self.fm_features_path, "r")

        self.imc_scale = float(imc_scale)
        self.imc_clip = bool(imc_clip)

    def __len__(self) -> int:
        return len(self.src_paths)

    def _scale_imc(self, imc_patch: np.ndarray) -> np.ndarray:
        imc = imc_patch.astype(np.float32)
        if self.imc_clip:
            imc = np.clip(imc, 0.0, self.imc_scale)
        return imc / self.imc_scale

    def __getitem__(self, idx: int) -> dict:
        sample = os.path.basename(self.src_paths[idx]).split(".")[0]

        if self.cohort == "tupro":
            he_roi = np.load(self.src_paths[idx], mmap_mode="r")
            imc_roi = np.load(self.tgt_paths[idx], mmap_mode="r")

            if self.channels:
                imc_roi = imc_roi[:, :, self.channels]

            augment_x_offset = random.randint(0, 1000 - self.patch_size)
            augment_y_offset = random.randint(0, 1000 - self.patch_size)

            imc_patch = imc_roi[
                augment_y_offset : augment_y_offset + self.patch_size,
                augment_x_offset : augment_x_offset + self.patch_size,
                :,
            ]

            factor = int(he_roi.shape[1] / imc_roi.shape[1])  # assume height == width
            if not self.use_high_res:
                he_roi = scipy.ndimage.zoom(
                    he_roi, (1.0 / factor, 1.0 / factor, 1), order=1
                )
                he_patch = he_roi[
                    augment_y_offset : augment_y_offset + self.patch_size,
                    augment_x_offset : augment_x_offset + self.patch_size,
                    :,
                ]
            else:
                he_patch = he_roi[
                    factor * augment_y_offset : factor * augment_y_offset + factor * self.patch_size,
                    factor * augment_x_offset : factor * augment_x_offset + factor * self.patch_size,
                    :,
                ]
        elif self.cohort == "orion":
            import pyvips

            he_patch = np.array(Image.open(self.src_paths[idx]).convert("RGB"))
            imc_patch = pyvips.Image.new_from_file(
                self.tgt_paths[idx], memory=True, access="sequential"
            ).numpy()
            augment_x_offset = (he_patch.shape[1] - self.patch_size) // 2
            augment_y_offset = (he_patch.shape[1] - self.patch_size) // 2
            he_patch = he_patch[
                augment_y_offset : augment_y_offset + self.patch_size,
                augment_x_offset : augment_x_offset + self.patch_size,
                :,
            ]
            imc_patch = imc_patch[
                augment_y_offset : augment_y_offset + self.patch_size,
                augment_x_offset : augment_x_offset + self.patch_size,
                :,
            ]
            imc_patch = np.delete(imc_patch, 13, axis=2)
        elif self.cohort == "nsclc-imc":
            he_patch = np.array(Image.open(self.src_paths[idx]).convert("RGB"))
            imc_patch = np.load(self.tgt_paths[idx], mmap_mode="r")
            augment_x_offset = 0
            augment_y_offset = 0
        else:
            he_patch = np.array(Image.open(self.src_paths[idx]).convert("RGB"))
            imc_patch = np.load(self.tgt_paths[idx], mmap_mode="r")
            augment_x_offset = 0
            augment_y_offset = 0

        imc_patch = self._scale_imc(imc_patch)

        he_patch = he_patch.transpose((2, 0, 1))
        imc_patch = imc_patch.transpose((2, 0, 1))
        he_patch = torch.from_numpy(he_patch.astype(np.float32))
        imc_patch = torch.from_numpy(imc_patch.astype(np.float32))

        he_patch, imc_patch = self.shared_transforms(he_patch, imc_patch, p=self.p_shared)
        he_patch = self.he_transforms(he_patch, p=[self.p_jitter, self.p_hed, self.p_affine])

        if he_patch.shape[0] != 3:
            he_patch = torch.from_numpy(he_patch.transpose((2, 0, 1)))

        data = {
            "he_patch": he_patch.to(torch.float),
            "imc_patch": imc_patch.to(torch.float),
            "sample": sample,
            "x_offset": augment_x_offset,
            "y_offset": augment_y_offset,
            "he_path": self.src_paths[idx],
            "imc_path": self.tgt_paths[idx],
            "idx": idx,
        }

        if self.use_fm_features:
            data["fm_features"] = self.fm_features[sample][:]
        return data
