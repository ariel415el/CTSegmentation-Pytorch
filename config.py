import os
from dataclasses import dataclass
import torch


@dataclass
class ExperimentConfigs:
    # model configs#
    model_name: str = 'VGGUNet'
    n_classes: int = 2
    lr: float = 0.00001
    learnable_upsamples: bool = False
    slice_size: int = 1

    # data configs
    data_path = 'datasets/LiTS2017_(MS-(3, 15, 15)_MM-2_Crop-CL-1_margins-(1, 1, 1)_OB-0.5_MD-11)'
    val_set: str = 'A'
    resize: int = 128
    augment_data: bool = False       # Affine, noise, random intencity clipping etc.
    ignore_background: bool = False  # Adds a non-bg mask to the loss to be computed on
    delete_background: bool = False  # Erase all bg voxels
    hist_equalization: bool = False
    Z_normalization: bool = False    # normal standardization of volume intencities
    force_non_empty: bool = False    # For volumes with non-bg voxels. train only on non-empty chunks
    batch_size: int = 32
    num_workers: int = 0

    # train configs
    train_tag: str = ""
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_steps: int = 50 * 500
    eval_freq: int = 500
    debug_images = False

    def __str__(self):
        return f"{self.model_name}_R-{self.resize}" \
        f"{'_Aug' if self.augment_data else ''}" \
        f"{'_LUS' if self.learnable_upsamples else ''}" \
        f"{'_ZeroBG' if self.delete_background else ''}" \
        f"{'_MaskBg' if self.ignore_background else ''}" \
        f"{'_HistEq' if self.hist_equalization else ''}" \
        f"{'_ZNorm' if self.Z_normalization else ''}" \
        f"{'_' + self.train_tag if self.train_tag else ''}" \
        f"_V-{self.val_set}"
