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
    augment_data: bool = False
    ignore_background: bool = False
    delete_background: bool = False
    hist_equalization: bool = False
    Z_normalization: bool = False
    batch_size: int = 32
    num_workers: int = 1

    # train configs
    train_tag: str = ""
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_steps: int = 50 * 500
    eval_freq: int = 500

    def __str__(self):
        return f"{self.model_name}_R-{self.resize}" \
        f"{'_Aug' if self.augment_data else ''}" \
        f"{'_ZeroBG' if self.delete_background else ''}" \
        f"{'_MaskBg' if self.ignore_background else ''}" \
        f"{'_HistEq' if self.hist_equalization else ''}" \
        f"{'_ZNorm' if self.Z_normalization else ''}" \
        f"_V-{self.val_set}" \
        f"{'_' + self.train_tag if self.train_tag else ''}"
