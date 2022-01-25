import os
from dataclasses import dataclass
import torch


@dataclass
class ModelConfigs:
    model_name: str = 'VGGUNet'
    n_classes: int = 2
    lr: float = 0.00001
    learnable_upsamples: bool = False

    def __str__(self):
        return f"{self.model_name}" #(b={self.batch_size},S={self.slice_size})" \


@dataclass
class DataConfigs:
    data_path = 'datasets/LiTS2017_(MS-(3, 15, 15)_MM-2_Crop-CL-1_margins-(1, 1, 1)_OB-0.5_MD-11)'
    val_set: str = 'A'
    resize: int = 128
    augment_data: bool = False
    # ignore_background: bool = False
    delete_background: bool = False
    hist_equalization: bool = False
    Z_normalization: bool = False
    learnable_upsamples: bool = False
    batch_size: int = 32
    slice_size: int = 1
    num_workers: int = 0

    def __str__(self):
        return f"R-{self.resize}" \
        f"{'_Aug' if self.augment_data else ''}" \
        f"{'_ZeroBG' if self.delete_background else ''}" \
        f"{'_HistEq' if self.hist_equalization else ''}" \
        f"{'_ZNorm' if self.Z_normalization else ''}" \
        f"{'_LUS' if self.learnable_upsamples else ''}" \
        f"{self.val_set}" \
        # f"{'_MaskBg' if self.ignore_background else ''}" \


@dataclass
class TrainConfigs:
    train_tag: str = ""
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_steps: int = 3 * 200
    eval_freq: int = 200


def compose_experiment_name(model_configs, data_configs, train_configs):
    return f"automated_train_dir/{os.path.basename(data_configs.data_path)}/{model_configs}-{data_configs}" \
                f"{'_' + train_configs.train_tag if train_configs.train_tag else ''}"

