from dataclasses import dataclass
import torch


@dataclass
class ExperimentConfigs:
    # model configs
    model_name: str = 'VGGUNet'
    n_classes: int = 2
    learnable_upsamples: bool = False
    starting_lr: float = 0.00001

    # data configs
    data_path = 'datasets/LiTS2017_(MS-(3, 15, 15)_MM-2_Crop-CL-1_margins-(1, 1, 1)_OB-0.5_MD-11)'
    val_set: str = 'A'
    resize: int = 128
    augment_data: bool = False  # Affine, noise, random intencity clipping etc.
    ignore_background: bool = False  # Adds a non-bg mask to the loss to be computed on
    delete_background: bool = False  # Erase all bg voxels
    hist_equalization: bool = False
    Z_normalization: bool = False  # normal standardization of volume intencities
    force_non_empty: bool = False  # For volumes with non-bg voxels. train only on non-empty chunks
    batch_size: int = 32
    num_workers: int = 4

    # common configs
    slice_size: int = 1

    # train configs
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    decay_steps: float = 10000
    decay_factor: float = 1
    dice_loss_weight: int = 1
    wce_loss_weight: int = 1
    ce_loss_weight: int = 0
    ckpt_frequency: int = 10000
    train_steps: int = 50 * 500
    eval_freq: int = 500

    train_tag: str = ""

    def get_data_config(self):
        return DataConfigs(self.data_path, self.val_set, self.resize, self.slice_size, self.augment_data,
                           self.ignore_background, self.delete_background, self.hist_equalization, self.Z_normalization,
                           self.force_non_empty, self.batch_size, self.num_workers)

    def get_model_config(self):
        return ModelConfigs(self.model_name, self.n_classes, self.learnable_upsamples, self.slice_size,
                            self.starting_lr)

    def get_train_configs(self):
        return TrainConfigs(self.device, self.decay_steps, self.decay_factor, self.dice_loss_weight,
                            self.wce_loss_weight, self.ce_loss_weight, self.ckpt_frequency, self.train_steps, self.eval_freq)

    def __str__(self):
        return f"{'_Aug' if self.augment_data else ''}" \
               f"{'_LUS' if self.learnable_upsamples else ''}" \
               f"{'_ZeroBG' if self.delete_background else ''}" \
               f"{'_MaskBg' if self.ignore_background else ''}" \
               f"{'_HistEq' if self.hist_equalization else ''}" \
               f"{'_ZNorm' if self.Z_normalization else ''}" \
               f"{'_FNE' if self.force_non_empty else ''}" \
               f"_Loss({self.dice_loss_weight:.1f}Dice+{self.wce_loss_weight:.1f}WCE+{self.ce_loss_weight:.1f}CE)" \
               f"{'_' + self.train_tag if self.train_tag else ''}" \
               f"_V-{self.val_set}"


@dataclass
class ModelConfigs:
    model_name: str
    n_classes: int
    learnable_upsamples: bool
    slice_size: int
    starting_lr: float


@dataclass
class DataConfigs:
    data_path: str
    val_set: str
    resize: int
    slice_size: int
    augment_data: bool
    ignore_background: bool
    delete_background: bool
    hist_equalization: bool
    Z_normalization: bool
    force_non_empty: bool
    batch_size: int
    num_workers: int

@dataclass
class TrainConfigs:
    device: torch.device
    decay_steps: float
    decay_factor: float
    dice_loss_weight: int
    wce_loss_weight: int
    ce_loss_weight: int
    ckpt_frequency: int
    train_steps: int
    eval_freq: int

