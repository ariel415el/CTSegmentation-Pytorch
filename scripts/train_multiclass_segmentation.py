import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import ExperimentConfigs
from train import train_model

if __name__ == '__main__':
    outputs_dir = 'my_trained_models'
    os.makedirs(outputs_dir, exist_ok=True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    exp_config = ExperimentConfigs(model_name='VGGUNet2_5D', data_mode='multiclass', data_path='datasets/LiTS2017_C-(3, 10, 10)',
                                   starting_lr=0.00001, batch_size=32, slice_size=1, train_steps=30000,
                                   wce_loss_weight=0, dice_loss_weight=0, ce_loss_weight=1,
                                   augment_data=True,
                                   num_workers=2, eval_freq=1000)

    train_report = train_model(exp_config, outputs_dir)

    logging.info(train_report)
