import logging
import os
import pandas as pd

from datasets.ct_dataset import get_dataloaders
from cnn_trainer import CNNTrainer

from config import *
from models import get_model


def train_model(exp_config, outputs_dir):
    model = get_model(exp_config.get_model_config())
    dataloaders = get_dataloaders(exp_config.get_data_config())
    trainer = CNNTrainer(exp_config.get_train_configs())

    model_dir = f"{outputs_dir}/{exp_config}"

    # copy config file
    exp_config.write_to_file(model_dir)

    # Try to load checkpoint
    if os.path.exists(os.path.join(model_dir, "latest.pth")):
        logging.info("Starting from latest checkpoint")
        ckpt = torch.load(os.path.join(model_dir, "latest.pth"), map_location=torch.device("cpu"))
        trainer.load_state(ckpt['trainer'])
        model.load_state_dict(ckpt['model'])

    trainer.train_model(model, dataloaders, model_dir)

    train_report = trainer.get_report()

    return train_report


def run_multiple_experiments(outputs_dir):
    """
    Run multiple models with common configs on multple test sets and produce a report.
    """
    os.makedirs(outputs_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(outputs_dir, 'log-file.log'), format='%(asctime)s:%(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')

    full_report = pd.DataFrame()
    common_kwargs = dict(starting_lr=0.00001, num_workers=4, val_set='A', train_steps=100000,
                      dice_loss_weight=0, wce_loss_weight=0, ce_loss_weight=1,
                      augment_data=True, elastic_deformations=False, force_non_empty=0.5)
    for exp_config in [
        ExperimentConfigs(model_name='VGGUNet', **common_kwargs, resize=200, slice_size=1, batch_size=16),
        ExperimentConfigs(model_name='VGGUNet2_5D', **common_kwargs, resize=200, slice_size=3, batch_size=16),
        ExperimentConfigs(model_name='UNet3D', **common_kwargs, resize=128, slice_size=32, batch_size=2),
    ]:
        logging.info(f'#### {exp_config} ####')

        experiment_report = train_model(exp_config, outputs_dir)
        experiment_report['Model_name'] = str(exp_config)
        experiment_report['N_slices'] = exp_config.train_steps * exp_config.batch_size * exp_config.slice_size

        full_report = full_report.append(experiment_report, ignore_index=True)

    full_report = full_report.set_index('Model_name')
    full_report.to_csv(os.path.join(outputs_dir, f"Final-report.csv"), sep=',')



