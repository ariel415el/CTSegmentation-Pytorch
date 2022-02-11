import logging
import os
import random
import sys
import pandas as pd
import json

from datasets.ct_dataset import get_dataloaders
from evaluate import evaluate
from cnn_trainer import CNNTrainer

from config import *
from metrics import VolumeLoss
from models import get_model


def train(exp_config, outputs_dir):
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


def test(model_dir, ckpt_name='best'):
    """
    Test the model in a checkpoint on the entire dataset.
    """
    ckpt = torch.load(f'{model_dir}/{ckpt_name}.pth')

    config = ExperimentConfigs(**json.load(open(f"{model_dir}/exp_configs.json")))

    # get dataloaders
    config.batch_size = 1
    train_loader, val_loader = get_dataloaders(config.get_data_config())
    train_loader.dataset.transforms = val_loader.dataset.transforms  # avoid slicing and use full volumes

    # get model
    model = get_model(config.get_model_config())
    model.load_state_dict(ckpt['model'])
    model.to(config.device)

    volume_crieteria = VolumeLoss(config.dice_loss_weight, config.wce_loss_weight, config.ce_loss_weight)
    validation_report = evaluate(model, val_loader, config.device, volume_crieteria, outputs_dir=os.path.join(model_dir, f'ckpt-{ckpt_name}', "val_debug"))
    print(validation_report)
    # train_report = evaluate(model, train_loader, config.device, volume_crieteria, outputs_dir=os.path.join(model_dir, f'ckpt-{ckpt_name}', "train_debug"))
    #
    # report = pd.DataFrame()
    # report = report.append({f"{ckpt_name}-{k}": f"{train_report[k]:.3f} / {validation_report[k]:.3f}" for k in train_report}, ignore_index=True)
    # report.to_csv(os.path.join(model_dir, f'ckpt-{ckpt_name}', f"Test-Report.csv"), sep=',')


def run_single_experiment(outputs_dir):
    """
    Run a single experiment
    """
    os.makedirs(outputs_dir, exist_ok=True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    exp_config = ExperimentConfigs(model_name='VGGUNet', starting_lr=0.00001, batch_size=32, slice_size=1,
                                   train_steps=30000,
                                   wce_loss_weight=0, dice_loss_weight=0, ce_loss_weight=1,
                                   augment_data=True,
                                   num_workers=2, eval_freq=1000)

    train_report = train(exp_config, outputs_dir)

    logging.info(train_report)


def run_multiple_experiments(outputs_dir):
    """
    Run multiple models with common configs on multple test sets and produce a report.
    """
    os.makedirs(outputs_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(outputs_dir, 'log-file.log'), format='%(asctime)s:%(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')

    full_report = pd.DataFrame()
    common_kwargs = dict(starting_lr=0.00001, num_workers=4, train_steps=30000,
                         augment_data=True, elastic_deformations=False,
                         dice_loss_weight=0, wce_loss_weight=0, ce_loss_weight=1)
    for exp_config in [
        ExperimentConfigs(model_name='VGGUNet', **common_kwargs, val_set='A', resize=128, batch_size=32),
        ExperimentConfigs(model_name='VGGUNet', **common_kwargs, val_set='A', resize=256, batch_size=4),
        ExperimentConfigs(model_name='VGGUNet2_5D', **common_kwargs, val_set='A', slice_size=3, batch_size=32),

    ]:
        logging.info(f'#### {exp_config} ####')

        experiment_report = train(exp_config, outputs_dir)
        experiment_report['Model_name'] = str(exp_config)
        experiment_report['N_slices'] = exp_config.train_steps * exp_config.batch_size * exp_config.slice_size

        full_report = full_report.append(experiment_report, ignore_index=True)

    full_report = full_report.set_index('Model_name')
    full_report.to_csv(os.path.join(outputs_dir, f"Final-report.csv"), sep=',')


if __name__ == '__main__':
    outputs_root = '/mnt/storage_ssd/train_outputs'
    random.seed(1)
    torch.manual_seed(1)
    # run_single_experiment(f"{outputs_root}/train_dir_liver")
    # run_multiple_experiments(f"{outputs_root}/cluster_LiverTraining")
    test('/home/ariel/projects/MedicalImageSegmentation/AWS_scripts/dir_dir/cluster_batch-3-C/VGGUNet2_5D_Aug_Elastic_MaskBg_FNE-1.0_Loss(0.0Dice+0.0WCE+1.0CE)_V-C', 'best')
    # run_multiple_experiments(f"{outputs_root}/cluster_training_batch-4")
