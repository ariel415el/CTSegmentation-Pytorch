import argparse
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets.ct_dataset import get_dataloaders
from evaluate import evaluate

from config import *
from metrics import VolumeLoss
from models import get_model


def test(train_data_root, model_dir, ckpt_name='best', dump_debug_images=False):
    """
    Test the model in a checkpoint on the entire dataset.
    """
    ckpt = torch.load(f'{model_dir}/{ckpt_name}.pth')

    config = ExperimentConfigs(**json.load(open(f"{model_dir}/exp_configs.json")))
    config.data_path = train_data_root

    # get dataloaders
    config.batch_size = 1
    train_loader, val_loader = get_dataloaders(config.get_data_config())
    train_loader.dataset.transforms = val_loader.dataset.transforms  # avoid slicing and use full volumes

    # get model
    model = get_model(config.get_model_config())
    model.load_state_dict(ckpt['model'])
    model.to(config.device)

    volume_crieteria = VolumeLoss(config.dice_loss_weight, config.wce_loss_weight, config.ce_loss_weight)
    outputs_dir = os.path.join(model_dir, f'ckpt-{ckpt_name}', "val_debug") if dump_debug_images else None

    validation_report = evaluate(model, val_loader, config.device, volume_crieteria, outputs_dir=outputs_dir)
    print(validation_report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Lits2017 dataset')
    parser.add_argument('model_dir')
    parser.add_argument('train_data_root')
    parser.add_argument('--checkpoint_name', default='best')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    test(args.train_data_root, args.model_dir, args.checkpoint_name, dump_debug_images=args.debug)