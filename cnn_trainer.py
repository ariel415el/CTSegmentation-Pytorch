import os
from collections import defaultdict
import logging
from time import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from evaluate import evaluate
from config import *
from metrics import VolumeLoss


def iterate_dataloader(dataloader):
    """
    Iterates dataloader infinitely
    """
    while True:
        for sample in dataloader:
            yield sample


class CNNTrainer:
    def __init__(self, train_configs, smooth_score_size=10):
        """
        Manages the training process of a model and monitors it
        """
        self.config = train_configs
        self.volume_crieteria = VolumeLoss(self.config.dice_loss_weight, self.config.wce_loss_weight, self.config.ce_loss_weight)
        self.step = 0
        self.plot_data = defaultdict(list)
        self.plot_data_means = defaultdict(list)
        self.smooth_score_size = smooth_score_size
        self.train_time = 0
        self.pbar = tqdm(unit='Slices')

    def train_model(self, model, dataloaders, train_dir):
        self.train_dir = train_dir
        os.makedirs(self.train_dir, exist_ok=True)

        model.to(self.config.device)
        model.train()

        train_loader, val_loader = dataloaders

        logging.info('Training..')
        start = time()
        for sample in iterate_dataloader(train_loader):
            ct_volume = sample['ct'].to(device=self.config.device, dtype=torch.float32)
            gt_volume = sample['gt'].to(device=self.config.device, dtype=torch.long)
            mask_volume = sample['mask'].to(device=self.config.device, dtype=torch.bool)

            loss = model.train_one_sample(ct_volume, gt_volume, mask_volume, self.volume_crieteria)

            self.register_plot_data({'train-loss': loss})
            self.pbar.update(ct_volume.shape[0] * ct_volume.shape[-3])
            self.pbar.set_description(f"Train-step: {self.step}/{self.config.train_steps}, lr: {model.optimizer.param_groups[0]['lr']:.10f}")

            # Evaluation
            if self.step % self.config.eval_freq == 0:
                validation_report = evaluate(model, val_loader, self.config.device, self.volume_crieteria)
                validation_report['val-loss'] = validation_report.pop('Loss')
                self.register_plot_data(validation_report)
                self.plot()

                self.save_checkpoint(model, name='latest')
                if self.is_last_smoothed_score_best('Dice-class-1'):
                    self.save_checkpoint(model, name='best')

            if self.step % self.config.ckpt_frequency == 0:
                self.save_checkpoint(model, name=f'step-{self.step}')

            if self.step % self.config.decay_steps == 0 and self.step > 0:
                model.decay_learning_rate(self.config.decay_factor)

            self.step += 1
            if self.step > self.config.train_steps:
                break

        self.train_time += time() - start

    def is_last_smoothed_score_best(self, metric_name):
        return np.argmax(self.plot_data_means[metric_name]) == len(self.plot_data_means[metric_name]) - 1

    def get_report(self):
        report = {'Train time (H)': f"{self.train_time/3600:.2f}"}
        for k, v in self.plot_data_means.items():
            idx = np.argmax(v)
            report[f'{k}-({self.smooth_score_size}-smooth) Step'] = idx
            report[f'{k}-({self.smooth_score_size}-smooth) Score'] = f"{v[idx]:.3f}"
        return report

    def register_plot_data(self, loss_dict):
        for k, v in loss_dict.items():
            self.plot_data[k].append(float(v))
            self.plot_data_means[k].append(np.mean(self.plot_data[k][-self.smooth_score_size:]))

    def plot(self):
        metric_groups = [['train-loss', 'val-loss'], ['Dice-class-1']]
        for metric_group in metric_groups:
            nvalues = max([len(self.plot_data[k]) for k in metric_group])
            for k in metric_group:
                plt.plot(np.linspace(0, nvalues - 1, len(self.plot_data[k])), self.plot_data[k],
                         alpha=0.5, label=f"{k}: {self.plot_data[k][-1]:.3f}")
                plt.plot(np.linspace(0, nvalues - 1, len(self.plot_data_means[k])), self.plot_data_means[k],
                         alpha=0.5, label=f"avg-last-{self.smooth_score_size}: {self.plot_data_means[k][-1]:.3f}")
            plt.legend()
            plt.savefig(f'{self.train_dir}/Plot({",".join(metric_group)}).png')
            plt.clf()

    def save_checkpoint(self, model, name):
        """
        Saves model weights and trainer inner state to file
        """
        trainer_state_dict = dict(step=self.step, data=self.plot_data, data_means=self.plot_data_means, train_time=self.train_time)
        torch.save(dict(trainer=trainer_state_dict, model=model.get_state_dict()), f'{self.train_dir}/{name}.pth')

    def load_state(self, trainer_state):
        logging.info("loaded trainer from file")
        self.step = trainer_state['step']
        self.plot_data = trainer_state['data']
        self.plot_data_means = trainer_state['data_means']
        self.train_time = trainer_state['train_time']

