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


def iterate_dataloader(dataloader):
    while True:
        for sample in dataloader:
            yield sample


class CNNTrainer:
    def __init__(self, train_configs, train_dir, smooth_score_size=10):
        os.makedirs(train_dir, exist_ok=True)
        self.train_dir = train_dir
        self.config = train_configs
        self.step = 0
        self.pbar = tqdm(unit='Slices')
        self.data = defaultdict(list)
        self.data_means = defaultdict(list)
        self.smooth_score_size = smooth_score_size
        self.train_time = 0

    def train_model(self, model, dataloaders):
        model.to(self.config.device)
        model.train()
        train_loader, val_loader = dataloaders
        logging.info('Training..')
        start = time()
        for sample in iterate_dataloader(train_loader):
            ct_volume = sample['ct'].to(device=self.config.device, dtype=torch.float32)
            gt_volume = sample['gt'].to(device=self.config.device, dtype=torch.long)
            mask_volume = sample['mask'].to(device=self.config.device, dtype=torch.bool)

            loss = model.train_one_sample(ct_volume, gt_volume, mask_volume)
            self.register_data({'train-loss': loss})

            self.pbar.update(self.config.batch_size * self.config.slice_size)
            self.pbar.set_description(f"Train-step: {self.step}/{self.config.train_steps}, lr: {model.optimizer.param_groups[0]['lr']:.10f}")

            # Evaluation
            if self.step % self.config.eval_freq == 0:
                validation_report = evaluate(model, val_loader, self.config.device)
                validation_report['val-loss'] = validation_report.pop('Loss')
                self.register_data(validation_report)
                self.plot()

                self.save_checkpoint(model, name='latest')
                if self.is_last_smoothed_score_best('Dice-class-1'):
                    self.save_checkpoint(model, 'best')

            self.step += 1
            if self.step > self.config.train_steps:
                break

        self.train_time += time() - start

    def is_last_smoothed_score_best(self, metric_name):
        return np.argmax(self.data_means[metric_name]) == len(self.data_means[metric_name]) - 1

    def get_report(self):
        report = dict(Train_time=self.train_time)
        for k, v in self.data_means.items():
            idx = np.argmax(v)
            report[f'{k}-smoothed({self.smooth_score_size})'] = (f"Step={idx},score={v[idx]:.2f}")
        return report

    def register_data(self, loss_dict):
        for k, v in loss_dict.items():
            self.data[k].append(float(v))
            self.data_means[k].append(np.mean(self.data[k][-self.smooth_score_size:]))

    def plot(self):
        metric_groups = [['train-loss', 'val-loss'], ['Dice-class-1']]
        for metric_group in metric_groups:
            nvalues = max([len(self.data[k]) for k in metric_group])
            for k in metric_group:
                plt.plot(np.linspace(0, nvalues - 1, len(self.data[k])), self.data[k],
                         alpha=0.5, label=f"{k}: {self.data[k][-1]:.2f}")
                plt.plot(np.linspace(0, nvalues - 1, len(self.data_means[k])), self.data_means[k],
                         alpha=0.5, label=f"avg-last-{self.smooth_score_size}: {self.data_means[k][-1]:.2f}")
            plt.legend()
            plt.savefig(f'{self.train_dir}/Plot({",".join(metric_group)}).png')
            plt.clf()

    def get_state(self):
        return dict(step=self.step, data=self.data, data_means=self.data_means, train_time=self.train_time)

    def save_checkpoint(self, model, name):
        torch.save(dict(trainer=self.get_state(), model=model.get_state_dict()), f'{self.train_dir}/{name}.pth')

    def load_state(self, trainer_state):
        logging.info("loaded trainer from file")
        self.step = trainer_state['step']
        self.data = trainer_state['data']
        self.data_means = trainer_state['data_means']
        self.train_time = trainer_state['train_time']

