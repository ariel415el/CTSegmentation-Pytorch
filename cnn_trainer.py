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
            self.data['train-loss'].append(loss)

            self.pbar.update(self.config.batch_size * self.config.slice_size)
            self.pbar.set_description(f"Train-step: {self.step}/{self.config.train_steps}, lr: {model.optimizer.param_groups[0]['lr']:.10f}")

            # Evaluation
            if self.step % self.config.eval_freq == 0:
                evaluation_report = evaluate(model, val_loader, self.config.device)
                evaluation_report.pop("Slice/sec")
                self.register_data(evaluation_report)
                self.plot()

                self.save()
                if self.is_last_smoothed_score_best('Dice-class-1'):
                    torch.save(model.get_state_dict(), f'{self.train_dir}/best.pth')

            self.step += 1
            if self.step > self.config.train_steps:
                break

        self.train_time = time() - start

    def is_last_smoothed_score_best(self, metric_name):
        return np.argmax(self.data_means[metric_name]) == len(self.data_means[metric_name]) - 1

    def get_best_smoothed(self):
        result = dict(Train_time=self.train_time)
        for k, v in self.data_means.items():
            idx = np.argmax(v)
            result[f'{k}-smoothed({self.smooth_score_size})'] = (f"Step={idx},score={v[idx]:.2f}")
        return result

    def register_data(self, loss_dict):
        for k, v in loss_dict.items():
            self.data[k].append(float(v))
            self.data_means[k].append(np.mean(self.data[k][-self.smooth_score_size:]))

    def plot(self):
        for k, v in self.data.items():
            nvalues = len(self.data[k])
            plt.plot(range(nvalues), self.data[k], label=k)
            plt.plot(np.linspace(0, nvalues - 1, len(self.data_means[k])), self.data_means[k],label=f"avg-last-{self.smooth_score_size}")
            plt.legend()
            plt.savefig(f'{self.train_dir}/{k}.png')
            plt.clf()

    def save(self):
        torch.save(dict(step=self.step, data=self.data,data_means=self.data_means), f'{self.train_dir}/trainer.pt')

    def try_load(self):
        path = os.path.join(self.train_dir, "trainer.pt")
        if os.path.exists(path):
            trainer_state = torch.load(path, map_location=self.config.device)
            logging.info("loaded trainer from file")
            self.step = trainer_state['step']
            self.data = trainer_state['data']
            self.data_means = trainer_state['data_means']

