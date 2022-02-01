import os
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from evaluate import evaluate
from config import *


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

    def train_model(self, model, dataloaders):
        model.to(self.config.device)
        model.train()
        train_loader, val_loader = dataloaders

        done_training = False
        while not done_training:
            for sample in train_loader:
                ct_volume = sample['ct'].to(device=self.config.device, dtype=torch.float32)
                gt_volume = sample['gt'].to(device=self.config.device, dtype=torch.long)
                mask_volume = sample['mask'].to(device=self.config.device, dtype=torch.bool)

                losses = model.train_one_sample(ct_volume, gt_volume, mask_volume)
                self.register_data(losses)

                self.pbar.update(self.config.batch_size * self.config.slice_size)
                self.pbar.set_description(
                    f"Train-step: {self.step}/{self.config.train_steps}, Losses: {','.join([f'{k}: {v:.3f}' for k, v in losses.items()])}, lr: {model.optimizer.param_groups[0]['lr']:.10f}")

                # Evaluation
                if self.step % self.config.eval_freq == 0:
                    debug_dir = f"{self.train_dir}/eval-step-{self.step}" if self.config.debug_images else None
                    evaluation_report = evaluate(model, val_loader, self.config.device, outputs_dir=debug_dir)
                    evaluation_report.pop("Slice/sec")
                    self.register_data(evaluation_report)
                    self.plot()

                    torch.save(model.get_state_dict(), f'{self.train_dir}/best.pth')
                    self.save(f'{self.train_dir}/trainer.pt')
                self.step += 1
                if self.step > self.config.train_steps:
                    done_training = True
                    break

    def get_best_smoothed(self):
        result = dict()
        for k, v in self.data_means.items():
            idx = np.argmax(v)
            result[f'{k}-smoothed({smooth_score_size})'] = (v[idx], idx)
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

    def save(self, path):
        torch.save(dict(step=self.step, data=self.data,data_means=self.data_means), path)

    def try_load(self, path):
        if os.path.exists(path):
            model = torch.load(path)['trainer']


