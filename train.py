import os
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from evaluate import evaluate
from config import *


class plotter:
    def __init__(self, plotting_dir):
        os.makedirs(plotting_dir, exist_ok=True)
        self.plotting_dir = plotting_dir
        self.data = defaultdict(list)
        self.data_means = defaultdict(list)
        self.n = 0

    def register_data(self, data):
        self.n += 1

        for k, v in data.items():
            self.data[k].append(float(v))

    def plot(self):
        for k, v in self.data.items():
            last_mean = np.mean(self.data[k] if len(self.data[k]) < 5 else self.data[k][-5:])
            self.data_means[k].append(last_mean)
            nvalues = len(self.data[k])
            plt.plot(range(nvalues), self.data[k], label=k)
            plt.plot(np.linspace(0, nvalues-1, len(self.data_means[k])), self.data_means[k], label="avg-last-5")
            plt.legend()
            plt.savefig(f'{self.plotting_dir}/{k}.png')
            plt.clf()


def train_model(model, dataloaders, train_dir):
    loss_plotter = plotter(train_dir)
    train_loader, val_loader = dataloaders

    # Begin training
    pbar = tqdm(unit='Slices')
    step = 0
    model.train()
    while step < train_steps:
        for sample in train_loader:
            ct_volume = sample['ct'].to(device=device, dtype=torch.float32)
            gt_volume = sample['gt'].to(device=device, dtype=torch.long)
            mask_volume = (sample['mask'] if ignore_background else torch.ones_like(sample['mask'])).to(device=device)

            losses = model.train_one_sample(ct_volume, gt_volume, mask_volume, step)
            loss_plotter.register_data(losses)

            slices = ct_volume.shape[0] * ct_volume.shape[-3]
            pbar.update(slices)
            pbar.set_description(f"Train-step: {step}/{train_steps}, Losses: {','.join([f'{k}: {v:.3f}' for k, v in losses.items()])}, lr: {model.optimizer.param_groups[0]['lr']:.10f}")

            # Evaluation round
            if step % eval_freq == 0:
                evaluation_report = evaluate(model, val_loader, f"{train_dir}/eval-step-{step}")
                model.step_scheduler(evaluation_report['Dice-non-bg'])
                evaluation_report.pop("Slice/sec")
                loss_plotter.register_data(evaluation_report)
                loss_plotter.plot()

                torch.save(model.get_state_dict(), f'{train_dir}/step-{step}.pth')

            step += 1
