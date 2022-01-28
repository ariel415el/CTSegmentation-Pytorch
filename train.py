import os
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from evaluate import evaluate
from config import *


class Plotter:
    def __init__(self, plotting_dir, mean_size=10):
        os.makedirs(plotting_dir, exist_ok=True)
        self.plotting_dir = plotting_dir
        self.data = defaultdict(list)
        self.data_means = defaultdict(list)
        self.n = 0
        self.mean_size = mean_size
        self.dice_data = None
        self.case_to_idx = None

    def register_data(self, loss_dict):
        self.n += 1
        for k, v in loss_dict.items():
            self.data[k].append(float(v))

    # def register_evaluation_metrics(self, evaluation_report):
    #     next_entry = np.stack([v for v in evaluation_report.values()], axis=0)[:, :, None]
    #     if self.dice_data is None:
    #         self.case_to_idx = {k: i for i,k in enumerate(evaluation_report)}
    #         self.dice_data = next_entry
    #     else:
    #         self.dice_data = np.concatenate([self.dice_data, next_entry], axis=-1)

    # def get_dice_per_case(self):
    #     return self.dice_data[:, 1, -1].mean()

    def plot(self):
        for k, v in self.data.items():
            nvalues = len(self.data[k])
            plt.plot(range(nvalues), self.data[k], label=k)
            last_mean = np.mean(self.data[k] if len(self.data[k]) < self.mean_size else self.data[k][-self.mean_size:])
            self.data_means[k].append(last_mean)
            plt.plot(np.linspace(0, nvalues-1, len(self.data_means[k])), self.data_means[k], label=f"avg-last-{self.mean_size}")
            plt.legend()
            plt.savefig(f'{self.plotting_dir}/{k}.png')
            plt.clf()
        #
        # class_idx = 1
        # for case_num, idx in self.case_to_idx.items():
        #     values = self.dice_data[idx, class_idx]
        #     plt.plot(range(len(values)), values, label=case_num)

        # plt.legend()
        # plt.savefig(f'{self.plotting_dir}/Dice.png')
        # plt.clf()


def train_model(model, dataloaders, train_dir, train_configs):
    device = train_configs.device
    model.to(device)
    train_loader, val_loader = dataloaders

    # Begin training
    loss_plotter = Plotter(train_dir)
    pbar = tqdm(unit='Slices')
    step = 0
    model.train()
    max_score = -np.inf
    while step < train_configs.train_steps:
        for sample in train_loader:
            ct_volume = sample['ct'].to(device=device, dtype=torch.float32)
            gt_volume = sample['gt'].to(device=device, dtype=torch.long)
            mask_volume = sample['mask'].to(device=device, dtype=torch.bool)

            losses = model.train_one_sample(ct_volume, gt_volume, mask_volume, step)
            loss_plotter.register_data(losses)

            slices = ct_volume.shape[0] * ct_volume.shape[-3]
            pbar.update(slices)
            pbar.set_description(f"Train-step: {step}/{train_configs.train_steps}, Losses: {','.join([f'{k}: {v:.3f}' for k, v in losses.items()])}, lr: {model.optimizer.param_groups[0]['lr']:.10f}")

            # Evaluation
            if step % train_configs.eval_freq == 0:
                evaluation_report = evaluate(model, val_loader, train_configs.device, f"{train_dir}/eval-step-{step}")
                score = evaluation_report['Dice-class-1']
                model.step_scheduler(score)
                evaluation_report.pop("Slice/sec")
                loss_plotter.register_data(evaluation_report)
                loss_plotter.plot()
                torch.save(loss_plotter.data, f'{train_dir}/metrics.pt')

                ckpt = model.get_state_dict()
                ckpt.update({'step': step, 'Dice-class-1': score})
                torch.save(ckpt, f'{train_dir}/latest.pth')
                if score > max_score:
                    max_score = score
                    torch.save(ckpt, f'{train_dir}/best.pth')

            step += 1
