import os
from collections import defaultdict

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from evaluate import evaluate


def plot_scores(values, path):
    plt.plot(range(len(values)), values)
    plt.savefig(path)
    plt.clf()


class plotter:
    def __init__(self, plotting_dir):
        os.makedirs(plotting_dir, exist_ok=True)
        self.plotting_dir = plotting_dir
        self.data = defaultdict(list)
        self.n = 0

    def register_data(self, data):
        self.n += 1

        for k, v in data.items():
            self.data[k].append(v)

    def plot(self):
        for k, v in self.data.items():
            plot_scores(self.data[k], f'{self.plotting_dir}/{k}.png')


def train_model(model,  dataloaders, device, train_steps, train_dir):
    loss_plotter = plotter(train_dir)
    train_loader, val_loader = dataloaders

    eval_freq = train_steps // 50

    # Begin training
    pbar = tqdm(unit='Slices')
    step = 0
    while step < train_steps:
        for sample in train_loader:
            ct_volume = sample['ct'].to(device=device, dtype=torch.float32)
            gt_volume = sample['gt'].to(device=device, dtype=torch.long)
            mask_volume = sample['mask'].to(device=device)

            losses = model.train_one_sample(ct_volume, gt_volume, mask_volume, step)
            loss_plotter.register_data(losses)

            slices = ct_volume.shape[0] * ct_volume.shape[-3]
            pbar.update(slices)
            pbar.set_description(f"Train-step: {step}/{train_steps}, Losses: {[f'{k}: {v:.3f}' for k, v in losses.items()]}, lr: {model.optimizer.param_groups[0]['lr']}")

            # Evaluation round
            if step % eval_freq == 0:
                evaluation_report = evaluate(model, val_loader, device, f"{train_dir}/eval-step-{step}")
                model.step_scheduler(evaluation_report['Dice-non-bg'])
                evaluation_report.pop("Slice/sec")
                loss_plotter.register_data(evaluation_report)
                loss_plotter.plot()

                torch.save(model.get_state_dict(), f'{train_dir}/latest.pth')

            step += 1
