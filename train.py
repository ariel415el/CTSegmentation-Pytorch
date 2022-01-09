import os
from collections import defaultdict

import torch
from tqdm import tqdm

from evaluate import evaluate
from utils import plot_scores


class plotter:
    def __init__(self, plotting_dir, plot_freq=500):
        os.makedirs(plotting_dir, exist_ok=True)
        self.plotting_dir = plotting_dir
        self.data = defaultdict(list)
        self.n = 0
        self.plot_freq = plot_freq

    def plot(self, data):
        self.n += 1

        for k, v in data.items():
            self.data[k].append(v)

        if self.n % self.plot_freq == 0:
            for k, v in self.data.items():
                plot_scores(self.data[k], f'{self.plotting_dir}/{k}.png')


def train_model(model,  dataloaders, device, train_steps, train_dir):
    loss_plotter = plotter(train_dir)
    train_loader, val_loader = dataloaders

    eval_freq = train_steps // 30

    # Begin training
    pbar = tqdm(unit='Slices')
    global_step = 0
    while global_step < train_steps:
        for b_idx, (ct_volume, gt_volume) in enumerate(train_loader):
            ct_volume = ct_volume.to(device=device, dtype=torch.float32)
            gt_volume = gt_volume.to(device=device, dtype=torch.long)

            losses = model.train_one_sample(ct_volume, gt_volume, global_step)
            loss_plotter.plot(losses)

            pbar.update(ct_volume.shape[0] * ct_volume.shape[-3])
            pbar.set_description(f"GS: {global_step}/{train_steps}, Loss: {losses}")

            # Evaluation round
            if global_step % eval_freq == 0:
                val_score = evaluate(model, val_loader, device, f"{train_dir}/eval-step-{global_step}")
                model.step_scheduler(val_score)
                loss_plotter.plot({'val_scaore': val_score})

                torch.save(model.get_state_dict(), f'{train_dir}/checkpoint_epoch{global_step}.pth')

            global_step += 1
