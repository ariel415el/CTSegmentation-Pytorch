import torch
from torch import optim
from tqdm import tqdm

from evaluate import evaluate
from utils import plot_scores


def train_model(model,  dataloaders, device, train_steps, train_dir):
    train_loader, val_loader = dataloaders

    eval_freq = train_steps // 30

    # Begin training
    losses = []
    val_scores = []
    pbar = tqdm(unit='Slices')
    global_step = 0
    while global_step < train_steps:
        for b_idx, (ct_volume, gt_volume) in enumerate(train_loader):
            ct_volume = ct_volume.to(device=device, dtype=torch.float32)
            gt_volume = gt_volume.to(device=device, dtype=torch.long)

            loss = model.train_one_sample(ct_volume, gt_volume, global_step)

            losses.append(loss)

            pbar.update(ct_volume.shape[-3])
            pbar.set_description(f"GS: {global_step}/{train_steps}, Loss: {loss}")

            # Evaluation round
            if global_step % eval_freq == 0:
                val_score = evaluate(model, val_loader, device, f"{train_dir}/eval-step-{global_step}")
                val_scores.append(val_score)
                model.step_scheduler(val_score)

                plot_scores(losses, f'{train_dir}/losses.png')
                plot_scores(val_scores, f'{train_dir}/val_scores.png')

                torch.save(model.get_state_dict(), f'{train_dir}/checkpoint_epoch{global_step}.pth')

            global_step += 1
