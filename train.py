import torch
from torch import optim
from tqdm import tqdm

from evaluate import evaluate
from utils import plot_scores


def train_model(model, criterion,  dataloaders, device, lr, train_steps, train_dir):
    model.net = model.net.to(device)
    model.net.train()

    train_loader, val_loader = dataloaders

    optimizer = optim.RMSprop(model.net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize val Dice score

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

            pred = model.net(ct_volume)

            loss = criterion(pred, gt_volume)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            pbar.update(ct_volume.shape[-3])
            pbar.set_description(f"GS: {global_step}/{train_steps}, Loss: {loss.item()}")

            # Evaluation round
            if global_step % eval_freq == 0:
                val_score = evaluate(model, val_loader, device, f"{train_dir}/eval-step-{global_step}")
                val_scores.append(val_score)
                scheduler.step(val_score)

                plot_scores(losses, f'{train_dir}/losses.png')
                plot_scores(val_scores, f'{train_dir}/val_scores.png')

                torch.save({'net': model.net.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'global_step':global_step,
                            },
                           f'{train_dir}/checkpoint_epoch{global_step}.pth')

            global_step += 1
