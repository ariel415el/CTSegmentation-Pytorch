import torch
from torch import optim, nn
from tqdm import tqdm

from evaluate import evaluate
from models.Vnet.net import VNet
from dice_score import compute_dice_loss
from utils import plot_scores
from datasets.ct_dataset import get_dataloaders


def volume_loss(preds, gts):
    """
    :param preds: float array of shape (1, n_class, slices, H, W) contating class logits
    :param gts: uint8 array of shape (1, slices, H, W) containing segmentation labels
    """
    dice_loss = compute_dice_loss(preds, gts.unsqueeze(1))
    return dice_loss


class VnetModel:
    def __init__(self, n_channels, n_classes):
        self.net = VNet(n_channels, n_classes, d=4)

    def predict_volume(self, ct_volume):
        """
        ct_volume.shape = (b, slices, H, W)
        returns prdiction of shape (b, n_classes, slices, H, W)
        """
        return self.net(ct_volume.unsqueeze(1))

    def train(self, data_path, device, epochs=5, batch_size=1, lr=0.0001, val_percent=0.1, train_dir="train_dir"):
        self.net = self.net.to(device)

        train_loader, val_loader = get_dataloaders(data_path, val_percent, batch_size, train_by_volume=True)

        optimizer = optim.RMSprop(self.net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize val Dice score
        global_step = 0

        # Begin training
        losses = []
        val_scores = []
        pbar = tqdm()
        for epoch in range(epochs):
            self.net.train()
            for b_idx, (ct_volume, gt_volume) in enumerate(train_loader):
                ct_volume = ct_volume.to(device=device, dtype=torch.float32)
                gt_volume = gt_volume.to(device=device, dtype=torch.long)

                pred = self.net(ct_volume.unsqueeze(1))

                loss = volume_loss(pred, gt_volume)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                pbar.update(ct_volume.shape[0])
                pbar.set_description(f"Epoch: {epoch}/{epochs}, Step {b_idx}/{len(train_loader)}, Loss: {loss.item()}")

                global_step += 1

            # Evaluation round
            val_score = evaluate(self, val_loader, device, f"{train_dir}/eval-epoch-{epoch}")
            val_scores.append(val_score)
            scheduler.step(val_score)

            plot_scores(losses, f'{train_dir}/losses.png')
            plot_scores(val_scores, f'{train_dir}/val_scores.png')

            torch.save({'model': self.net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'global_step':global_step,
                        'epoch':epoch
                        },
                       f'{train_dir}/checkpoint_epoch{epoch + 1}.pth')

