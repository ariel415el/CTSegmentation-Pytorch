import torch
from torch import optim, nn
from tqdm import tqdm

from evaluate import evaluate
from models.Unet.net import UNet
from dice_score import compute_dice_loss
from utils import plot_scores
from datasets.ct_dataset import get_dataloaders


def slice_loss_loss(preds, gts):
    """
    :param preds: float array of shape (1, n_class, H, W) contating class logits
    :param gts: uint8 array of shape (1, H, W) containing segmentation labels
    """
    ce_loss = nn.CrossEntropyLoss()(preds, gts)
    dice_loss = compute_dice_loss(preds.unsqueeze(2), gts.unsqueeze(1).unsqueeze(2))
    return ce_loss + dice_loss


class UnetModel:
    def __init__(self, n_channels, n_classes, bilinear=True):
        self.net = UNet(n_channels, n_classes, bilinear=bilinear)

    def predict_volume(self, ct_volume):
        """
        ct_volume.shape = (b, slices, H, W)
        returns prdiction of shape (b, n_classes, slices, H, W)
        """
        pred_volume = []
        with torch.no_grad():
            for i in range(ct_volume.shape[1]):
                image = ct_volume[:, i].to(dtype=torch.float32)
                pred_volume.append(self.net(image.unsqueeze(1)))
        pred_volume = torch.stack(pred_volume, dim=2).cpu()
        return pred_volume

    def train(self, data_path, device, epochs=5, batch_size=1, lr=0.0001, val_percent=0.1, eval_freq=5000, train_dir="train_dir"):
        self.net = self.net.to(device)

        train_loader, val_loader = get_dataloaders(data_path, val_percent, batch_size)

        optimizer = optim.RMSprop(self.net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize val Dice score
        global_step = 0

        # Begin training
        losses = []
        val_scores = []
        pbar = tqdm()
        for epoch in range(epochs):
            self.net.train()
            for b_idx, (ct_slice, gt_slice) in enumerate(train_loader):  #  ct_volumes.shape = (b, c, slices, H, W)
                ct_slice = ct_slice.to(device=device, dtype=torch.float32)
                gt_slice = gt_slice.to(device=device, dtype=torch.long)

                pred = self.net(ct_slice.unsqueeze(1))

                loss = slice_loss_loss(pred, gt_slice)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                pbar.update(ct_slice.shape[0])
                pbar.set_description(f"Epoch: {epoch}/{epochs}, Step {b_idx}/{len(train_loader)}, Loss: {loss.item()}")

                # Evaluation round
                if global_step % eval_freq == 0:
                    val_score = evaluate(self, val_loader, device, f"{train_dir}/eval-step-{global_step}")
                    val_scores.append(val_score)
                    scheduler.step(val_score)

                    plot_scores(losses, f'{train_dir}/losses.png')
                    plot_scores(val_scores, f'{train_dir}/val_scores.png')

                global_step += 1

            torch.save({'model': self.net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'global_step':global_step,
                        'epoch':epoch
                        },
                       f'{train_dir}/checkpoint_epoch{epoch + 1}.pth')

