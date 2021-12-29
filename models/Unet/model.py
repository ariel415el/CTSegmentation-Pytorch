import torch
from torch import optim, nn
from tqdm import tqdm
import torch.nn.functional as F

from evaluate import evaluate
from models.Unet.net import UNet
from dice_score import compute_dice_loss
from utils import plot_scores, get_dataloaders


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

    def train(self, dataset, device, epochs=5, batch_size=1, lr=0.001, val_percent=0.1, eval_freq=1000, train_dir="train_dir"):
        self.net = self.net.to(device)

        train_loader, val_loader = get_dataloaders(dataset, val_percent, batch_size)

        optimizer = optim.RMSprop(self.net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
        global_step = 0

        # Begin training
        losses = []
        val_scores = []
        pbar = tqdm()
        for epoch in range(epochs):
            self.net.train()
            for b_idx, (ct_volumes, gt_volumes) in enumerate(train_loader):  #  ct_volumes.shape = (b, c, slices, H, W)
                for i in range(ct_volumes.shape[1]):
                    image = ct_volumes[:, i].to(device=device, dtype=torch.float32)
                    gt = gt_volumes[:, i].to(device=device, dtype=torch.long)

                    pred = self.net(image.unsqueeze(1))

                    loss = slice_loss_loss(pred, gt)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())
                    global_step += 1

                    pbar.update(image.shape[0])
                    pbar.set_description(f"Epoch: {epoch}/{epochs}, Step {b_idx}/{len(train_loader)}, Loss: {loss.item()}")

            # Evaluation round
            val_score = evaluate(self.net, val_loader, device, f"{train_dir}/eval-epoch-{epoch}")
            val_scores.append(val_score)
            scheduler.step(val_score)

            plot_scores(val_scores, f'{train_dir}/val_scores.png')
            plot_scores(losses, f'{train_dir}/losses.png')

            torch.save(self.net.state_dict(), f'{train_dir}/checkpoint_epoch{epoch + 1}.pth')


