import torch

from datasets.slice_dataset import SlicesDataset, VolumeDataset
from models.Unet.model import UnetModel

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # dataset = SlicesDataset('datasets/processed-data-(0.5,2)')
    dataset = VolumeDataset('datasets/processed-data-(0.5,2)')

    model = UnetModel(n_channels=1, n_classes=3, bilinear=True)

    model.train(dataset, device)



