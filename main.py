import torch
from models.Unet.model import UnetModel
from models.Vnet.model import VnetModel

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    data_path = 'datasets/processed-data-(0.5,2)'

    # model = UnetModel(n_channels=1, n_classes=3, bilinear=True)
    model = VnetModel(n_channels=1, n_classes=3)

    model.train(data_path, device, epochs=20)



