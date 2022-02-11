from models.Unet.model import UnetModel
from models.VGGUnet.model import VGGUnetModel, VGGUnet2_5DModel
from models.Unet3D.model import UNet3DModel
from models.DARN.model import DARNModel
from models.Vnet.model import VnetModel
from models.Res2UNet.model import HeavyUnetModel, ResUnetModel, RecurrentUnetModel, Res2UnetModel


def get_model(config):
    if config.model_name == 'UNet':
        model = UnetModel(n_channels=1,
                                 n_classes=config.n_classes,
                                 p=64,
                                 lr=config.starting_lr,
                                 bilinear_upsample=not config.learnable_upsamples,
                                 eval_batchsize=32)
    elif config.model_name == 'VGGUNet':
        model = VGGUnetModel(n_classes=config.n_classes,
                                    lr=config.starting_lr,
                                    bilinear_upsample=not config.learnable_upsamples,
                                    eval_batchsize=8)
    elif config.model_name == 'VGGUNet2_5D':
        assert config.slice_size ==3
        model = VGGUnet2_5DModel(n_classes=config.n_classes,
                                        lr=config.starting_lr,
                                        bilinear_upsample=not config.learnable_upsamples,
                                        eval_batchsize=32)
    elif config.model_name == 'UNet3D':
        model = UNet3DModel(n_classes=config.n_classes,
                                   trilinear_upsample=not config.learnable_upsamples,
                                   slice_size=config.slice_size,
                                   p=32,
                                   lr=config.starting_lr)

    elif config.model_name == 'DARN':
        model = DARNModel(n_classes=config.n_classes,
                                   trilinear_upsample=not config.learnable_upsamples,
                                   slice_size=config.slice_size,
                                   p=8,
                                   lr=config.starting_lr)

    elif config.model_name == 'HeavyUNet':
        model = HeavyUnetModel(n_channels=1,
                                 n_classes=config.n_classes,
                                 p=48,
                                 lr=config.starting_lr,
                                 eval_batchsize=32)
    elif config.model_name == 'ResUNet':
        model = ResUnetModel(n_channels=1,
                                  n_classes=config.n_classes,
                                  p=48,
                                  lr=config.starting_lr,
                                  eval_batchsize=32)
    elif config.model_name == 'RecurrentUNet':
        model = RecurrentUnetModel(n_channels=1,
                                      n_classes=config.n_classes,
                                      p=48,
                                      lr=config.starting_lr,
                                      eval_batchsize=32)
    elif config.model_name == 'Res2Unet':
        model = Res2UnetModel(n_channels=1,
                                      n_classes=config.n_classes,
                                      p=32,
                                      lr=config.starting_lr,
                                      eval_batchsize=32)

    else:
        raise Exception("No such train method")

    return model