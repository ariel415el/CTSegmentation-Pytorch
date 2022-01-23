import torch
mode = "train"

# model configs
model_name = 'UNet3D'
n_classes = 2

# data config
data_path = 'datasets/LiST2017_(MS-(3, 15, 15)_MM-2_Crop-CL-1_margins-(1, 1, 1)_OB-0.5_MD-11)'
val_cases = [19, 76, 50, 92, 88, 122, 100, 71, 23, 28, 9, 119, 39]
resize = 128
augment_data = False
ignore_background = False

# other configs
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


# Train configs
num_workers = 1
train_steps = 100000
eval_freq = 500