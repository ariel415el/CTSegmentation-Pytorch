# MedicalImageSegmentation-Pytorch
Compare methods of segmenting volumatric CT data of liver tumors

# Steps for training/testing pretrained models on LiTS2017
This repo segments CT in two steps:
- 1: Localization: A liver-localization CNN that gets the entire CT volume roughly segments the liver area.
- 2: Segmentation: Two additional CNNs or a single multiclass CNN is trained on volumes cropped around the liver to fine-segment the liver and tumors

## Step 1: Download LiTS dataset
Download dataset ~50GB from kaggle 
- https://www.kaggle.com/andrewmvd/liver-tumor-segmentation
- https://www.kaggle.com/andrewmvd/liver-tumor-segmentation-part-2

## Step 2: Preprocess data
In this step we prepare the data for training.
- 1: Loclization CNN does not need the full resolution of the volumes so we create a light low resolution replica of the dataset
- 2: We crop volumes around the liver for the segmentation network.

Do the same for both the train and the test folders.

Optinaly you can instruct the preprocess to normalize the axial axis of the volumes to have the same size in Mms but I find it unecessary.

```
cd datasets
python3 preprocess_dat.py <LiTS root dir>
```
You should now have tow additional folders in the 'datasets' directory with the same file structure as the original dataset

## Step 3 (Optional) Train model:
Edit main.py and set preprocessed data path, model name, etc
run 
```
pytnon3 main.py
```

# Or Download pretrained models
Download the folders from 
https://drive.google.com/drive/folders/1y-fZWUCsae2gzSXOkeVutH925Q0URfTN?usp=sharing
and place it in the main folder

# Test trained model

# Inference on test data


# Experiment with classical method
set the path in classic_methods/thresholding.py to unprocessed LiTS root dir
run
```
cd classic_methods
python3 thresholding.py
```


# Credits
I consoluted the follwing repos:
- https://github.com/milesial/Pytorch-UNet
- https://github.com/mattmacy/vnet.pytorch
- https://github.com/assassint2017/MICCAI-LITS2017
- https://github.com/nexus-kgp/adseg
- https://github.com/navamikairanda/R2U-Net
