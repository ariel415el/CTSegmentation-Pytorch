# MedicalImageSegmentation-Pytorch
Compare methods of segmenting volumatric CT data of liver tumors

# download LiTS dataset
Download dataset ~50GB from kaggle 
- https://www.kaggle.com/andrewmvd/liver-tumor-segmentation
- https://www.kaggle.com/andrewmvd/liver-tumor-segmentation-part-2

# Preprocess data
Crop around liver (drop non-relevant areas) to avoid loading entire dtaset for training. 
```
cd datasets
python3 preprocess_dat.py <LiTS root dir>
```

# Train model:
Edit main.py and set preprocessed data path, model name, etc
run 
```
pytnon3 main.py
```

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