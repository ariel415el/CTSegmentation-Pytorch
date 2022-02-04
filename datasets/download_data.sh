kaggle datasets download -d andrewmvd/liver-tumor-segmentation
kaggle datasets download -d andrewmvd/liver-tumor-segmentation-part-2
unzip liver-tumor-segmentation.zip
unzip liver-tumor-segmentation-part-2.zip
mkdir ct
mv volume_pt*/* ct
mv segmentation seg
ls ct | wc -l # should be 131
ls seg | wc -l # should be 131
mkdir LiTS2017
mv ct LiTS2017
mv seg LiTS2017
rm -rf volume_pt*