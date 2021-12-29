import os
import numpy as np
from torch.utils.data import Dataset
from datasets.utils import read_volume


class SlicesDataset(Dataset):
    def __init__(self, data_root):
        ct_files = sorted(os.listdir(os.path.join(data_root, 'ct')))
        segmentation_files = sorted(os.listdir(os.path.join(data_root, 'seg')))
        self.cts = []
        self.segs = []
        for ct_path, seg_path in zip(ct_files, segmentation_files):
            self.cts.append(read_volume(os.path.join(data_root, 'ct', ct_path)))
            self.segs.append(read_volume(os.path.join(data_root, 'seg', seg_path)))

        self.cts = np.concatenate(self.cts)[:, None]
        self.segs = np.concatenate(self.segs)
        print(f"Done loading {self.cts.shape[0]} slices")

    def __len__(self):
        return len(self.cts)

    def __getitem__(self, i):
        return self.cts[i], self.segs[i]

class VolumeDataset(Dataset):
    def __init__(self, data_root):
        ct_files = sorted(os.listdir(os.path.join(data_root, 'ct')))
        segmentation_files = sorted(os.listdir(os.path.join(data_root, 'seg')))
        self.cts = []
        self.segs = []
        for ct_path, seg_path in zip(ct_files, segmentation_files):
            self.cts.append(read_volume(os.path.join(data_root, 'ct', ct_path)))
            self.segs.append(read_volume(os.path.join(data_root, 'seg', seg_path)))

        print(f"Done loading {len(self.cts)} volumes")

    def __len__(self):
        return len(self.cts)

    def __getitem__(self, i):
        return self.cts[i], self.segs[i]