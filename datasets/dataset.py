import numpy as np
import os
from torch.utils.data import Dataset
import torch
import csv
#from extract_lips import get_lips

class MyDataset(Dataset):
    labels = ["ama", "recitation", "tsun", "normal", "sexy"]

    def __init__(self, dataroot, datafile, rand_pad):
        self.rand_pad = rand_pad
        self.files = []
        with open(os.path.join(dataroot, datafile), "r") as f:
            self.files = [path.rstrip() for path in f.readlines()]
    
    def __getitem__(self, idx):
        data_path = self.files[idx]
        path_sep = data_path.split("\\")
        if "" in path_sep:
            path_sep.remove("")
        rand = self._load_rand(data_path)
        rand = self._rand_padding(rand, self.rand_pad)
        kind = self._load_label(path_sep[-2])

        return {
            'rand': torch.FloatTensor(rand),
            'label': torch.LongTensor(kind),
        }
    
    def __len__(self):
        return len(self.files)
    
    def _load_rand(self, p):
        rand = []
        with open(p, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                row.remove('')
                number = list(map(float, row))
                number = np.array(number)
                number = number.reshape(number.shape[0]//2, 2)
                number[:, 1] *= -1
                rand.append(np.linalg.norm(number[48:, :]))
        rand = np.array(rand)
        rand_min = np.min(rand)
        rand_max = np.max(rand)
        rand = (rand - rand_min) \
            / (rand_max - rand_min)
        #         rand.append(number)
        # rand = np.array(rand)
        # rand = [rand[i]-rand[i+1] for i in range(rand.shape[0]-1)]
        # rand = np.stack(rand, axis=0).astype(np.float32)
        return rand
    
    def _load_label(self, kind):
        result = np.zeros(1,)
        result[0] = MyDataset.labels.index(kind)
        return result
    
    def _rand_padding(self, array, length):
        size = length - array.shape[0]
        pd_arr = np.zeros([size,])
        return np.hstack([array, pd_arr])


if __name__ == "__main__":
    dataset = MyDataset(
        "data", "train_path.txt", 1112
    )
    data = dataset.__getitem__(0).get("rand")
    print(data)
