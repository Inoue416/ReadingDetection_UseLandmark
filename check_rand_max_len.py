import csv
import numpy as np
import os

def load_rand(p):
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

files = ["train_path.txt", "val_path.txt"]
root = "data"
max_len = 0
for file in files:
    paths = []
    with open(os.path.join(root, file), "r") as f:
        for i in f.readlines():
            buf = load_rand(i.rstrip())
            if (max_len < buf.shape[0]):
                max_len = buf.shape[0]
print(max_len)
        