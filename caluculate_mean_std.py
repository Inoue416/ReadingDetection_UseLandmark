import os
import numpy as np
import csv
from copy import deepcopy

DATA_ROOT = './data'
PATH_FILES = ['train_path.txt', 'val_path.txt']

def load_paths(file_path):
    paths = []
    with open(file_path, 'r') as f:
        paths = [s.rstrip() for s in f.readlines()]
    return paths


def load_data(path):
    rand = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row.remove('')
            number = list(map(float, row))
            number = np.array(number)
            number = number.reshape(number.shape[0]//2, 2)
            rand.append(number)
    rand = np.array(rand)
    return rand


if __name__ == '__main__':
    train_paths = load_paths(os.path.join(DATA_ROOT, PATH_FILES[0]))
    val_paths = load_paths(os.path.join(DATA_ROOT, PATH_FILES[1]))
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    print('train data load.')
    step = len(train_paths)//10
    count = 0
    progress = 0
    bar = '#' * progress + " " * (10-progress)
    for train_path in train_paths:
        data = load_data(train_path)
        train_x += deepcopy(data[:, :, 0]).reshape(-1).tolist()
        train_y += deepcopy(data[:, :, 1]).reshape(-1).tolist()
        count += 1
        print(f"\r\033[K[{bar}] {progress/10*100:.02f}% ({progress}/{10})", end="")
        if count % step == 0:
            progress += 1
            bar = '#' * progress + " " * (10-progress)

    print('\nval data load.')
    step = len(val_paths) // 10
    count = 0
    progress = 0
    bar = '#' * progress + " " * (10-progress)
    for val_path in val_paths:
        data = load_data(val_path)
        val_x += deepcopy(data[:, :, 0]).reshape(-1).tolist()
        val_y += deepcopy(data[:, :, 1]).reshape(-1).tolist()
        count += 1
        print(f"\r\033[K[{bar}] {progress/10*100:.02f}% ({progress}/{10})", end="")
        if count % step == 0:
            progress += 1
            bar = '#' * progress + " " * (10-progress)
    print()
    total_x = []
    total_x += deepcopy(train_x)
    total_x += deepcopy(val_x)
    total_y = []
    total_y += deepcopy(train_y)
    total_y += deepcopy(val_y)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    val_x = np.array(val_x)
    val_y = np.array(val_y)
    total_x = np.array(total_x)
    total_y = np.array(total_y)

    print('train x mean: ', train_x.mean())
    print('train y mean: ', train_y.mean())
    print('\ntrain x std: ', train_x.std(ddof=1))
    print('train y std: ', train_y.std(ddof=1))

    print("\nval x mean: ", val_x.mean())
    print('val y mean: ', val_y.mean())
    print('\nval x std: ', val_x.std(ddof=1))
    print('val y std: ', val_y.std(ddof=1))

    print('\ntotal x mean: ', total_x.mean())
    print('total y mean: ', total_y.mean())
    print('\ntotal x std: ', total_x.std(ddof=1))
    print('total y std: ', total_y.std(ddof=1))