import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
from cvtransforms import *
import torch
import glob
import re
import copy
import json
import random
import editdistance
#from extract_lips import get_lips



class MyDataset(Dataset):
    labels = ["ama", "recitation", "tsun", "normal", "sexy", "whis"]

    def __init__(self, dataroot, anno_root, datafile, rand_pad, txt_pad, phase):
        self.anno_root = anno_root
        self.phase = phase
        self.rand_pad = rand_pad
        self.txt_pad = txt_pad
        self.files = []
        with open(os.path.join(dataroot, datafile), "r") as f:
            self.files = [path.rstrip() for path in f.readlines()]
    
    def __getitem__(self, idx):
        data_path = self.files[idx]
        path_sep = data_path.split("/")
        if "" in path_sep:
            path_sep.remove("")
        