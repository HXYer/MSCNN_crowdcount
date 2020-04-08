import json
import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

class Proprecessing:
    def __init__(self, _unit, _gaussian_kernel):
        self.label_path = 'data/data1917/train.json'
        self.img_path = 'stage1/train'
        self.gaussian_kernel = _gaussian_kernel
        self.unit = _unit
    
    def read_train_json(self):
        train_image_names = os.listdir(self.img_path)
        with open(self.label_path) as f:
            train_labels = json.load(f)
        train_labels = train_labels['annotations']
        train_outputs = {}
        for one_train_sample in train_labels:
            name = one_train_sample['name']
            num = one_train_sample['num']
            if num >= 20:
                continue
            annotation = one_train_sample['annotation']
            positions = []
            for a in annotation:
                pos = [a['x'], a['y']]
                positions.append(pos)
            train_outputs[name] = {'num':num, 'positions':positions}
        return train_outputs

    def map_pixels(self, img, annotations, size):
        h, w = img.shape[:-1]
        sh, sw = size / h, size / w
        pixels = np.zeros((size, size))
        for a in annotations:
            x, y = int(a[0] * sw), int(a[1] * sh)
            if y >= size or x >= size:
                print("{},{} is out of range, skipping annotation for one picture".format(x, y))
            else:
                pixels[y, x] += self.unit
        pixels = cv2.GaussianBlur(pixels, (self.gaussian_kernel, self.gaussian_kernel), 0)
        return pixels

    def get_data(self, train_outputs, size):
        train_labels = {}
        for name, value in tqdm(train_outputs.items()):
            img = cv2.imread(name)
            annotations = value['positions']
            num = value['num']
            density_map = self.map_pixels(img, annotations, size // 8)
            train_labels[name] = {'density_map':density_map, 'num':num}
        return train_labels