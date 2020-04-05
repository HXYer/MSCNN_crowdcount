import json
import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

def read_train_json():
    train_image_names = os.listdir('stage1/train')
    with open('data.json') as f:
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

def map_pixels(img, annotations, size):
    gaussian_kernel = 15
    h, w = img.shape[:-1]
    sh, sw = size / h, size / w
    pixels = np.zeros((size, size))

    for a in annotations:
        x, y = int(a[0] * sw), int(a[1] * sh)
        if y >= size or x >= size:
            print("{},{} is out of range, skipping annotation for one picture".format(x, y))
        else:
            pixels[y, x] += 1

    pixels = cv2.GaussianBlur(pixels, (gaussian_kernel, gaussian_kernel), 0)
    return pixels

def get_data(train_outputs, size):
    train_labels = {}
    for name, value in tqdm(train_outputs.items()):
        img = cv2.imread(name)
        annotations = value['positions']
        density_map = map_pixels(img, annotations, size // 4)
        density_map = np.expand_dims(density_map, axis=-1)
        train_labels[name] = density_map
    return train_labels
	
def save_processed_data(data):
	train_labels = {}
	test_labels = {}
	real_labels = {}
	for index, (path, density_map) in enumerate(train_data.items()):
		if index % 10 == 1:
			real_labels[path] = density_map.tolist()
		elif index % 10 == 2:
			test_labels[path] = density_map.tolist()
		else:
			train_labels[path] = density_map.tolist()

	with open('stage1/train.json', 'w') as f:
		json.dump(train_labels, f, indent=4)
	with open('stage1/test.json', 'w') as f:
		json.dump(test_labels, f, indent=4)
	with open('stage1/real.json', 'w') as f:
		json.dump(real_labels, f, indent=4)
	print('achieve')

if __name__ == '__main__':
	train_outputs = read_train_json()
	train_data = get_data(train_outputs, 224)
	save_processed_data(train_data)