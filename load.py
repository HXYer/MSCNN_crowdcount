import json
import paddle 
from PIL import Image
import numpy as np
from multiprocessing import cpu_count
import cv2
from tqdm import tqdm

def train_mapper(sample):
    img, label = sample
    img = cv2.imread(img)
    img = cv2.resize(img, (224, 224))
    img = np.transpose(img, [2, 0, 1])
    # img = paddle.dataset.image.load_image(file=img, is_color=True)
    # img = paddle.dataset.image.simple_transform(im=img, resize_size=224, crop_size=224, is_color=True, is_train=True)
    img = img.flatten().astype('float32') / 255 * 2 - 1
    label = np.transpose(label, [2, 0, 1])
    return img, label
def train_r(path):
    def reader():
        with open(path, 'r') as f:
            data = json.load(f)
            for img, label in data.items():
                yield (img, np.array(label))
    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), 1024)

def test_mapper(sample):
    img, label = sample
    img = cv2.imread(img)
    img = cv2.resize(img, (224, 224))
    img = np.transpose(img, [2, 0, 1])
    # img = paddle.dataset.image.load_image(file=img, is_color=True)
    # img = paddle.dataset.image.simple_transform(im=img, resize_size=224, crop_size=224, is_color=True, is_train=True)
    img = img.flatten().astype('float32') / 255 * 2 - 1
    label = np.transpose(label, [2, 0, 1])
    return img, label
def test_r(path):
    def reader():
        with open(path, 'r') as f:
            data = json.load(f)
            for img, label in data.items():
                yield (img, np.array(label))
    return paddle.reader.xmap_readers(test_mapper, reader, cpu_count(), 1024)
	

def get_reader(BATCH_SIZE):
	# 读入训练集数据
	reader = train_r('stage1/train.json')
	reader = paddle.reader.shuffle(reader=reader, buf_size=512)
	train_reader = paddle.batch(reader=reader, batch_size=BATCH_SIZE)

	# 读入测试集数据
	reader = test_r('stage1/test.json')
	reader = paddle.reader.shuffle(reader=reader, buf_size=512)
	test_reader = paddle.batch(reader=reader, batch_size=BATCH_SIZE)
	return train_reader, test_reader
	
	
if __name__ == '__main__':
	a,b = get_reader(16)
	print('ok')