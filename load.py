import json
import paddle 
from PIL import Image
import numpy as np
from multiprocessing import cpu_count
import cv2
from tqdm import tqdm

class Data_reader():
    def __init__(self, _batch_size=32, _img_size=448):
        self.BATCH_SIZE = _batch_size
        self.train_path = 'stage1/train.json'
        self.test_path = 'stage1/test.json'
        self.img_size = _img_size

    def dataset_split(self, train_data):
        train_labels = {}
        test_labels = {}
        real_labels = {}
        for index, (path, value) in enumerate(train_data.items()):
            if index % 10 == 1:
                real_labels[path] = value['num']
            elif index % 10 == 2:
                test_labels[path] = value['density_map'].tolist()
            else:
                train_labels[path] = value['density_map'].tolist()

        with open('stage1/train.json', 'w') as f:
            json.dump(train_labels, f, indent=4)
        with open('stage1/test.json', 'w') as f:
            json.dump(test_labels, f, indent=4)
        with open('stage1/real.json', 'w') as f:
            json.dump(real_labels, f, indent=4)

    def mapper(self, sample):
        img, label = sample
        img = paddle.dataset.image.load_image(file=img, is_color=True)
        img = paddle.dataset.image.simple_transform(im=img, resize_size=self.img_size, crop_size=self.img_size, is_color=True, is_train=True)
        img = img.flatten().astype('float32') / 255 * 2 - 1
        label = np.expand_dims(label, axis=-1)
        label = np.transpose(label, [2, 0, 1])
        return img, label

    def load(self, path):
        def reader():
            with open(path, 'r') as f:
                data = json.load(f)
                for img, label in data.items():
                    yield (img, np.array(label))
        return paddle.reader.xmap_readers(self.mapper, reader, cpu_count(), 1024)

    def get_reader(self):
        # 读入训练集数据
        reader = self.load(self.train_path)
        reader = paddle.reader.shuffle(reader=reader, buf_size=512)
        train_reader = paddle.batch(reader=reader, batch_size=self.BATCH_SIZE)

        # 读入测试集数据
        reader = self.load(self.test_path)
        reader = paddle.reader.shuffle(reader=reader, buf_size=512)
        test_reader = paddle.batch(reader=reader, batch_size=self.BATCH_SIZE)
        return train_reader, test_reader