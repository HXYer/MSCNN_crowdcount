import paddle.fluid as fluid
import numpy as np


class Model():
    def __init__(self):
        self.img_size = [3, 224, 224]
        self.label_size = [1, 28, 28]
    
    def _conv2d_same(self, ipt, num):
        x = fluid.layers.conv2d(input=ipt, num_filters=num, filter_size=3, padding=1, act='relu')
        return x
    
    def _conv2d_dilation(self, ipt, num):
        x = fluid.layers.conv2d(input=ipt, num_filters=num, filter_size=3, padding=2, dilation=2, act='relu')
        return x
    
    def _maxpooling_half(self, ipt):
        x = fluid.layers.pool2d(input=ipt, pool_size=2, pool_stride=2, pool_type='max')
        return x

    def VGG(self, x):
        print("input.shape=", x.shape)
        x = self._conv2d_same(x, 64)
        x = fluid.layers.batch_norm(input=x)
        x = self._conv2d_same(x, 64)
        x = self._maxpooling_half(x)
        x = self._conv2d_same(x, 128)
        x = fluid.layers.batch_norm(input=x)
        x = self._conv2d_same(x, 128)
        x = self._maxpooling_half(x)
        x = self._conv2d_same(x, 256)
        x = fluid.layers.batch_norm(input=x)
        x = self._conv2d_same(x, 256)
        x = fluid.layers.batch_norm(input=x)
        x = self._conv2d_same(x, 256)
        x = self._maxpooling_half(x)
        x = self._conv2d_same(x, 512)
        x = self._conv2d_same(x, 512)
        x = self._conv2d_same(x, 512)
        x = fluid.layers.batch_norm(input=x)
        print("VGG output.shape=", x.shape)
        return x

    def dilations_cnn(self, x):
        x = self._conv2d_dilation(x, 512)
        x = self._conv2d_dilation(x, 512)
        x = self._conv2d_dilation(x, 512)
        x = self._conv2d_dilation(x, 256)
        x = self._conv2d_dilation(x, 128)
        x = self._conv2d_dilation(x, 64)
        x = fluid.layers.conv2d(input=x, num_filters=1, filter_size=1, act=None)
        print("final output.shape=", x.shape)
        return x

    def generator(self):
        image = fluid.layers.data(name='image',shape=self.img_size,dtype='float32')
        label = fluid.layers.data(name='label',shape=self.label_size,dtype='float32')
        predict = self.VGG(image)
        predict = self.dilations_cnn(predict)
        return image, label, predict
