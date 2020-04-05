import paddle.fluid as fluid
import numpy as np


def MSB_four(ipt, filters, name):
    x1 = fluid.layers.conv2d(input=ipt, 
                                num_filters=filters, 
                                filter_size=9,
                                stride=1, 
                                padding=4, 
                                name='x1_'+name, act='relu')
    x2 = fluid.layers.conv2d(input=x1, 
                                num_filters=filters, 
                                filter_size=7,
                                stride=1, 
                                padding=3, 
                                name='x2_'+name, act='relu')
    x3 = fluid.layers.conv2d(input=x2, 
                                num_filters=filters, 
                                filter_size=5,
                                stride=1, 
                                padding=2, 
                                name='x3_'+name, act='relu')
    x4 = fluid.layers.conv2d(input=x3, 
                                num_filters=filters, 
                                filter_size=3,
                                stride=1, 
                                padding=1, 
                                name='x4_'+name, act='relu')
    x = fluid.layers.concat(input=[x1, x2, x3, x4], axis=1)
    x = fluid.layers.batch_norm(input=x)
    x = fluid.layers.relu(x)
    return x

def MSB_three(ipt, filters, name):
    x1 = fluid.layers.conv2d(input=ipt, 
                                num_filters=filters, 
                                filter_size=7,
                                stride=1, 
                                padding=3, 
                                name='x1_'+name, act=None)
    x2 = fluid.layers.conv2d(input=x1, 
                                num_filters=filters, 
                                filter_size=5,
                                stride=1, 
                                padding=2, 
                                name='x2_'+name, act=None)
    x3 = fluid.layers.conv2d(input=x2, 
                                num_filters=filters, 
                                filter_size=3,
                                stride=1, 
                                padding=1, 
                                name='x3_'+name, act=None)
    x = fluid.layers.concat(input=[x1, x2, x3], axis=1)
    x = fluid.layers.batch_norm(input=x)
    x = fluid.layers.relu(x)
    return x

def MSCNN(ipt):
    conv_1 = fluid.layers.conv2d(input=ipt, 
                                num_filters=64, 
                                filter_size=9,
                                stride=1, 
                                padding=4, 
                                name='conv_1', act='relu')
    
    MSB_1 = MSB_four(conv_1, 16, 'first')

    pool_1 = fluid.layers.pool2d(input=MSB_1, 
                                pool_size=2, 
                                pool_padding=0,
                                pool_stride=2,
                                name='pool_1', pool_type='max')
    
    MSB_2 = MSB_four(pool_1, 32, 'second')

    MSB_3 = MSB_four(MSB_2, 32, 'third')

    pool_2 = fluid.layers.pool2d(input=MSB_3, 
                                pool_size=2, 
                                pool_padding=0, 
                                pool_stride=2,
                                name='pool_2', pool_type='max')

    MSB_4 = MSB_three(pool_2, 64, 'fourth')

    MSB_5 = MSB_three(MSB_4, 64, 'fifth')

    conv_2 =  fluid.layers.conv2d(input=MSB_5, 
                                num_filters=1000, 
                                filter_size=1,
                                stride=1, 
                                padding=0, 
                                name='conv_2', act='relu')
    
    out =  fluid.layers.conv2d(input=conv_2, 
                                num_filters=1, 
                                filter_size=1,
                                stride=1, 
                                padding=0, 
                                name='out', act='sigmoid')
    print(conv_1.shape)
    print(MSB_1.shape)
    print(pool_1.shape)
    print(MSB_2.shape)
    print(MSB_3.shape)
    print(pool_2.shape)
    print(MSB_4.shape)
    print(MSB_5.shape)
    print(conv_2.shape)
    print(out.shape)
    return out
	
	
if __name__ == '__main__':
	image = fluid.layers.data(name='image', shape=(3, 224, 224), dtype='float32')
	label = fluid.layers.data(name='label', shape=(1, 56, 56), dtype='float32')
	out = MSCNN(image)