import os
import warnings

# Ignore FutureWarning from numpy
warnings.simplefilter(action='ignore', category=FutureWarning)

import keras.backend as K
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
#os.environ["CUDA_VISIBLE_DEVICES"]="1";

# Check what kind of device we are using for calculations
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

tf.test.gpu_device_name()

import timeit

#device_name = tf.test.gpu_device_name()
#if device_name != '/device:GPU:1':
#  print(
#      '\n\nThis error most likely means that this notebook is not '
#      'configured to use a GPU.  Change this in Notebook Settings via the '
#      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
#  raise SystemError('GPU device not found')


# Perform convolutions using the GPU (graphics card)

def gpu0():
  with tf.device('/device:GPU:0'):
    random_image_gpu = tf.random.normal((1000, 100, 100, 3))
    net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
    return tf.math.reduce_sum(net_gpu)
"""

def gpu1():
  with tf.device('/device:GPU:1'):
    random_image_gpu = tf.random.normal((1000, 100, 100, 3))
    net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
    return tf.math.reduce_sum(net_gpu)

# We run each op once to warm up; see: https://stackoverflow.com/a/45067900
"""
gpu0()
#gpu1()


print('GPU0 (s):')
gpu0_time = timeit.timeit('gpu0()', number=1000, setup="from __main__ import gpu0")
print(gpu0_time)
"""
print('GPU1 (s):')
gpu1_time = timeit.timeit('gpu1()', number=1000, setup="from __main__ import gpu1")
print(gpu1_time)
"""