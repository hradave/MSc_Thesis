# -*- coding: utf-8 -*-
"""
Created on Mon March 24 19:36:00 2021

@author: David
"""

# add paths containing code and data
import os
import sys
sys.path.insert(0, '/home/davhr856/thesis/saved_models')
sys.path.insert(0, '/nobackup/data/davhr856')
sys.path.insert(0, '/nobackup/data/davhr856/data/TCGA_DX')

gpu = str(sys.argv[1])
model_name = str(sys.argv[2])
#model_name = 'resnet_block4_finetuned'
patch_size = str(sys.argv[3])

import math
import numpy as np
from timeit import default_timer as timer
import warnings
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import re
from sklearn import metrics

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]=gpu;

# import the necessary packages
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import classification_report

# Allow growth of GPU memory, otherwise it will always look like all the memory is being used
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Ignore FutureWarning from numpy
warnings.simplefilter(action='ignore', category=FutureWarning)

base_path = '/nobackup/data/davhr856/data/TCGA_DX/modelling/dataset_224/' + patch_size
save_path = os.path.join('/home/davhr856/thesis/saved_models', patch_size)

# create ImageDataGenerators
# dimensions of our images
img_width, img_height = 224, 224

#valid_data_dir = os.path.join(base_path, 'validation')
test_data_dir = os.path.join(base_path, 'test_extra')

### hyperparameters
batch_size = 64

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


test_datagen = ImageDataGenerator(preprocessing_function = tf.keras.applications.resnet.preprocess_input)


# initialize the testing generator
test_generator = test_datagen.flow_from_directory(
    directory=test_data_dir,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    interpolation = "bicubic",
    batch_size=batch_size)

"""
# initialize the validation generator
valid_generator = test_datagen.flow_from_directory(
    directory=valid_data_dir,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    interpolation = "bicubic",
    batch_size=batch_size)

print(valid_generator.class_indices)
#print(test_generator.class_indices)
"""

model_save_path = os.path.join(save_path, model_name)

#nvalid = len(valid_generator.labels)
ntest = len(test_generator.labels)
#validation_steps = nvalid / batch_size
test_steps = ntest / batch_size

# load model
start = timer()
model = keras.models.load_model(os.path.join(model_save_path, 'model'))

# run predict with MC DO
# monte carlo dropout
MC = 30
# reset the testing generator and then use our trained model to
# make predictions on the data
#print("[INFO] predicting on validation set...")
#valid_generator.reset()
#y_probas_valid = np.stack([model.predict(valid_generator, steps=validation_steps) for sample in range(MC)])

print("[INFO] predictiong on test set...")
test_generator.reset()
y_probas_test = np.stack([model.predict(test_generator, steps=test_steps) for sample in range(MC)])

# save np arrays
#np.save(os.path.join(model_save_path, 'valid_30MC'), y_probas_valid)
np.save(os.path.join(model_save_path, 'test_30MC'), y_probas_test)

print("Testing finished succesfully!")
