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

test_data_dir = os.path.join(base_path, 'test')

### hyperparameters
batch_size = 256

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


test_datagen = ImageDataGenerator(preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input)


# initialize the testing generator
test_generator = test_datagen.flow_from_directory(
    directory=test_data_dir,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    interpolation = "bicubic",
    batch_size=batch_size)


ntest = len(test_generator.labels)
model_save_path = os.path.join(save_path, model_name)
test_steps = ntest / batch_size

# load model
start = timer()
model = keras.models.load_model(os.path.join(model_save_path, 'model'))

# run predict with MC DO
# monte carlo dropout
MC = 10
# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
test_generator.reset()
y_probas = np.stack([model.predict(test_generator, steps=test_steps) for sample in range(MC)])

# save np array
np.save(os.path.join(model_save_path, 'test_10MC'), y_probas)

# define functions for testing accuracy
def patch_accuracy(test_generator, y_mc):
    test_generator.reset()
    ntest = len(test_generator.labels)

    # get mean of MC DO runs
    y_mean = np.mean(y_mc, axis = 0)
    # class with the higher average softmax output, (shape = ntest)
    y_pred = np.argmax(y_mean, axis=1)

    # patch level accuracy
    tn, fp, fn, tp = metrics.confusion_matrix(test_generator.labels, y_pred).ravel()
    patch_accuracy = (tn + tp) / (tn + fp + fn + tp)

    return patch_accuracy

def get_slide_labels(test_generator):
    test_generator.reset()
    ntest = len(test_generator.labels)
    # slide labels
    slide_labels = []
    slides = []
    pattern = '/(.*?)/'
    for i in range(ntest):
        slide = re.search(pattern, test_generator.filenames[i]).group(1)
        label = test_generator.labels[i]
        if not slide in slides:
            slides.append(slide)
            slide_labels.append(label)
    return slide_labels


def majority_voting(test_generator, y_mc):
    test_generator.reset()
    ntest = len(test_generator.labels)

    # get mean of MC DO runs
    y_mean = np.mean(y_mc, axis = 0)
    # class with the higher average softmax output, (shape = ntest)
    y_pred = np.argmax(y_mean, axis=1)


    # slide level accuracy
    pattern = '/(.*?)/'
    current_slide = re.search(pattern, test_generator.filenames[0]).group(1)
    slide_predictions = []
    patch_predictions = []
    slide_labels = get_slide_labels(test_generator)
    for i in range(ntest):
        slide = re.search(pattern, test_generator.filenames[i]).group(1)
        if slide == current_slide:
            patch_predictions.append(y_pred[i])
        else:
            slide_predictions.append(np.mean(patch_predictions))
            patch_predictions = []
            patch_predictions.append(y_pred[i])
        current_slide = slide
        if i == ntest-1:
            slide_predictions.append(np.mean(patch_predictions))

    tn, fp, fn, tp = metrics.confusion_matrix(slide_labels, np.round(slide_predictions)).ravel()
    slide_accuracy = (tn + tp) / (tn + fp + fn + tp)

    return slide_accuracy

majority_acc = majority_voting(test_generator, y_probas)
patch_acc = patch_accuracy(test_generator, y_probas)

end = timer()

logfile_name = os.path.join(model_save_path, model_name + '_test.txt')
logfile = open(logfile_name, 'a')
logfile.write('\nPrediction time: ' + str(end - start) + '\n')
logfile.write('\nPatch accuracy: ' + str(patch_acc) + '\n')
logfile.write('\nMajority voting accuracy: ' + str(majority_acc) + '\n')
logfile.close()



print("Testing finished succesfully!")
