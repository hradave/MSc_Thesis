# -*- coding: utf-8 -*-
"""
Created on Mon March 18 19:36:00 2021

@author: David
"""

# add paths containing code and data
import os
import sys
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
if not os.path.exists(save_path):
    os.makedirs(save_path)

# create ImageDataGenerators
# dimensions of our images
img_width, img_height = 224, 224

train_data_dir = os.path.join(base_path, 'train')
validation_data_dir = os.path.join(base_path, 'validation')
#test_data_dir = os.path.join(base_path, 'test')

### hyperparameters
epochs = 40
batch_size = 64

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(preprocessing_function = tf.keras.applications.resnet.preprocess_input,
                                    horizontal_flip = True,
                                    vertical_flip = True)
test_datagen = ImageDataGenerator(preprocessing_function = tf.keras.applications.resnet.preprocess_input)

train_generator = train_datagen.flow_from_directory(
    directory=train_data_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    interpolation = "bicubic",
    seed=42
)

valid_generator = test_datagen.flow_from_directory(
    directory=validation_data_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    interpolation = "bicubic",
    seed=42
)

# initialize the testing generator
"""
test_generator = test_datagen.flow_from_directory(
    directory=test_data_dir,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    interpolation = "bicubic",
    batch_size=batch_size)
"""

ntrain = len(train_generator.labels)
nvalid = len(valid_generator.labels)
#ntest = len(test_generator.labels)
steps_per_epoch = ntrain / batch_size
validation_steps = nvalid / batch_size
#test_steps = ntest / batch_size


### Build model
baseModel = tf.keras.applications.ResNet50(
    include_top=False,
    weights=None,
    input_shape=(224, 224, 3)
)

#unfreeze all layers in baseModel
baseModel.trainable = True

# create a new model on top
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
inputs = keras.Input(shape=(224, 224, 3))
# create a new model on top
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
headModel = baseModel(inputs)

headModel = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = tf.keras.layers.Flatten(name="flatten")(headModel)
headModel = tf.keras.layers.Dropout(0.5)(headModel, training = True)
headModel = tf.keras.layers.Dense(100, activation="relu")(headModel)
headModel = tf.keras.layers.Dropout(0.5)(headModel, training = True)
outputs = tf.keras.layers.Dense(2, activation="softmax")(headModel)

model = Model(inputs=inputs, outputs=outputs)
# compile the model
model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.000001), metrics=["accuracy"])

print(model.summary())
model.summary()

start = timer()
H = model.fit(train_generator, steps_per_epoch = steps_per_epoch, epochs = epochs,
                       verbose = 2, validation_steps = validation_steps,
                       validation_data = valid_generator,
                       use_multiprocessing = True,
                       workers = 6)
end = timer()

model_save_path = os.path.join(save_path, model_name)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

logfile_name = os.path.join(model_save_path, model_name + '.txt')

logfile = open(logfile_name, 'a')
logfile.write('\nModel name: ' + str(model_name))
logfile.write('\nPatch size: ' + str(patch_size))
logfile.write('\nTrain top model: ' + str(end - start) + '\n')
logfile.close()

# save history to log
with open(logfile_name, 'a') as f:
    print(H.history, file=f)

# save model
model.save(os.path.join(model_save_path, "model"))


print("Training finished succesfully!")

#### Fine-tune model
"""
train_generator.reset()
valid_generator.reset()

#model_save_path = os.path.join(save_path, model_name)
#if not os.path.exists(model_save_path):
#    os.makedirs(model_save_path)

#logfile_name = os.path.join(model_save_path, model_name + '.txt')

model.trainable = True

# unfreeze last convolutional block
#for layer in model.layers:
#    if layer.name == 'conv4_block1_1_conv':
#        break
#    layer.trainable = False

# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
#baseModel.trainable = True

# It's important to recompile your model after you make any changes
# to the `trainable` attribute of any inner layer, so that your changes
# are take into account
model.compile(optimizer=keras.optimizers.Adam(0.00001),  # Very low learning rate
              loss="binary_crossentropy",
              metrics=["accuracy"])

start = timer()
H_tuned = model.fit(train_generator, steps_per_epoch = steps_per_epoch, epochs = epochs,
                       verbose = 2, validation_steps = validation_steps,
                       validation_data = valid_generator,
                       use_multiprocessing = True,
                       workers = 6)

end = timer()
logfile = open(logfile_name, 'a')
logfile.write('\nFine-tune model: ' + str(end - start) + '\n')
logfile.close()

with open(logfile_name, 'a') as f:
    print(H_tuned.history, file=f)


# save model
model.save(os.path.join(model_save_path, "model"))
"""
