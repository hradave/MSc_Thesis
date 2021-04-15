# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 19:36:00 2021

@author: David
"""

import random
import numpy as np
import os
import shutil
import math

from timeit import default_timer as timer
import sys
import pandas as pd

# add paths containing code and data
sys.path.insert(0, '/nobackup/data/davhr856/git_clones/deep-histopath/deephistopath/wsi')
sys.path.insert(0, '/nobackup/data/davhr856/git_clones/deep-histopath')
sys.path.insert(0, '/nobackup/data/davhr856/data/TCGA')
sys.path.insert(0, '/nobackup/data/davhr856')


# number of WSI images in the training + test sets
n = int(sys.argv[1])
train_perc = 0.7
val_perc = 0.8
test_perc = 1

# file paths
path = '/nobackup/data/davhr856/data/TCGA_DX/sample/tiles'
# result path
result_path = '/nobackup/data/davhr856/data/TCGA_DX/sample/dataset'

if not os.path.exists(result_path):
    os.makedirs(result_path)
else:
    shutil.rmtree(result_path)
    os.makedirs(result_path)

os.makedirs(os.path.join(result_path,'train','GBM'))
os.makedirs(os.path.join(result_path,'validation','GBM'))
os.makedirs(os.path.join(result_path,'test','GBM'))
os.makedirs(os.path.join(result_path,'train','LGG'))
os.makedirs(os.path.join(result_path,'validation','LGG'))
os.makedirs(os.path.join(result_path,'test','LGG'))

def get_patches(file_name, path):
    patches = []
    for patch in os.listdir(path):
        if patch.startswith(file_name):
            patches.append(patch)

    return patches


classes_lookup = pd.read_csv('/nobackup/data/davhr856/data/TCGA_DX/sample/TCGA_DX_sample_classes.csv')

lgg = []
gbm = []

start_total = timer()

counter = 1

for tile_name in os.listdir(path):

    #print(counter)
    counter += 1

    #print(tile_name)

    # remove the last part of the filename
    file = tile_name[:tile_name.find('__')]
    tile_class = classes_lookup[classes_lookup['filename'] == file].values[0][1]

    if tile_class == 'GBM':
        gbm.append(file)
    else:
        lgg.append(file)

# randomly sample n training + test WSI images
lgg_dataset = random.sample(lgg, int(n/2))
lgg_train = lgg_dataset[:math.floor(train_perc*n/2)]
lgg_validation = lgg_dataset[math.floor(train_perc*n/2):math.floor(val_perc*n/2)]
lgg_test = lgg_dataset[math.floor(val_perc*n/2):math.floor(test_perc*n/2)]
print(len(lgg_train))
print(len(lgg_validation))
print(len(lgg_test))


gbm_dataset = random.sample(gbm, int(n/2))
gbm_train = gbm_dataset[:math.floor(train_perc*n/2)]
gbm_validation = gbm_dataset[math.floor(train_perc*n/2):math.floor(val_perc*n/2)]
gbm_test = gbm_dataset[math.floor(val_perc*n/2):math.floor(test_perc*n/2)]
print(len(gbm_train))
print(len(gbm_validation))
print(len(gbm_test))


for lgg_file in lgg_train:
    patches = get_patches(lgg_file, path)
    for patch in patches:
        shutil.copyfile(os.path.join(path,patch), os.path.join(result_path,'train','LGG',patch))
for lgg_file in lgg_validation:
    patches = get_patches(lgg_file, path)
    for patch in patches:
        shutil.copyfile(os.path.join(path,patch), os.path.join(result_path,'validation','LGG',patch))
for lgg_file in lgg_test:
    patches = get_patches(lgg_file, path)
    for patch in patches:
        shutil.copyfile(os.path.join(path,patch), os.path.join(result_path,'test','LGG',patch))



for gbm_file in gbm_train:
    patches = get_patches(gbm_file, path)
    for patch in patches:
        shutil.copyfile(os.path.join(path,patch), os.path.join(result_path,'train','GBM',patch))
for gbm_file in gbm_validation:
    patches = get_patches(gbm_file, path)
    for patch in patches:
        shutil.copyfile(os.path.join(path,patch), os.path.join(result_path,'validation','GBM',patch))
for gbm_file in gbm_test:
    patches = get_patches(gbm_file, path)
    for patch in patches:
        shutil.copyfile(os.path.join(path,patch), os.path.join(result_path,'test','GBM',patch))


"""
# patches
for lgg_file in lgg_chosen[:math.floor(train_perc*n/2)]:
    shutil.copyfile(os.path.join(path,lgg_file), os.path.join(result_path,'train','LGG',lgg_file))
for lgg_file in lgg_chosen[math.floor(train_perc*n/2):math.floor(val_perc*n/2)]:
    shutil.copyfile(os.path.join(path,lgg_file), os.path.join(result_path,'validation','LGG',lgg_file))
for lgg_file in lgg_chosen[math.floor(val_perc*n/2):math.floor(test_perc*n/2)]:
    shutil.copyfile(os.path.join(path,lgg_file), os.path.join(result_path,'test','LGG',lgg_file))

for gbm_file in gbm_chosen[:math.floor(train_perc*n/2)]:
    shutil.copyfile(os.path.join(path,gbm_file), os.path.join(result_path,'train','GBM',gbm_file))
for gbm_file in gbm_chosen[math.floor(train_perc*n/2):math.floor(val_perc*n/2)]:
    shutil.copyfile(os.path.join(path,gbm_file), os.path.join(result_path,'validation','GBM',gbm_file))
for gbm_file in gbm_chosen[math.floor(val_perc*n/2):math.floor(test_perc*n/2)]:
    shutil.copyfile(os.path.join(path,gbm_file), os.path.join(result_path,'test','GBM',gbm_file))
"""
# print('-------------------------- LGG -------------------------')
# print(lgg_chosen)
# print('-------------------------- GBM -------------------------')
# print(gbm_chosen)



end_total = timer()
print("Time to class ALL tiles: %s\n" % str(end_total-start_total))
