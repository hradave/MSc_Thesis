# -*- coding: utf-8 -*-
"""
Created on Mon March 21 19:36:00 2021

@author: David
"""

# add paths containing code and data
import os
import sys
sys.path.insert(0, '/nobackup/data/davhr856')
sys.path.insert(0, '/nobackup/data/davhr856/data/TCGA_DX')

import math
import numpy as np
from timeit import default_timer as timer
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = None


base_path = '/nobackup/data/davhr856/data/TCGA_DX/modelling'
old_path = os.path.join(base_path, "dataset", "medium")
new_path = os.path.join(base_path, "dataset_224", "medium")

counter = 0
resizing_time = []
# loop through all folders and files
for datasplit in os.listdir(old_path):
    for gbm_lgg in os.listdir(os.path.join(old_path, datasplit)):
        for file_folder in os.listdir(os.path.join(old_path, datasplit, gbm_lgg)):
            for patch in os.listdir(os.path.join(old_path, datasplit, gbm_lgg, file_folder)):
                counter += 1
                print(counter)
                print(patch)
                start = timer()
                # load patch image
                im = Image.open(os.path.join(old_path, datasplit, gbm_lgg, file_folder, patch))
                new_w = 224
                new_h = 224
                im_resized = im.resize((new_w, new_h), PIL.Image.BICUBIC)

                # save resized patch
                save_path = os.path.join(new_path, datasplit, gbm_lgg, file_folder)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                im_resized.save(os.path.join(save_path, patch))

                end = timer()
                resizing_time.append(end - start)
                print("Time to resize: %s\n" % str(end - start))

print("Average time to resize one patch: %s\n" % str(np.mean(resizing_time)))
