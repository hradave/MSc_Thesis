# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 19:36:00 2021

@author: David
"""

# add paths containing code and data
import sys
sys.path.insert(0, '/nobackup/data/davhr856/git_clones/deep-histopath/deephistopath/wsi')
sys.path.insert(0, '/nobackup/data/davhr856/git_clones/deep-histopath')
sys.path.insert(0, '/nobackup/data/davhr856/data/TCGA')

# input arguments
gbm_lgg = sys.argv[1]
folder_name = sys.argv[2]
file_name = sys.argv[3]

# import scripts from the deep-histopath github repo
import slide
import util
import filter
import tiles
import os

# base WSI library
import openslide
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None

import numpy as np
#%matplotlib inline
#import matplotlib.pyplot as plt
#plt.rcParams["figure.figsize"] = (10,10)

import os
# add paths containing code and data
import sys
sys.path.insert(0, '/nobackup/data/davhr856')
import csv
import math


path = os.path.join('/nobackup/data/davhr856/data/TCGA_DX/original',gbm_lgg,folder_name,file_name)
print(path)

slide_current = openslide.OpenSlide(path)
try:
    original_magnification = slide_current.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
except:
    original_magnification = 0

original_resolution = slide_current.dimensions

if int(original_magnification) == 40:
    # open slide at level 1, and downscale by 2
    factor = 2
    # read and resize image
    new_w = math.floor(slide_current.level_dimensions[1][0] / factor)
    new_h = math.floor(slide_current.level_dimensions[1][1] / factor)
    whole_slide_image = slide_current.read_region((0, 0), 1, slide_current.level_dimensions[1])
    whole_slide_image = whole_slide_image.convert("RGB")
    whole_slide_image = whole_slide_image.resize((new_w, new_h), PIL.Image.BICUBIC) # maybe overwrite whole_slide_image if RAM issues

    # save resized image
    img_path = os.path.join('/nobackup/data/davhr856/data/TCGA_DX/sample/resized',file_name[:-4] + '.png')
    whole_slide_image.save(img_path)
elif int(original_magnification) == 20:
    # open slide at level 1 and don't downscale
    whole_slide_image = slide_current.read_region((0, 0), 1, slide_current.level_dimensions[1])
    whole_slide_image = whole_slide_image.convert("RGB")
    # save resized image
    img_path = os.path.join('/nobackup/data/davhr856/data/TCGA_DX/resized',file_name[:-4] + '.png')
    whole_slide_image.save(img_path)
else:
    # don't save the image
    new_resolution = original_resolution
