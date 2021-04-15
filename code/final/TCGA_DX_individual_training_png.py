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
from timeit import default_timer as timer


path = os.path.join('/nobackup/data/davhr856/data/TCGA_DX/original',gbm_lgg,folder_name,file_name)
print(path)
start = timer()
start_read = timer()
slide_current = openslide.OpenSlide(path)
end_read = timer()
print("Time open: %s\n" % str(end_read-start_read))
try:
    original_magnification = slide_current.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
except:
    original_magnification = 0

original_resolution = slide_current.dimensions

if int(original_magnification) == 40:
    # open slide at level 0, and downscale by 2
    factor = 2
    # read and resize image
    start_png = timer()
    new_w = math.floor(slide_current.level_dimensions[0][0] / factor)
    new_h = math.floor(slide_current.level_dimensions[0][1] / factor)
    whole_slide_image = slide_current.read_region((0, 0), 0, slide_current.level_dimensions[0])
    whole_slide_image = whole_slide_image.convert("RGB")
    whole_slide_image = whole_slide_image.resize((new_w, new_h), PIL.Image.BICUBIC) # maybe overwrite whole_slide_image if RAM issues
    end_png = timer()
    print("Time to png: %s\n" % str(end_png-start_png))
    # save resized image
    start_save = timer()
    img_path = os.path.join('/nobackup/data/davhr856/data/TCGA_DX/dev/resized', gbm_lgg + '_' + file_name[:-4] + '.png')
    whole_slide_image.save(img_path)
    end_save = timer()
    print("Time save: %s\n" % str(end_save-start_save))
elif int(original_magnification) == 20:
    # open slide at level 0 and don't downscale
    start_png = timer()
    whole_slide_image = slide_current.read_region((0, 0), 0, slide_current.level_dimensions[0])
    whole_slide_image = whole_slide_image.convert("RGB")
    end_png = timer()
    print("Time to png: %s\n" % str(end_png-start_png))
    # save image
    start_save = timer()
    img_path = os.path.join('/nobackup/data/davhr856/data/TCGA_DX/dev/resized', gbm_lgg + '_' + file_name[:-4] + '.png')
    whole_slide_image.save(img_path)
    end_save = timer()
    print("Time save: %s\n" % str(end_save-start_save))
else:
    # don't save the image
    new_resolution = original_resolution

end = timer()
print("Time: %s\n" % str(end-start))
