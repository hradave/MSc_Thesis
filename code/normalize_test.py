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

# increase pixel limit of opencv
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,32).__str__()
import staintools
# import scripts from the deep-histopath github repo
import slide
import util
import filter
import tiles
import os

import numpy as np
#%matplotlib inline
#import matplotlib.pyplot as plt
#plt.rcParams["figure.figsize"] = (10,10)

image13 = staintools.read_image(slide.get_training_image_path(13))
print(image13.nbytes/1024/1024/1024)
print(image13.dtype)
image1 = staintools.read_image(slide.get_training_image_path(1))
print(image1.nbytes/1024/1024/1024)
print(image1.dtype)

target = image13
to_transform = image1

# Stain normalize for unfiltered WSI
normalizer = staintools.StainNormalizer(method='vahadane')
normalizer.fit(target)
transformed = normalizer.transform(to_transform)
filter.save_display(save = True, display = False, info = None, np_img = transformed, slide_num = 1, filter_num = 1, display_text = "Transformed", file_text = "rgb")
