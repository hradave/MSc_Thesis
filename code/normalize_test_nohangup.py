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

import staintools
# import scripts from the deep-histopath github repo
import slide
import util
import filter
import tiles
import os

import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)

from datetime import datetime

now = datetime.now()
until = datetime(2021, 2, 11, 16, 30)

while now.time() < until.time():
    now = datetime.now()
    
    image2 = staintools.read_image(slide.get_training_image_path(4))
    image3 = staintools.read_image(slide.get_training_image_path(6))

    target3 = image3
    to_transform3 = image2

    # Stain normalize for unfiltered WSI
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(target3)
    transformed3 = normalizer.transform(to_transform3)

    #image1 = staintools.read_image(slide.get_training_image_path(2))


