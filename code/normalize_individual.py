# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 19:36:00 2021

@author: David
"""

# import
import numpy as np
import sys
import os
import openslide
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None
from timeit import default_timer as timer
import math

# add paths containing code and data
sys.path.insert(0, '/nobackup/data/davhr856/git_clones/deep-histopath/deephistopath/wsi')
sys.path.insert(0, '/nobackup/data/davhr856/git_clones/deep-histopath')
sys.path.insert(0, '/nobackup/data/davhr856/data/TCGA')
sys.path.insert(0, '/nobackup/data/davhr856')
sys.path.insert(0, '/nobackup/data/davhr856/git_clones/HEnorm_python')

file_name = sys.argv[1]

# import scripts from the deep-histopath github repo
import slide
import util
import filter
import tiles

import normalizeStaining

# file paths
path = '/nobackup/data/davhr856/data/TCGA_DX/sample/filtered_before_normalize'
# normalized image path
result_path = os.path.join('/nobackup/data/davhr856/data/TCGA_DX/sample/normalized')

counter = 1
start_total = timer()

print(file_name)
print(counter)
counter += 1
start = timer()

# read png image file as np
img = slide.open_image_np(os.path.join(path,file_name))

img = normalizeStaining.normalizeStaining(img)[0] #return an np.ndarray

img = util.np_to_pil(img)
# save normaluzed image
img.save(os.path.join(result_path, file_name[:-4] + '_normalized' + '.png'))

end = timer()
print("Time to normalize and save image: %s\n" % str(end-start))

end_total = timer()
print("Time to normalize and save ALL images: %s\n" % str(end_total-start_total))
