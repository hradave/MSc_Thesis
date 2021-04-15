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

# import scripts from the deep-histopath github repo
import slide
import util
import filter
import tiles

# own filter function building on filter.apply_image_filters
def apply_image_filters_custom(rgb):
    """
    Apply filters to image as NumPy array and optionally save and/or display filtered images.

    Args:
    rgb: Image as NumPy array.


    Returns:
    Resulting filtered image as a NumPy array.
    """

    #save_display(save, display, info, rgb, slide_num, 1, "Original", "rgb")
    print(rgb.dtype)
    print(rgb.nbytes/1024/1024/1024)

    hyst = filter.filter_hysteresis_threshold(filter.filter_complement(filter.filter_rgb_to_grayscale(img)), low = 50, high = 100,output_type="bool")
    mask_final = filter.filter_remove_small_holes(hyst, min_size=10000, output_type="bool")
    #print(mask_remove_small_holes.dtype)
    #print(mask_remove_small_holes.nbytes/1024/1024/1024)

    # apply final mask to rgb image
    rgb = util.mask_rgb(rgb, mask_final)
    #print(rgb.dtype)
    #print(rgb.nbytes/1024/1024/1024)
    # replace black pixels with white
    #rgb[np.where((rgb==[0,0,0]).all(axis=2))] = [255,255,255]
    #print(rgb.dtype)
    #print(rgb.nbytes/1024/1024/1024)

    return rgb

# file paths
path = '/nobackup/data/davhr856/data/TCGA_DX/sample/normalized'
# filtered image path
result_path = os.path.join('/nobackup/data/davhr856/data/TCGA_DX/sample/filtered_after_normalize')
if not os.path.exists(result_path):
    os.makedirs(result_path)

for file_name in os.listdir(os.path.join(path)):

    print(file_name)
    start = timer()

    # read png image file as np
    img = slide.open_image_np(os.path.join(path,file_name))

    img = apply_image_filters_custom(rgb = img)

    img = util.np_to_pil(img)
    # save filtered image
    img.save(os.path.join(result_path, file_name[:-4] + '_filtered_after_norm' + '.png'))

    end = timer()
    print("Time to apply filters and save image: %s\n" % str(end-start))
