# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 19:36:00 2021

@author: David
"""
# RUN ON HARLEY
# add paths containing code and data
import sys
sys.path.insert(0, '/flush/davhr856/git_clones/deep-histopath/deephistopath/wsi')
sys.path.insert(0, '/flush/davhr856/git_clones/deep-histopath')
sys.path.insert(0, '/flush/common/TCGA/histology/Diagnostic')
sys.path.insert(0, '/flush/davhr856')

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


import csv
import math

data_path = '/flush/common/TCGA/histology/Diagnostic'
path = '/flush/davhr856/data/TCGA_DX/resized'
if not os.path.exists(path):
    os.makedirs(path)


# name of csv file
csv_name = os.path.join(path,"slide_resized.csv")
counter = 1

slide_resolutions = []

for folder in os.listdir(data_path):
    # do only 20 images
    if not os.path.isdir(os.path.join(data_path, folder)):
        continue
    if counter > 2:
        break
    for folder2 in folder:
        for filename in os.listdir(os.path.join(data_path,folder,folder2)):
            # do only 20 images
            if counter > 2:
                break
            if filename.endswith('.svs'):

                print(filename)
                print(counter)

                counter += 1

                slide_current = openslide.OpenSlide(os.path.join(data_path,folder,folder2,filename))
                try:
                    original_magnification = slide_current.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
                except:
                    original_magnification = 0

                original_resolution = slide_current.dimensions

                if int(original_magnification) == 40:
                    # open slide at level 1, and downscale by 2
                    factor = 2
                    # read and resize image
                    new_w = math.floor(slide_current.level_dimensions[0][0] / factor)
                    new_h = math.floor(slide_current.level_dimensions[0][1] / factor)
                    whole_slide_image = slide_current.read_region((0, 0), 0, slide_current.level_dimensions[0])
                    whole_slide_image = whole_slide_image.convert("RGB")
                    whole_slide_image = whole_slide_image.resize((new_w, new_h), PIL.Image.BICUBIC) # maybe overwrite whole_slide_image if RAM issues
                elif int(original_magnification) == 20:
                    # open slide at level 1 and don't downscale
                    whole_slide_image = slide_current.read_region((0, 0), 0, slide_current.level_dimensions[0])
                    whole_slide_image = whole_slide_image.convert("RGB")
                else:
                    # don't save the image
                    new_resolution = original_resolution
                    #csvwriter.writerow([filename,original_magnification,original_resolution,new_resolution])
                    slide_resolutions.append([filename,original_magnification,original_resolution,new_resolution, folder, new_resolution[0], new_resolution[1], new_resolution[0] * new_resolution[1]]])
                    continue


                img_path = os.path.join(path, folder + '_' + filename[:-4] + '.png')
                whole_slide_image.save(img_path)
                new_resolution = whole_slide_image.size

                slide_resolutions.append([filename,original_magnification,original_resolution,new_resolution, folder, new_resolution[0], new_resolution[1], new_resolution[0] * new_resolution[1]])

# writing to csv file
with open(csv_name, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile, delimiter = ';')
    # writing the fields
    fields = ['filename','original_magnification', 'original_resolution', 'new_resolution', 'class', 'pixels_x', 'pixels_y', 'pixels']
    csvwriter.writerow(fields)
    csvwriter.writerows(slide_resolutions)
