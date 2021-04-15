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


path = '/nobackup/data/davhr856/data/TCGA_DX'


# name of csv file  
csv_name = os.path.join(path,"slide_downsamples_GBM.csv")
counter = 1

slide_downsamples = []
        
for folder in os.listdir(os.path.join(path,'original','GBM')):
    for filename in os.listdir(os.path.join(path,'original','GBM',folder)):
        if filename.endswith('.svs'):            
            print(filename)
            print(counter)
            counter += 1
            slide_current = openslide.OpenSlide(os.path.join(path,'original','GBM',folder,filename))
            
            try:
                original_magnification = slide_current.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
            except:
                original_magnification = 0
            dimensions = slide_current.level_dimensions
            level_downsamples = slide_current.level_downsamples
            pixels = dimensions[0][0]*dimensions[0][1]
            try:
                MPP = slide_current.properties[openslide.PROPERTY_NAME_MPP_X]
            except:
                MPP = 0
            
            
            slide_downsamples.append([folder, filename, original_magnification, dimensions, level_downsamples, MPP, pixels])
              
# writing to csv file  
with open(csv_name, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile, delimiter = ';')
    # writing the fields
    fields = ['folder','filename','original_magnification', 'dimensions', 'level_downsamples', 'MPP', 'pixels']
    csvwriter.writerow(fields)
    csvwriter.writerows(slide_downsamples)
                
              

        
                