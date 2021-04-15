# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 19:36:00 2021

@author: David
"""

# import
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = None
import numpy as np
import os
import patchify
from timeit import default_timer as timer
import sys
import csv

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

tile_size = int(sys.argv[1])



def tile_image(rgb, pxl, file_name, tiles_path):
    save_path = os.path.join(tiles_path,file_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    patches = patchify.patchify(rgb, (pxl, pxl, 3), step = pxl)
    count_all_patches = 0
    count_good_patches = 0
    for r in range(patches.shape[0]):
        for c in range(patches.shape[1]):
            count_all_patches += 1
            patch = patches[r,c,0,:,:,:]
            if filter.tissue_percent(patch) > 90:
                patch = util.np_to_pil(patch)
                patch.save(os.path.join(save_path, file_name[:-4] + '__' + str(r) + '-' + str(c) + '_' + str(count_good_patches) + '.png'))
                count_good_patches += 1
    return [file_name[:-4], count_all_patches, count_good_patches]



# file paths
base_path = '/nobackup/data/davhr856/data/TCGA_DX/dev'
path = '/nobackup/data/davhr856/data/TCGA_DX/dev/resized'
# result path
tiles_path = '/nobackup/data/davhr856/data/TCGA_DX/dev/patches'

# name of csv file
csv_name = os.path.join(base_path, "tiles.csv")
tiles_stats = []

if not os.path.exists(tiles_path):
    os.makedirs(tiles_path)

start_total = timer()

counter = 1

for file_name in os.listdir(os.path.join(path)):

    print(counter)
    counter += 1

    print(file_name)
    start = timer()

    # read png image file as np
    img = slide.open_image_np(os.path.join(path,file_name))

    tile_stat = tile_image(rgb = img, pxl = tile_size, file_name = file_name, tiles_path = tiles_path)
    tiles_stats.append(tile_stat)

    end = timer()
    print("Time to tile one image: %s\n" % str(end-start))

end_total = timer()
print("Time to tile ALL images: %s\n" % str(end_total-start_total))

# writing to csv file
with open(csv_name, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile, delimiter = ';')
    # writing the fields
    fields = ['filename', 'count_all_patches', 'count_good_patches']
    csvwriter.writerow(fields)
    csvwriter.writerows(tiles_stats)
