# -*- coding: utf-8 -*-
"""
Created on Mon March 08 19:36:00 2021

@author: David
"""

# add paths containing code and data
import os
import sys
sys.path.insert(0, '/nobackup/data/davhr856')
sys.path.insert(0, '/nobackup/data/davhr856/data/TCGA')
sys.path.insert(0, '/nobackup/data/davhr856/git_clones/deep-histopath')
sys.path.insert(0, '/nobackup/data/davhr856/git_clones/deep-histopath/deephistopath/wsi')

# import scripts from the deep-histopath github repo
import slide
import util
import filter
import tiles

import math
import numpy as np
import csv
from timeit import default_timer as timer
from skimage import img_as_bool
import patchify
import staintools
import openslide
import PIL
PIL.Image.MAX_IMAGE_PIXELS = None

# patching function
def patch_image(rgb, pxl, file_name, patches_path, target_image, normalizer, gbm_lgg):
    save_path = os.path.join(patches_path, gbm_lgg + "_" + file_name[:-4])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    patches = patchify.patchify(rgb, (pxl, pxl, 3), step = pxl)
    count_all_patches = 0
    count_good_patches = 0
    for r in range(patches.shape[0]):
        for c in range(patches.shape[1]):
            count_all_patches += 1
            patch = patches[r,c,0,:,:,:]
            if filter.tissue_percent(patch) > 95:
                #change black pixels (mask) to white (background)
                patch[np.where((patch==[0,0,0]).all(axis=2))] = [255,255,255]
                # normalize patch
                patch = patch.copy()
                patch = normalizer.transform(patch)
                # save patch
                patch = util.np_to_pil(patch)

                # resize to 224,224
                new_w = 224
                new_h = 224
                patch = patch.resize((new_w, new_h), PIL.Image.BICUBIC)

                patch.save(os.path.join(save_path, gbm_lgg + "_" + file_name[:-4] + '__' + str(r) + '-' + str(c) + '_' + str(count_good_patches) + '.png'))
                count_good_patches += 1
    return [gbm_lgg, file_name[:-4], str(pxl), str(count_all_patches), str(count_good_patches), str(round(count_good_patches / count_all_patches, 2))]




base_path = '/nobackup/data/davhr856/data/TCGA_DX'
#base_path = '/nobackup/data/davhr856/data/test_preprocess'
log_path = os.path.join(base_path, 'logs', 'preprocess_log.txt')
log_done_path = os.path.join(base_path, 'logs', 'preprocess_done.txt')
log_patches_path = os.path.join(base_path, 'logs', 'patches.txt')
# get list of file_names that have been preprocessed already
with open(log_done_path) as f:
    files_done = f.readlines()

# name of csv file
csv_name = os.path.join(base_path, "preprocess.csv")

counter = 1
patch_stats = []
# loop through all folders and files
for folder in os.listdir(os.path.join(base_path, "dataset_extra_test")):
    # folder can be GBM or LGG
    for id in os.listdir(os.path.join(base_path, "dataset_extra_test", folder)):
        for file_name in os.listdir(os.path.join(base_path,'dataset_extra_test',folder,id)):
            if file_name.endswith('.svs'):
                print(counter)
                counter += 1
                path = os.path.join(base_path, 'dataset_extra_test', folder, id, file_name)
                print(path)

                # check if file_name has been preprocessed already
                if file_name + "\n" in files_done:
                    continue

                start = timer()
                start_read = timer()
                slide_current = openslide.OpenSlide(path)

                try:
                    original_magnification = slide_current.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
                except:
                    original_magnification = 40 # where it is missing, it is 40 (empirically)
                original_resolution = slide_current.dimensions

                # read WSI at max and min resolution
                whole_slide_image = slide_current.read_region((0, 0), 0, slide_current.level_dimensions[0]).convert("RGB")
                thumbnail = slide_current.read_region((0, 0), slide_current.level_count-1, slide_current.level_dimensions[slide_current.level_count-1]).convert("RGB")

                end_read = timer()
                print("Time to open svs file: %s\n" % str(end_read-start_read))

                logfile = open(log_path,'a')
                logfile.write('\n')
                logfile.write('\nFile: ' + path)
                logfile.write('\nFile opened in: ' + str(end_read-start_read))
                logfile.close()

                if int(original_magnification) == 40:
                    # downscale by 2 to get 20x
                    factor = 2
                    # resize image
                    start_png = timer()
                    new_w = math.floor(slide_current.level_dimensions[0][0] / factor)
                    new_h = math.floor(slide_current.level_dimensions[0][1] / factor)

                    whole_slide_image = whole_slide_image.resize((new_w, new_h), PIL.Image.BICUBIC)
                    end_png = timer()
                    print("Time to resize: %s\n" % str(end_png-start_png))

                    logfile = open(log_path,'a')
                    logfile.write('\nImage resized: ' + str(end_png-start_png))
                    logfile.close()

                    new_resolution = whole_slide_image.size

                    """
                    # save resized image
                    start_save = timer()
                    img_path = os.path.join('/nobackup/data/davhr856/data/TCGA_DX/dev/test', gbm_lgg + '_' + file_name[:-4] + '.png')
                    whole_slide_image.save(img_path)
                    end_save = timer()
                    print("Time to save resized image: %s\n" % str(end_save-start_save))

                    logfile = open(os.path.join(base_path, 'dev', 'test', 'preprocess_log.txt'),'a')
                    logfile.write('\nResized image saved: ' + str(end_save-start_save))
                    logfile.close()
                    """
                elif int(original_magnification) == 20:
                    new_resolution = original_resolution
                    """
                    # save image
                    start_save = timer()
                    img_path = os.path.join(base_path, 'dev', 'test', gbm_lgg + '_' + file_name[:-4] + '.png')
                    whole_slide_image.save(img_path)
                    end_save = timer()
                    print("Time to save image: %s\n" % str(end_save-start_save))

                    logfile = open(os.path.join(base_path, 'dev', 'test', 'preprocess_log.txt'),'a')
                    logfile.write('\nImage saved: ' + str(end_save-start_save))
                    logfile.close()
                    """
                else:
                    # don't save the image
                    new_resolution = original_resolution
                    # exit script
                    sys.exit()

                ##### apply filters
                # convert wsi and thumbnail to np array
                start_np = timer()
                thumbnail = util.pil_to_np_rgb(thumbnail)
                whole_slide_image = util.pil_to_np_rgb(whole_slide_image)
                end_np = timer()
                print("Time to convert image to np array: %s\n" % str(end_np - start_np))

                logfile = open(log_path,'a')
                logfile.write('\nImages converted to np: ' + str(end_np - start_np))
                logfile.close()

                """
                # filter out blue and green pens
                start_pen = timer()
                bluepen_filter = filter.filter_blue_pen(thumbnail)
                greenpen_filter = filter.filter_green_pen(thumbnail)
                pen_filter = bluepen_filter & greenpen_filter
                end_pen = timer()
                print("Time to filter pens: %s\n" % str(end_pen - start_pen))

                logfile = open(log_path,'a')
                logfile.write('\nPen filters done: ' + str(end_pen - start_pen))
                logfile.close()
                """

                # filter out background
                start_bg = timer()
                bg_filter = filter.filter_green_channel(thumbnail)
                end_bg = timer()
                print("Time to filter bg: %s\n" % str(end_bg - start_bg))

                logfile = open(log_path,'a')
                logfile.write('\nBg filter done: ' + str(end_bg - start_bg))
                logfile.close()

                # remove small holes
                start_holes = timer()
                holes_filter = filter.filter_remove_small_holes(bg_filter, min_size = 100, output_type = "bool")
                end_holes = timer()
                print("Time to fill holes: %s\n" % str(end_holes - start_holes))

                logfile = open(log_path,'a')
                logfile.write('\nHoles filter done: ' + str(end_holes - start_holes))
                logfile.close()

                # combine masks and upscale it to whole_slide_image size
                start_upscale = timer()
                #mask = pen_filter & holes_filter
                mask = holes_filter
                mask = util.np_to_pil(mask)
                new_w = whole_slide_image.shape[1]
                new_h = whole_slide_image.shape[0]
                mask_upscaled = img_as_bool(mask.resize((new_w, new_h), PIL.Image.BILINEAR))
                end_upscale = timer()
                print("Time to upscale mask: %s\n" % str(end_upscale - start_upscale))

                logfile = open(log_path,'a')
                logfile.write('\nMask upscaled: ' + str(end_upscale - start_upscale))
                logfile.close()


                # apply upscaled mask to large image
                start_filter = timer()
                whole_slide_image = util.mask_rgb(whole_slide_image, mask_upscaled)
                end_filter = timer()
                print("Time to apply filters: %s\n" % str(end_filter - start_filter))

                # calculate size of wsi
                wsi_size = whole_slide_image.nbytes/1024/1024/1024
                #del whole_slide_image
                print("WSI variable: " + str(wsi_size))

                logfile = open(log_path,'a')
                logfile.write('\nFiltering done: ' + str(end_filter - start_filter))
                logfile.write('\nSize of wsi variable: ' + str(wsi_size))
                logfile.close()

                """
                # save filtered image
                filtered = util.np_to_pil(whole_slide_image)
                filtered.save(os.path.join(base_path, 'dev', 'test', gbm_lgg + '_' + file_name[:-4] + '_filtered' + '.png'))
                end_filter_save = timer()
                print("Time to apply filters and save filtered image: %s\n" % str(end_filter_save - start_filter_save))

                logfile = open(os.path.join(base_path, 'dev', 'test', 'preprocess_log.txt'),'a')
                logfile.write('\nFiltering done: ' + str(end_filter_save - start_filter_save))
                logfile.close()
                """

                ## patching and normalizing
                # set up normalizer
                target_image_path = os.path.join(base_path, "LGG_TCGA-DU-7012-01Z-00-DX1.3D0ACB4F-8CBB-47F9-B015-0AFF70928A62__66-66_6468.png")
                target_image = staintools.read_image(target_image_path)
                normalizing_method = "vahadane"

                normalizer = staintools.StainNormalizer(method = normalizing_method)
                normalizer.fit(target_image)

                patch_sizes = {
                  336: "small",
                  672: "medium"
                }

                for patch_size in patch_sizes:
                    start_patching = timer()
                    patches_path = os.path.join(base_path, 'patches_extra_test', patch_sizes[patch_size])
                    if folder == "LGG":
                        patches_path = os.path.join(patches_path, 'LGG')
                    else:
                        patches_path = os.path.join(patches_path, 'GBM')

                    if not os.path.exists(patches_path):
                        os.makedirs(patches_path)

                    patch_stat = patch_image(rgb = whole_slide_image, pxl = int(patch_size), file_name = file_name, patches_path = patches_path, target_image = target_image, normalizer = normalizer, gbm_lgg = folder)
                    patch_stats.append(patch_stat)
                    end_patching = timer()
                    print("Time to do patching: %s\n" % str(end_patching - start_patching))

                    logfile = open(log_path,'a')
                    logfile.write('\nPatching stats for ' + str(patch_size) + ' pxls: ' + str(patch_stat))
                    logfile.write('\nTime to do patching: ' + str(end_patching - start_patching))
                    logfile.close()

                    logfile = open(log_patches_path,'a')
                    logfile.write('\n' + patch_stat[0] + ";" + patch_stat[1] + ";" + str(patch_stat[2]) + ";" + str(patch_stat[3]) + ";" + str(patch_stat[4]) + ";" + str(patch_stat[5]))
                    logfile.close()

                end = timer()
                print("Time total: %s\n" % str(end-start))

                logfile = open(log_path,'a')
                logfile.write('\nTotal time: ' + str(end-start))
                logfile.close()


# writing to csv file
with open(csv_name, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile, delimiter = ';')
    # writing the fields
    fields = ["class", "file_name", "patch_size", "count_all_patches", "count_good_patches", "good_patch_ratio"]
    csvwriter.writerow(fields)
    csvwriter.writerows(patch_stats)
