# -*- coding: utf-8 -*-
"""
Created on Mon March 29 19:36:00 2021

@author: David
"""

# add paths containing code and data
import os
import sys
sys.path.insert(0, '/home/davhr856/thesis/saved_models')
sys.path.insert(0, '/nobackup/data/davhr856')
sys.path.insert(0, '/nobackup/data/davhr856/data/TCGA_DX')

gpu = str(sys.argv[1])
model_name = str(sys.argv[2])
#model_name = 'resnet_block4_finetuned'
patch_size = str(sys.argv[3])

import math
import numpy as np
from timeit import default_timer as timer
import warnings
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import re
from sklearn import metrics
import pandas as pd
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]=gpu;

# import the necessary packages
import keras
from sklearn.metrics import classification_report

# Allow growth of GPU memory, otherwise it will always look like all the memory is being used
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Ignore FutureWarning from numpy
warnings.simplefilter(action='ignore', category=FutureWarning)

base_path = '/nobackup/data/davhr856/data/TCGA_DX/modelling/dataset_224/' + patch_size
save_path = os.path.join('/home/davhr856/thesis/saved_models', patch_size)
model_save_path = os.path.join(save_path, model_name)
log_file_name = 'slide_predictions_log_' + str(model_name) + '_' + str(patch_size) + '.txt'
log_path = os.path.join(model_save_path, 'slide_predictions', log_file_name)

if not os.path.exists(os.path.join(model_save_path, 'slide_predictions')):
    os.makedirs(os.path.join(model_save_path, 'slide_predictions'))

logfile = open(log_path,'a')
logfile.write('\n')
logfile.write('\nSlide aggregation results for model: ' + str(model_name) + ' - ' + str(patch_size))
logfile.close()

# create ImageDataGenerators
# dimensions of our images
img_width, img_height = 224, 224

valid_data_dir = os.path.join(base_path, 'validation')
test_data_dir = os.path.join(base_path, 'test_extra')

### hyperparameters
batch_size = 64

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


test_datagen = ImageDataGenerator(preprocessing_function = tf.keras.applications.resnet.preprocess_input)


# initialize the testing generator
test_generator = test_datagen.flow_from_directory(
    directory=test_data_dir,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    interpolation = "bicubic",
    batch_size=batch_size)

# initialize the validation generator
valid_generator = test_datagen.flow_from_directory(
    directory=valid_data_dir,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    interpolation = "bicubic",
    batch_size=batch_size)

print(valid_generator.class_indices)
print(test_generator.class_indices)

nvalid = len(valid_generator.labels)
ntest = len(test_generator.labels)
validation_steps = nvalid / batch_size
test_steps = ntest / batch_size


# import numpy array with MC simulations (predictions)
y_30MC_test = np.load(os.path.join(model_save_path,'test_30MC.npy'))
y_30MC_valid = np.load(os.path.join(model_save_path, 'valid_30MC.npy'))

print('Test predictions shape: ')
print(str(y_30MC_test.shape))
print('Validation predictions shape: ')
print(str(y_30MC_valid.shape))

# helper functions
def patch_accuracy(test_generator, y_mc):
    test_generator.reset()
    ntest = len(test_generator.labels)

    # get mean of MC DO runs
    y_mean = np.mean(y_mc, axis = 0)
    # class with the higher average softmax output, (shape = ntest)
    y_pred = np.argmax(y_mean, axis=1)

    # patch level accuracy
    tn, fp, fn, tp = metrics.confusion_matrix(test_generator.labels, y_pred).ravel()
    patch_accuracy = (tn + tp) / (tn + fp + fn + tp)
    return patch_accuracy

def get_slide_labels(test_generator):
    test_generator.reset()
    ntest = len(test_generator.labels)
    # slide labels
    slide_labels = []
    slides = []
    pattern = '/(.*?)/'
    for i in range(ntest):
        slide = re.search(pattern, test_generator.filenames[i]).group(1)
        label = test_generator.labels[i]
        if not slide in slides:
            slides.append(slide)
            slide_labels.append(label)
    return slide_labels

#________________________________________________________________________________________________________________________________________#

################ Slide aggregation methods
# 1. Majority voting
def majority_voting(test_generator, y_mc):
    # returns the slide predictions (0,1) for every slide in test_generator in the same order
    test_generator.reset()
    ntest = len(test_generator.labels)

    # get mean of MC DO runs
    y_mean = np.mean(y_mc, axis = 0)
    # class with the higher average softmax output, (shape = ntest)
    y_pred = np.argmax(y_mean, axis=1)

    # slide level predictions
    pattern = '/(.*?)/'
    current_slide = re.search(pattern, test_generator.filenames[0]).group(1)
    slide_predictions = []
    patch_predictions = []
    for i in range(ntest):
        slide = re.search(pattern, test_generator.filenames[i]).group(1)
        if slide == current_slide:
            patch_predictions.append(y_pred[i])
        else:
            slide_predictions.append(np.mean(patch_predictions))
            patch_predictions = []
            patch_predictions.append(y_pred[i])
        current_slide = slide
        if i == ntest-1:
            slide_predictions.append(np.mean(patch_predictions))

    slide_prediction_probs = [1-i for i in slide_predictions] # probability of GBM
    return slide_prediction_probs

#________________________________________________________________________________________________________________________________________#

# 4. Standard MIL
def standard_MIL(test_generator, y_mc, uncertainty_threshold):
    # returns the slide predictions (0,1) for every slide in test_generator in the same order
    test_generator.reset()
    ntest = len(test_generator.labels)
     # 0 = certain, 1 = uncertain
    # get mean of MC DO runs
    y_mean = np.mean(y_mc, axis = 0)
    # class with the higher average softmax output, (shape = ntest)
    y_pred = np.argmax(y_mean, axis=1)
    # get certainty of each patch with standard deviation, which is the same value for both class outputs
    # multiply by 2, to obtain the possible range [0,1]
    y_sd = np.std(y_mc[:,:,0], axis = 0)*2 #shape = (ntest,)

    def slide_prediction(patch_predictions, patch_uncertainties, uncertainty_threshold):
        slide_prediction = 1
        for j in range(len(patch_uncertainties)):
            if patch_predictions[j] == 0:
                if patch_uncertainties[j] < uncertainty_threshold:
                    slide_prediction = 0
                    break
        return slide_prediction
    # slide level predictions
    pattern = '/(.*?)/'
    current_slide = re.search(pattern, test_generator.filenames[0]).group(1)
    slide_predictions = []
    patch_predictions = []
    patch_uncertainties = []
    for i in range(ntest):
        slide = re.search(pattern, test_generator.filenames[i]).group(1)
        if slide == current_slide:
            patch_predictions.append(y_pred[i])
            patch_uncertainties.append(y_sd[i])
        else:
            slide_predictions.append(slide_prediction(patch_predictions, patch_uncertainties, uncertainty_threshold))
            patch_predictions = []
            patch_uncertainties = []

            patch_predictions.append(y_pred[i])
            patch_uncertainties.append(y_sd[i])
        current_slide = slide
        if i == ntest-1:
            slide_predictions.append(slide_prediction(patch_predictions, patch_uncertainties, uncertainty_threshold))
    return slide_predictions

#________________________________________________________________________________________________________________________________________#

# 8. pathologists_MIL
def pathologists_MIL(test_generator, y_mc):
    # returns the slide predictions (0,1) for every slide in test_generator in the same order
    test_generator.reset()
    ntest = len(test_generator.labels)
     # 0 = certain, 1 = uncertain
    # get mean of MC DO runs
    y_mean = np.mean(y_mc, axis = 0)
    def slide_prediction(patch_probs_0):
        return np.max(patch_probs_0)

    # slide level accuracy
    pattern = '/(.*?)/'
    current_slide = re.search(pattern, test_generator.filenames[0]).group(1)
    slide_predictions = []
    patch_probs_0 = []
    for i in range(ntest):
        slide = re.search(pattern, test_generator.filenames[i]).group(1)
        if slide == current_slide:
            patch_probs_0.append(y_mean[i][0])
        else:
            slide_predictions.append(slide_prediction(patch_probs_0))
            patch_probs_0 = []
            patch_probs_0.append(y_mean[i][0])
        current_slide = slide
        if i == ntest-1:
            slide_predictions.append(slide_prediction(patch_probs_0))

    return slide_predictions

#________________________________________________________________________________________________________________________________________#

# 5. Weighted collective assumption
def weighted_collective_MIL(test_generator, y_mc):
    # returns the slide predictions (0,1) for every slide in test_generator in the same order
    test_generator.reset()
    ntest = len(test_generator.labels)
    # 0 = certain, 1 = uncertain
    # get mean of MC DO runs
    y_mean = np.mean(y_mc, axis = 0) # (ntest,2)
    # get certainty of each patch with standard deviation, which is the same value for both class outputs
    y_sd = np.std(y_mc[:,:,0], axis = 0) * 2 #len = ntest
    weights = [-1*(item-1) for item in y_sd] # len = ntest

    def slide_prediction(patch_predictions, patch_weights):
        #the class probability of class 0
        weighted_prob_0 = []

        for j in range(len(patch_predictions)):
            prob_0 = patch_predictions[j][0] # softmax output for class 0 for this patch
            weighted_prob_0.append(patch_weights[j] * prob_0)

        bag_0_prob = sum(weighted_prob_0) / sum(patch_weights)
        #print("bag 0 weighted probability = " + str(bag_0_prob))
        if bag_0_prob<0.5:
            slide_prediction = 1
        else:
            slide_prediction = 0
        return bag_0_prob

    # slide level predictions
    pattern = '/(.*?)/'
    current_slide = re.search(pattern, test_generator.filenames[0]).group(1)
    slide_predictions = []
    patch_predictions = []
    patch_weights = []
    for i in range(ntest):
        slide = re.search(pattern, test_generator.filenames[i]).group(1)
        if slide == current_slide:
            patch_predictions.append(y_mean[i])
            patch_weights.append(weights[i])
        else:
            slide_predictions.append(slide_prediction(patch_predictions, patch_weights))
            patch_predictions = []
            patch_weights = []

            patch_predictions.append(y_mean[i])
            patch_weights.append(weights[i])
        current_slide = slide
        if i == ntest-1:
            slide_predictions.append(slide_prediction(patch_predictions, patch_weights))
    return slide_predictions

#________________________________________________________________________________________________________________________________________#

# 2. Logistic Regression
def logistic_regression_aggregation(test_generator, valid_generator, y_mc, y_mc_valid):
    # returns the slide predictions (0,1) for every slide in test_generator in the same order

    slide_labels = get_slide_labels(test_generator)

    def build_df(generator, y_mc):

        generator.reset()
        slide_labels = get_slide_labels(generator)

        n = len(generator.labels)
        # 0 = certain, 1 = uncertain
        # get mean of MC DO runs
        y_mean = np.mean(y_mc, axis = 0) # (ntest,2)
        y_pred = np.argmax(y_mean, axis=1) # (ntest)
        # get certainty of each patch with standard deviation, which is the same value for both class outputs
        y_sd = np.std(y_mc[:,:,0], axis = 0) * 2 #len = ntest
        weights = [-1*(item-1) for item in y_sd] # len = ntest
        y_labels = generator.labels # (ntest) patch true labels

        # slide level accuracy
        pattern = '/(.*?)/'
        current_slide = re.search(pattern, generator.filenames[0]).group(1)

        logreg_df = pd.DataFrame({})

        GBM95 = []
        LGG95 = []

        GBM95_count = 0
        LGG95_count = 0

        patches = []
        patch_count = 0

        for i in range(n):
            slide = re.search(pattern, generator.filenames[i]).group(1)
            if slide == current_slide:
                # put patch into appropriate counter
                if y_pred[i] == 0 and weights[i]>=0.95:
                    GBM95_count += 1

                if y_pred[i] == 1 and weights[i]>=0.95:
                    LGG95_count += 1

                patch_count += 1
            else:
                # append one count to list
                GBM95.append(GBM95_count)
                LGG95.append(LGG95_count)
                patches.append(patch_count)

                GBM95_count = 0
                LGG95_count = 0
                patch_count = 0

                # put patch into appropriate counter
                if y_pred[i] == 0 and weights[i]>=0.95:
                    GBM95_count += 1

                if y_pred[i] == 1 and weights[i]>=0.95:
                    LGG95_count += 1

                patch_count += 1

            current_slide = slide
            if i == n-1:

                # append one count to list
                GBM95.append(GBM95_count)
                LGG95.append(LGG95_count)
                patches.append(patch_count)

        # fill up df
        logreg_df['GBM95'] = GBM95
        logreg_df['LGG95'] = LGG95

        logreg_df['total_patches'] = patches
        logreg_df['slide_label'] = [1-i for i in slide_labels]

        # create ratios df
        logreg_df_ratios = pd.DataFrame({})
        logreg_df_ratios['GBM95_ratio'] = logreg_df.GBM95 / logreg_df.total_patches
        logreg_df_ratios['LGG95_ratio'] = logreg_df.LGG95 / logreg_df.total_patches
        logreg_df_ratios['slide_label'] = logreg_df.slide_label

        return logreg_df_ratios

    df_test = build_df(test_generator, y_mc)
    df_valid = build_df(valid_generator, y_mc_valid)

    # logistic regression
    logreg_X_valid = sm.add_constant(df_valid.iloc[:,:-1])
    logreg_Y_valid = df_valid.iloc[:,-1]

    logreg_X_test = sm.add_constant(df_test.iloc[:,:-1])
    logreg_Y_test = df_test.iloc[:,-1]

    logreg = sm.Logit(logreg_Y_valid, logreg_X_valid).fit(method = 'lbfgs', maxiter = 100)
    print(logreg.summary())
    logreg_predictions_probs = logreg.predict(logreg_X_test)

    slide_predictions = logreg_predictions_probs
    return [slide_predictions, logreg]


#________________________________________________________________________________________________________________________________________#

# 3. Spatial location

def flip_patches(y_pred_window, weight_window, threshold):
    # flips patches in a window, if conditions are satisfied
    flip = 0
    y_pred_window_flipped = np.array(y_pred_window)
    weight_window = np.array(weight_window)
    lgg_weights = weight_window[np.where(y_pred_window_flipped==1)]
    gbm_weights = weight_window[np.where(y_pred_window_flipped==0)]
    #print(lgg_weights)
    #print(gbm_weights)
    #print(y_pred_window_flipped)

    if len(lgg_weights) > 0 and len(gbm_weights) > 0:
        # flip possibility
        if max(lgg_weights) < threshold and max(gbm_weights) > threshold and len(lgg_weights) < len(gbm_weights):
            # flip lgg to gbm if conditions given
            #print("flip lgg to gbm")
            y_pred_window_flipped = [0 for i in range(len(y_pred_window_flipped))]
            flip = 1
        if max(gbm_weights) < threshold and max(lgg_weights) > threshold and len(gbm_weights) < len(lgg_weights):
            # flip lgg to gbm if conditions given
            #print("flip gbm to lgg")
            y_pred_window_flipped = [1 for i in range(len(y_pred_window_flipped))]
            flip = 1
    #print(y_pred_window_flipped)
    return [y_pred_window_flipped,flip]

def spatial_flipping(test_generator, y_mc, threshold):
    # returns the slide predictions (0,1) for every slide in test_generator in the same order
    test_generator.reset()
    ntest = len(test_generator.labels)
    # 0 = certain, 1 = uncertain
    # get mean of MC DO runs
    y_mean = np.mean(y_mc, axis = 0) # (ntest,2)
    y_pred = np.argmax(y_mean, axis=1)
    # get certainty of each patch with standard deviation, which is the same value for both class outputs
    y_sd = np.std(y_mc[:,:,0], axis = 0) * 2 #len = ntest
    weights = [-1*(item-1) for item in y_sd] # len = ntest
    pattern = '__(.*?)_'
    window_size = 9 #size of the square window

    # updated predictions list
    y_pred_updated = y_pred.copy()
    slide_predictions = []

    # build dictionary of filename_base - start
    file_start_dict = {}
    count = 0
    for i in range(len(test_generator.filenames)):
        filename_base = test_generator.filenames[i][:test_generator.filenames[i].find('__')]
        if not filename_base in file_start_dict:
            file_start_dict[filename_base] = (i, count)
            count = 1
        else:
            count += 1
            new_value = (file_start_dict[filename_base][0], count)
            file_start_dict[filename_base] = new_value

    # loop through the slides and save patches in a list
    slide_counter = 0
    for slide in file_start_dict:
        print("Flipping slide number: " + str(slide_counter))
        slide_counter += 1
        start = file_start_dict[slide][0]
        end = start + file_start_dict[slide][1]
        patches = test_generator.filenames[start:end] # filenames belonging to current slide
        y_pred_patches = y_pred[start:end] # patch predictions (0,1) belonging to current slide
        print('Initial majority voting: ' + str(np.mean(y_pred_patches)))
        slide_label = test_generator.labels[start]
        print('Slide label: ' + str(slide_label))

        # get patch coordinates
        x_list = []
        y_list = []
        for patch in patches:
            coords = re.search(pattern, patch).group(1)
            x = coords[:coords.find("-")]
            y = coords[coords.find("-")+1:]
            x_list.append(int(x))
            y_list.append(int(y))

        # loop through the map with smaller non-overlapping windows
        x_steps = (max(x_list)-(min(x_list)-1))//window_size
        y_steps = (max(y_list)-(min(y_list)-1))//window_size
        windows = x_steps * y_steps


        x_list_patch_candidates = []
        y_list_patch_candidates = []

        gbm_patch_coords_x = []
        gbm_patch_coords_y = []
        lgg_patch_coords_x = []
        lgg_patch_coords_y = []

        gbm_patch_coords_x_flipped = []
        gbm_patch_coords_y_flipped = []
        lgg_patch_coords_x_flipped = []
        lgg_patch_coords_y_flipped = []

        weights_patch = []
        counter = 0
        flip_counter = 0

        gbm_count = 0
        lgg_count = 0
        gbm_count_flipped = 0
        lgg_count_flipped = 0

        for x_step in range(x_steps):
            for y_step in range(y_steps):

                # lists containing info about the patches in a window
                y_pred_window = []
                weight_window = []
                ind_window = [] # index of patches in test_generator
                for x_patch in range(window_size):
                    for y_patch in range(window_size):
                        patch_coords = str(min(x_list)+x_step*window_size+x_patch) + '-' + str(min(y_list)+y_step*window_size+y_patch)
                        patch_candidate = slide + '__' + patch_coords + '_'
                        ind = [ind for ind, patch in enumerate(patches) if re.search(patch_candidate+'.+', patch)]
                        # ind is an empty list if patch doesn't exist, otherwise it's the index in the list
                        if ind:
                            #print(patch_coords)
                            counter += 1
                            x_list_patch_candidates.append(int(str(min(x_list)+x_step*window_size+x_patch)))
                            y_list_patch_candidates.append(int(str(min(y_list)+y_step*window_size+y_patch)))
                            weight_i = weights[start+ind[0]]
                            y_pred_i = y_pred[start+ind[0]]
                            weights_patch.append(weight_i)
                            if y_pred_i == 1:
                                lgg_count += 1
                                lgg_patch_coords_x.append(int(str(min(x_list)+x_step*window_size+x_patch)))
                                lgg_patch_coords_y.append(int(str(min(y_list)+y_step*window_size+y_patch)))
                            else:
                                gbm_count += 1
                                gbm_patch_coords_x.append(int(str(min(x_list)+x_step*window_size+x_patch)))
                                gbm_patch_coords_y.append(int(str(min(y_list)+y_step*window_size+y_patch)))
                            # add info to window lists
                            y_pred_window.append(y_pred_i)
                            weight_window.append(weight_i)
                            ind_window.append(start + ind[0])

                # flip patches
                y_pred_window_flipped, flip = flip_patches(y_pred_window, weight_window, threshold)
                # update y_pred list
                for i in range(len(ind_window)):
                    y_pred_updated[ind_window[i]] = y_pred_window_flipped[i]
                if flip == 1:
                    flip_counter += 1

                """
                for x_patch in range(window_size):
                    for y_patch in range(window_size):
                        patch_coords = str(min(x_list)+x_step*window_size+x_patch) + '-' + str(min(y_list)+y_step*window_size+y_patch)
                        patch_candidate = slide + '__' + patch_coords + '_'
                        ind = [ind for ind, patch in enumerate(patches) if re.search(patch_candidate+'.+', patch)]
                        # ind is an empty list if patch doesn't exist, otherwise it's the index in the list
                        if ind:
                            #print(patch_coords)
                            #x_list_patch_candidates.append(int(str(min(x_list)+x_step*window_size+x_patch)))
                            #y_list_patch_candidates.append(int(str(min(y_list)+y_step*window_size+y_patch)))
                            weight_i = weights[start+ind[0]]
                            y_pred_i = y_pred_updated[start+ind[0]]
                            #weights_patch.append(weight_i)
                            if y_pred_i == 1:
                                lgg_count_flipped += 1
                                lgg_patch_coords_x_flipped.append(int(str(min(x_list)+x_step*window_size+x_patch)))
                                lgg_patch_coords_y_flipped.append(int(str(min(y_list)+y_step*window_size+y_patch)))
                            else:
                                gbm_count_flipped += 1
                                gbm_patch_coords_x_flipped.append(int(str(min(x_list)+x_step*window_size+x_patch)))
                                gbm_patch_coords_y_flipped.append(int(str(min(y_list)+y_step*window_size+y_patch)))
                """

        #print('GBM: ' + str(gbm_count))
        #print('LGG: ' + str(lgg_count))
        #print("Flipped!")
        #print('GBM: ' + str(gbm_count_flipped))
        #print('LGG: ' + str(lgg_count_flipped))
        #print('Total patches looked at: ' + str(counter))
        #print('Number of flips: ' + str(flip_counter))
        y_pred_patches_updated = y_pred_updated[start:end]
        slide_prediction = np.mean(y_pred_patches_updated)
        slide_predictions.append(slide_prediction)
        print('Updated majority voting: ' + str(np.mean(y_pred_patches_updated)))
        print('______________________')

    slide_prediction_probs = [1-i for i in slide_predictions] # probability of GBM
    return slide_prediction_probs


#____________________________________________________ aggregations  _______________________________________________________#

majority_voting_predictions = majority_voting(test_generator, y_30MC_test)

print('Majority voting predictions: ' + str(majority_voting_predictions))
logfile = open(log_path,'a')
logfile.write('\n')
logfile.write('\nMajority voting predictions: ' + str(majority_voting_predictions))
logfile.close()

#________________________________________________________________________________________________________________________________________#

weighted_collective_MIL_predictions = weighted_collective_MIL(test_generator, y_30MC_test)

print('Weighted collective MIL predictions: ' + str(weighted_collective_MIL_predictions))
logfile = open(log_path,'a')
logfile.write('\n')
logfile.write('\nWeighted collective MIL predictions: ' + str(weighted_collective_MIL_predictions))
logfile.close()

#________________________________________________________________________________________________________________________________________#

logistic_regression_aggregation_predictions= logistic_regression_aggregation(test_generator, valid_generator, y_30MC_test, y_30MC_valid)

print('Logistic regression predictions: ' + str(logistic_regression_aggregation_predictions[0]))
logfile = open(log_path,'a')
logfile.write('\n')
logfile.write('\nLogistic regression predictions: ' + str(logistic_regression_aggregation_predictions[0].tolist()))
logfile.write('\n')
logfile.write(str(logistic_regression_aggregation_predictions[1].summary()))
logfile.close()

#________________________________________________________________________________________________________________________________________#

# gridsearch for optimal threshold value
thresholds = np.linspace(0.00001, 0.01, 100)
slide_labels = get_slide_labels(valid_generator)
accs = []
for threshold in thresholds:
    standard_MIL_predictions = standard_MIL(valid_generator, y_30MC_valid, threshold)
    tn, fp, fn, tp = metrics.confusion_matrix(slide_labels, standard_MIL_predictions).ravel()
    slide_accuracy = (tn + tp) / (tn + fp + fn + tp)
    slide_accuracy
    accs.append(slide_accuracy)

opt_threshold = thresholds[accs.index(max(accs))]
print('Optimal parameter: ' + str(opt_threshold))
standard_MIL_predictions = standard_MIL(test_generator, y_30MC_test, opt_threshold)

print('Standard MIL predictions: ' + str(standard_MIL_predictions))
logfile = open(log_path,'a')
logfile.write('\n')
logfile.write('\nStandard MIL optimal parameter: ' + str(opt_threshold))
logfile.write('\nStandard MIL predictions: ' + str(standard_MIL_predictions))
logfile.close()

#________________________________________________________________________________________________________________________________________#

pathologists_MIL_predictions = pathologists_MIL(test_generator, y_30MC_test)

print('Pathologists MIL predictions: ' + str(pathologists_MIL_predictions))
logfile = open(log_path,'a')
logfile.write('\n')
logfile.write('\nPathologists MIL predictions: ' + str(pathologists_MIL_predictions))
logfile.close()

#________________________________________________________________________________________________________________________________________#

# gridsearch for optimal threshold value
thresholds = np.linspace(0.8, 0.99, 10)
slide_labels = get_slide_labels(valid_generator)
slide_labels = [1-i for i in slide_labels] #gbm = 1
accs = []
count = 0
max_count = len(thresholds)
for threshold in thresholds:
    count += 1
    print('Spatial flipping gridsearch: ' + str(count) + '/' + str(max_count))

    spatial_flipping_predictions = spatial_flipping(valid_generator, y_30MC_valid, threshold)
    tn, fp, fn, tp = metrics.confusion_matrix(slide_labels, np.round(spatial_flipping_predictions)).ravel()
    slide_accuracy = (tn + tp) / (tn + fp + fn + tp)
    slide_accuracy
    accs.append(slide_accuracy)

opt_threshold = thresholds[accs.index(max(accs))]
print('Optimal parameter: ' + str(opt_threshold))
spatial_flipping_predictions = spatial_flipping(test_generator, y_30MC_test, opt_threshold)

print('Spatial flipping predictions: ' + str(spatial_flipping_predictions))
logfile = open(log_path,'a')
logfile.write('\n')
logfile.write('\nSpatial flipping optimal parameter: ' + str(opt_threshold))
logfile.write('\nSpatial flipping predictions: ' + str(spatial_flipping_predictions))
logfile.close()

#________________________________________________________________________________________________________________________________________#

print('SLIDE AGGREGATION FINISHED SUCCESSFULLY!')
