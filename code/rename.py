# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 19:36:00 2021

@author: David
"""

import pandas as pd
import os
# add paths containing code and data
import sys
sys.path.insert(0, '/nobackup/data/davhr856')

names_lookup = pd.read_csv('/nobackup/data/davhr856/code/rename.csv')

path = '/nobackup/data/davhr856/data/TCGA_flat/training_slides'

for filename in os.listdir(path):
    newname = names_lookup[names_lookup['original_name'] == filename].values[0][1]
    os.rename(r'/nobackup/data/davhr856/data/TCGA_flat/training_slides/'+ filename, r'/nobackup/data/davhr856/data/TCGA_flat/training_slides/'+ newname)