#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import string

data_dir = sys.argv[1]

with open(data_dir+'/training_new.txt','r') as f:
    files_ordered = f.readlines()
files_ordered[-1] += '\r\n'

with open(data_dir+'/order_of_validation.txt','r') as f:
    images_val = f.readlines()
    
with open(data_dir+'/validation.txt','r') as f:
    labels_val = f.readlines()

    
for f in zip(images_val,labels_val):
    files_ordered.append(string.join([f[0].strip(),f[1]],'\t'))

# create list of all training images
with open(data_dir+'/training_new_plus_validation.txt', 'w') as fid:
    for f in files_ordered:
        fid.write('%s' % f) 

# create list of test images
files_ordered = np.sort(os.listdir(data_dir+'/Test'))
with open(data_dir+'/order_of_test.txt','w') as fid:
    for f in files_ordered:
        fid.write('%s\r\n' % f)