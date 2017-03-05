#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import string

training_new = sys.argv[1]
order_of_validation = sys.argv[2]
validation = sys.argv[3]
training_new_plus_validation = sys.argv[4]
test_dir = sys.argv[5]
order_of_test = sys.argv[6]

with open(training_new,'r') as f:
    files_ordered = f.readlines()
files_ordered[-1] += '\r\n'

with open(order_of_validation,'r') as f:
    images_val = f.readlines()
    
with open(validation,'r') as f:
    labels_val = f.readlines()

    
for f in zip(images_val,labels_val):
    files_ordered.append(string.join([f[0].strip(),f[1]],'\t'))

# create list of all training images
with open(training_new_plus_validation, 'w') as fid:
    for f in files_ordered:
        fid.write('%s' % f) 

# create list of test images
files_ordered = np.sort(os.listdir(test_dir))
with open(order_of_test,'w') as fid:
    for f in files_ordered:
        fid.write('%s\r\n' % f)