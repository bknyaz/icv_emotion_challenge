#!/bin/bash
# Author: Boris Knyazev
# Solution 1 - Matlab based

# Testing environment:
# Ubuntu 16.04 LTS
# gcc 5.4.0
# dlib
# Matlab R2015b
# MatConvNet
# VLFeat
# 32GB RAM
# Xeon CPU E5-2620 v3 @ 2.40GHz
# Optional: CUDA 7.5, cuDNN-v5, NVIDIA GTX 980 Ti

#------------------------------
# Set your data folders and paths here
## data root folder
DATA_DIR="/home/boris/Project/data/images/icv_emotions"
## folders with all original images
TRAINING_DIR=$DATA_DIR"/Training"
TEST_DIR=$DATA_DIR"/Validation"
FINAL_TEST_DIR=$DATA_DIR"/Test"
## images list files
TRAINING_IMG=$DATA_DIR"/training_new.txt"
TEST_IMG=$DATA_DIR"/order_of_validation.txt"
TEST_IMG_LABELS=$DATA_DIR"/validation.txt"
## output file
SUBMISSION_FILE=$DATA_DIR"/predictions.txt"

## These folders will be created and face images will be kept there
TRAINING_DIR_dlib=$DATA_DIR"/Training_dlib"
TRAINING_DIR_dlib_no_align=$DATA_DIR"/Training_dlib_no_align"
TEST_DIR_dlib=$DATA_DIR"/Validation_dlib"
TEST_DIR_dlib_no_align=$DATA_DIR"/Validation_dlib_no_align"

### For the final test these folders and files also will be created
# list of all preprocessed training images
TRAINING_IMG_ALL=$DATA_DIR"/training_new_plus_validation.txt"
# all preprocessed training images
TRAINING_DIR_dlib_ALL=$DATA_DIR"/Training_dlib_all"
FINAL_TEST_DIR_dlib=$DATA_DIR"/Test_dlib"
FINAL_TEST_DIR_dlib_no_align=$DATA_DIR"/Test_dlib_no_align"
FINAL_TEST_IMG=$DATA_DIR"/order_of_test.txt"

## Matlab must be install
MATLAB="/usr/local/MATLAB/R2016a/bin/matlab"
#------------------------------

# Once the folders and paths above are set correctly, the rest of the script should be run automatically

# All necessary code, tools and models will be saved in the current directory

#------------------------------
# download, unzip shape_predictor_68_face_landmarks.dat
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
# download dlib and compile dlib examples
## use my fork of dlib, in which I modified face_landmark_detection_ex.cpp a little bit to make it possible to save face images
git clone https://github.com/bknyaz/dlib
cd dlib/examples
mkdir build; cd build; cmake .. ; cmake --build .
mkdir $TRAINING_DIR_dlib; mkdir $TRAINING_DIR_dlib_no_align; mkdir $TEST_DIR_dlib; mkdir $TEST_DIR_dlib_no_align; mkdir $FINAL_TEST_DIR_dlib; mkdir $FINAL_TEST_DIR_dlib_no_align
## Extract and align training, validation and test faces in parallel
## This will take most of the time in our pipeline (about 3 hours)
./face_landmark_detection_ex ./../../shape_predictor_68_face_landmarks.dat $TRAINING_DIR $TRAINING_DIR_dlib $TRAINING_DIR_dlib_no_align 1280 \
& ./face_landmark_detection_ex ./../../shape_predictor_68_face_landmarks.dat $TEST_DIR $TEST_DIR_dlib $TEST_DIR_dlib_no_align 1280 \
& ./face_landmark_detection_ex ../../../shape_predictor_68_face_landmarks.dat $FINAL_TEST_DIR $FINAL_TEST_DIR_dlib $FINAL_TEST_DIR_dlib_no_align 1280

cd ../../..

#------------------------------

# download and compile vlfeat, matconvnet, liblinear
## vlfeat 
wget http://www.vlfeat.org/download/vlfeat-0.9.20-bin.tar.gz && tar xzf vlfeat-0.9.20-bin.tar.gz && mv vlfeat-0.9.20 vlfeat

## matconvnet
wget https://github.com/vlfeat/matconvnet/archive/v1.0-beta23.tar.gz && tar xzf v1.0-beta23.tar.gz && mv matconvnet-1.0-beta23 matconvnet
### it's fine to build just a CPU version, but it will be slower
$MATLAB -nodisplay -nosplash -nodesktop -nojvm -r "cd matconvnet; addpath matlab; vl_compilenn; quit"

## liblinear
### use my fork of liblinear, in which I modified a few files to use openmp so that SVM models are trained much faster if you have a multi-core CPU
git clone https://github.com/bknyaz/liblinear
$MATLAB -nodisplay -nosplash -nodesktop -nojvm -r "run('liblinear/matlab/make.m'); quit"

# Download my code for emotion recognition
## Model
git clone https://github.com/bknyaz/autocnn_unsup
## ICV data processing and parameters
git clone https://github.com/bknyaz/icv_emotion_challenge
cp icv_emotion_challenge/solution1/icv.m autocnn_unsup/experiments/
cp icv_emotion_challenge/solution1/icv_prediction.m autocnn_unsup/experiments/
cp icv_emotion_challenge/solution1/icv_write_submission.m autocnn_unsup/experiments/
cp icv_emotion_challenge/solution1/autocnn_icv.m autocnn_unsup/
cp icv_emotion_challenge/solution1/autocnn_prediction.m autocnn_unsup/

# Concatenate training and validation data
# This script will create files $TRAINING_IMG_ALL and $FINAL_TEST_IMG
python concat_train_val.py $TRAINING_IMG $TEST_IMG $TEST_IMG_LABELS $TRAINING_IMG_ALL $FINAL_TEST_DIR_dlib $FINAL_TEST_IMG
mkdir $TRAINING_DIR_dlib_ALL
cp $TRAINING_DIR_dlib/* $TRAINING_DIR_dlib_ALL/
cp $TEST_DIR_dlib/* $TRAINING_DIR_dlib_ALL/

# Train a model in Matlab and save predictions
## Set the number of threads equal to the number of physical cores (6 in my case)
export OMP_NUM_THREADS=6
## Define the network architecture
network_arch="1024c15-12p-conv0"
## Run main script and write the results to $SUBMISSION_FILE (predictions.txt)
$MATLAB -nodisplay -nosplash -nodesktop -nojvm -r "cd autocnn_unsup/experiments; icv('$DATA_DIR','$TRAINING_DIR_dlib_ALL','$TRAINING_IMG_ALL','$FINAL_TEST_DIR_dlib','$FINAL_TEST_IMG','$SUBMISSION_FILE','$network_arch',25); quit" -logfile boris_autocnn_train_$network_arch.log

# Trained model icv_22969_5folds_1024c15-12p-conv0.mat
# md5sum: d3ffe11c70e21c81aa844d259e578578
# Example how to get predictions for some new data using this model:
# model="icv_22969_5folds_1024c15-12p-conv0.mat"
# $MATLAB -nodisplay -nosplash -nodesktop -nojvm -r "cd autocnn_unsup/experiments; icv_prediction('$FINAL_TEST_DIR_dlib', '$FINAL_TEST_IMG', '$model', '$SUBMISSION_FILE') 
