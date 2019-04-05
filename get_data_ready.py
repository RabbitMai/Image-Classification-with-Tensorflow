#import all the necessary libraries
import os, shutil
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from keras.utils import to_categorical

%matplotlib inline
np.random.seed(1)

#dataset was uncompressed
orginal_dataset_dir = 'C:/Users/wmai/Kaggle_ML_Practice/dogs_vs_cats'

#to store smaller dataset for this project
base_dir = 'C:/Users/wmai/Kaggle_ML_Practice/dogs_cats_small'
os.mkdir(base_dir)

#directory for training (create a subfolder inside dogs_cats_small)
train_dir = os.path.join(base_dir,'train')
os.mkdir(train_dir)

#directory for testing
test_dir = os.path.join(base_dir,'test')
os.mkdir(test_dir)


#copy 1250 cat images to train folder
fnames_cat = ['cat.{}.jpg'.format(i) for i in range(1250)]

for fname in fnames_cat:
    src_dir = os.path.join(orginal_dataset_dir+'/train', fname)
    target_dir = os.path.join(train_dir, fname)
    shutil.copyfile(src_dir, target_dir)


#copy 1250 dog images to train folder
fnames_dog = ['dog.{}.jpg'.format(i) for i in range(1250)]

for fname in fnames_dog:
    src_dir = os.path.join(orginal_dataset_dir+'/train', fname)
    target_dir = os.path.join(train_dir, fname)
    shutil.copyfile(src_dir, target_dir)


#copy next 250 cat images to test folder
fnames_cat_test = ['cat.{}.jpg'.format(i) for i in range(1250,1500)]

for fname in fnames_cat_test:
    src_dir = os.path.join(orginal_dataset_dir+'/train', fname)
    target_dir = os.path.join(test_dir, fname)
    shutil.copyfile(src_dir, target_dir)


#copy next 250 dog images to test folder
fnames_dog_test = ['dog.{}.jpg'.format(i) for i in range(1250,1500)]

for fname in fnames_dog_test:
    src_dir = os.path.join(orginal_dataset_dir+'/train', fname)
    target_dir = os.path.join(test_dir, fname)
    shutil.copyfile(src_dir, target_dir)
