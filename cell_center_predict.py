#tutorial on setting up a unet model for kaggle submission
# by kjetil amdal-saevik: https://www.kaggle.com/keegil

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm #adds a progress bar for all for loops
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images #image processing
from skimage.transform import resize
from skimage.morphology import label
from skimage.util import invert
from skimage.color import rgb2gray

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

from scipy import ndimage

import tensorflow as tf

#define global constants
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1
TRAIN_PATH = 'input/stage1_train/'
TEST_PATH = 'input/stage1_test/'
N_CLASSES = 1

warnings.filterwarnings('ignore',category=UserWarning,module='skimage')
seed=42
random.seed = seed
np.random.seed = seed

def prob_to_centers(x, cutoff=0.5):
	lab_img = label(x > cutoff) #skimage finds connected regions and labels them
	centers = []
	cx = np.zeros(lab_img.max(),dtype=int)
	cy = np.zeros(lab_img.max(),dtype=int)
	for i in range(1, lab_img.max() + 1):
		cxy = np.round(ndimage.measurements.center_of_mass(lab_img == i))
		centers.append(cxy)
		cx[i-1] = int(cxy[0])
		cy[i-1] = int(cxy[1])
        #yield rle_encoding(lab_img == i) #make a run length encoding of each segment
	return cx,cy
#get all folder ids in each set
test_ids = next(os.walk(TEST_PATH))[1]

#get all the image data
sys.stdout.flush() #force python to write everything out, not keep in a buffer
	
# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
	path = TEST_PATH + id_
	img = imread(path + '/images/' + id_ + '.png')[:,:,:3]
	sizes_test.append([img.shape[0], img.shape[1]])
	
	if(IMG_CHANNELS == 1): #actually want grayscale
			if(img[0,0,0]!=img[0,0,1]):  #invert contrast on fluoures
				img = 255*rgb2gray(img)
				img=invert(img)
				img=np.expand_dims(img,axis=-1)
				#print("inverted image ",img)
			img = img[:,:,:1]
			
	img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
	X_test[n] = img
print('Done!')

model = load_model('model-dsbowl2018-1.h5')
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_test_t = (preds_test > 0.75).astype(np.uint8)

#flatten the predictions to one layer
preds_test_flat = preds_test_t

# Create list of upsampled test masks, so that the masks match the original image sizes
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test_flat[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))

#need take the whole mask of the training set and break them down into individual cells
#use skimage to find separate cells, may have trouble if they overlap
new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
	cx,cy = prob_to_centers(preds_test_upsampled[n])
	#rles.extend(rle)
	#new_test_ids.extend([id_] * len(rle))
	if(n<10):
		pltImg = resize(X_test[n], (sizes_test[n][0], sizes_test[n][1]), mode='constant', preserve_range=True)
		plt.subplot(221)
		imshow(np.squeeze(pltImg))
		plt.subplot(222)
		plt.plot(cy,cx,'ro')
		plt.axis([0,sizes_test[n][1],sizes_test[n][0],0])
		plt.subplot(223)
		imshow(np.squeeze(preds_test_upsampled[n]))
		plt.show()