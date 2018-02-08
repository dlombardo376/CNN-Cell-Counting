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
from skimage.morphology import label,dilation
from skimage.util import invert
from skimage.color import rgb2gray
from skimage.segmentation import find_boundaries

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
IMG_WIDTH = 96
IMG_HEIGHT = 96
IMG_CHANNELS = 1
TRAIN_PATH = 'input/stage1_train/'
TEST_PATH = 'input/stage1_test/'
N_CLASSES = 2

warnings.filterwarnings('ignore',category=UserWarning,module='skimage')
seed=42
random.seed = seed
np.random.seed = seed
def growCells(in_density,bound_density, width, height):
	#make a starting list of cells by labelling "inner" regions
	nuclei = label(in_density > 0.9)
	allCells = np.zeros((nuclei.max(),width, height))
	cellCount = 0
	
	#while the list is greater than zero:
	while nuclei.max()>0:
		#choose one cell
		cell = (nuclei == 1)
		
		#calculate the edge
		cell_boundary = find_boundaries(cell,mode='inner')
		
		#calculate the probability this edge being inner, and of being boundary
		in_prob = np.average(in_density[cell>0])
		bound_prob = np.average(bound_density[cell>0])
		
		#grow cell, maybe by dilations method, and get new edge
		new_cell = dilation(cell)
		
		#repeat until boundary probability stops increasing
		counter = 0
		while(counter<width):
			counter = counter + 1
			new_in_prob = np.average(in_density[new_cell>0])
			new_bound_prob = np.average(bound_density[new_cell>0])
			if(new_bound_prob > bound_prob):
				cell = new_cell
				new_cell = dilation(cell)
			else:
				break
		#if cell intersects another nuclei, delete that nuclei from the list
		
		#save grown cell
		allCells[cellCount] = cell
		cellCount = cellCount + 1
		
		#remove cell from list of starting nuclei
		nuclei = nuclei - 1
		nuclei = np.maximum(nuclei,0)
	return allCells, cellCount
	
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
preds_nuclei = (preds_test[:,:,:,0] > 0.5).astype(np.uint8)
preds_inner = preds_test[:,:,:,0]
preds_bound = preds_test[:,:,:,1]

# Create list of upsampled test masks, so that the masks match the original image sizes
preds_inner_upsampled = []
preds_bound_upsampled = []
for i in range(len(preds_test)):
	preds_inner_upsampled.append(resize(np.squeeze(preds_inner[i]), 
									   (sizes_test[i][0], sizes_test[i][1]), 
									   mode='constant', preserve_range=True))
	preds_bound_upsampled.append(resize(np.squeeze(preds_bound[i]), 
									   (sizes_test[i][0], sizes_test[i][1]), 
									   mode='constant', preserve_range=True))

#need take the whole mask of the training set and break them down into individual cells
#use skimage to find separate cells, may have trouble if they overlap
new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
	#rles.extend(rle)
	#new_test_ids.extend([id_] * len(rle))
	if(n<5):
		allCells, cellCount = growCells(preds_inner_upsampled[n],
					preds_bound_upsampled[n],sizes_test[n][0], sizes_test[n][1])
		pltImg = resize(X_test[n], (sizes_test[n][0], sizes_test[n][1]), mode='constant', preserve_range=True)
		plt.subplot(221)
		imshow(np.squeeze(pltImg))
		plt.subplot(222)
		imshow(np.squeeze(preds_inner_upsampled[n]))
		plt.subplot(223)
		imshow(np.squeeze(preds_bound_upsampled[n]))
		
		plotCells = np.zeros((sizes_test[n][0], sizes_test[n][1]))
		for i in range(cellCount):
			plotCells = np.maximum(plotCells,allCells[i]*(i+1))
		plt.subplot(224)
		imshow(plotCells)
		plt.show()