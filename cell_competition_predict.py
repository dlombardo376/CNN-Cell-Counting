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
from skimage.morphology import label,dilation, watershed
from skimage.util import invert
from skimage.color import rgb2gray
from skimage.segmentation import find_boundaries
from skimage.feature import peak_local_max

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
IMG_WIDTH = 128
IMG_HEIGHT = 128
MASK_WIDTH = 256
MASK_HEIGHT = 256
IMG_CHANNELS = 1
TRAIN_PATH = 'input/stage1_train/'
TEST_PATH = 'input/stage1_test/'

warnings.filterwarnings('ignore',category=UserWarning,module='skimage')
seed=42
random.seed = seed
np.random.seed = seed

def mean_iou(y_true,y_pred):
	#define metric function here
	prec = []
	for t in np.arange(0.5,1.0,0.05): #sweep over a range of thresholds
		y_pred = tf.to_int32(y_pred > t) #find all predicted values greater than threshold
		score, up_opt = tf.metrics.mean_iou(y_true,y_pred,2) #mean iou at given threshold
		K.get_session().run(tf.local_variables_initializer()) #to run the tensorflow operation
		with tf.control_dependencies([up_opt]):
			score = tf.identity(score)
		prec.append(score) #sum all the threshold mean iou's
	return K.mean(K.stack(prec),axis=0)

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
	lab_img = label(x)#label(x > cutoff) #skimage finds connected regions and labels them
	for i in range(1, lab_img.max() + 1):
		if(np.sum(lab_img==i)>10):
			yield rle_encoding(lab_img == i) #make a run length encoding of each segment
	
#get all folder ids in each set
test_ids = next(os.walk(TEST_PATH))[1]
#test_ids = test_ids[:5]

#get all the image data
sys.stdout.flush() #force python to write everything out, not keep in a buffer
	
# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
X_test_mask = np.zeros((len(test_ids), MASK_HEIGHT, MASK_WIDTH, IMG_CHANNELS), dtype=np.uint8)
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
	
	X_test_mask[n] = resize(img, (MASK_HEIGHT, MASK_WIDTH), mode='constant', preserve_range=True)
	img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
	X_test[n] = img
print('Done!')

model_centers = load_model('model-128x128x1-centers.h5')
preds_inner = model_centers.predict(X_test, verbose=1)

model_mask = load_model('model-256x256x1-dataAug-2.h5',custom_objects={'mean_iou':mean_iou})
preds_mask = model_mask.predict(X_test_mask, verbose=1)

# Create list of upsampled test masks, so that the masks match the original image sizes
preds_inner_upsampled = []
preds_mask_upsampled = []
for i in range(len(preds_mask)):
	preds_inner_upsampled.append(resize(np.squeeze(preds_inner[i]), 
									   (sizes_test[i][0], sizes_test[i][1]), 
									   mode='constant', preserve_range=True))
	preds_mask_upsampled.append(resize(np.squeeze(preds_mask[i]), 
									   (sizes_test[i][0], sizes_test[i][1]), 
									   mode='constant', preserve_range=True))

#need take the whole mask of the training set and break them down into individual cells
#use skimage to find separate cells, may have trouble if they overlap
new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
	#clean up the mask and make labels
	mask_thresh = (preds_mask_upsampled[n] > 0.5)
	mask_labels = label(mask_thresh)
	
	#find predictions for the cell centers
	center_norm = preds_inner_upsampled[n]/np.amax(preds_inner_upsampled[n])
	center_thresh = (center_norm > 0.5).astype(np.uint8)
	#center_thresh = np.minimum(center_thresh,mask_thresh)
	center_labels = label(center_thresh)
	centers = np.zeros((sizes_test[n][0], sizes_test[n][1]), dtype=np.uint8)
	for i in range(1,center_labels.max()+1):
		new_center = ndimage.measurements.center_of_mass(center_labels==i)
		centers[int(new_center[0]),int(new_center[1])] = 1 

	#see if a single mask ever is without centers
	for i in range(1, mask_labels.max()+1):
		current_mask = (mask_labels == i).astype(np.uint8)
		intersections = np.sum(np.multiply(current_mask,centers))
		if(intersections == 0): 
			#print(n, intersections)
			new_center = ndimage.measurements.center_of_mass(current_mask)
			centers[int(new_center[0]),int(new_center[1])] = 1
		
	distance = ndimage.distance_transform_edt(mask_thresh)
	local_maxi = peak_local_max(distance, labels=mask_thresh,footprint=np.ones((3, 3)),indices=False)
	markers = ndimage.label(centers)[0]#ndimage.label(local_maxi)[0]
	cell_watershed = watershed(-distance,markers,mask=mask_thresh)
	rle = list(prob_to_rles(cell_watershed))
	rles.extend(rle)
	new_test_ids.extend([id_] * len(rle))
	
	if(n<2):
		print(id_)
		pltImg = resize(X_test[n], (sizes_test[n][0], sizes_test[n][1]), mode='constant', preserve_range=True)
		plt.subplot(221)
		imshow(np.squeeze(pltImg))
		plt.subplot(222)
		imshow(np.squeeze(center_thresh))
		plt.subplot(223)
		imshow(np.squeeze(mask_labels))
		
		plt.subplot(224)
		imshow(cell_watershed)
		#plt.scatter(*zip(*centers))
		#plt.axis([0,sizes_test[n][1], sizes_test[n][0],0])
		plt.show()

	
# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-1.csv', index=False)