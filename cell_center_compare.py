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
	
def compareCenters(centerList,maskList):
	numDiff = (len(centerList) - len(maskList))
	aveDist = 0
	
	trueList = maskList[:]
	testList = centerList[:]
	distanceList = []

	#print(len(trueList))
	#print(len(testList))
	
	while(len(testList)>0):
		maxIndex = -1
		maxDist = 100000
		for i in range(len(trueList)):
			xDist = testList[0][0] - trueList[i][0]
			yDist = testList[0][1] - trueList[i][1]
			newDist = np.sqrt(xDist**2 + yDist**2)
			if(newDist < maxDist):
				maxDist = newDist
				maxIndex = i
				#print('found max',maxIndex,maxDist)
		
		if(maxIndex>-1):
			distanceList.append(maxDist)
			aveDist = aveDist + maxDist
			del trueList[maxIndex]
		del testList[0]
	
	#print(len(distanceList))
	aveDist = aveDist/len(distanceList)	
	
	return aveDist,numDiff,min(distanceList),max(distanceList)
	
#get all folder ids in each set
train_ids = next(os.walk(TRAIN_PATH))[1]
#train_ids = train_ids[23:26]

#get all the image data
sys.stdout.flush() #force python to write everything out, not keep in a buffer
	
# Get and resize test images
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
X_train_mask = np.zeros((len(train_ids), MASK_HEIGHT, MASK_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
	path = TRAIN_PATH + id_
	img = imread(path + '/images/' + id_ + '.png')[:,:,:3]
	sizes_test.append([img.shape[0], img.shape[1]])
	
	if(IMG_CHANNELS == 1): #actually want grayscale
			if(img[0,0,0]!=img[0,0,1]):  #invert contrast on fluoures
				img = 255*rgb2gray(img)
				img=invert(img)
				img=np.expand_dims(img,axis=-1)
				#print("inverted image ",img)
			img = img[:,:,:1]
	
	X_train_mask[n] = resize(img, (MASK_HEIGHT, MASK_WIDTH), mode='constant', preserve_range=True)
	img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
	X_train[n] = img
		
print('Done!')

model_centers = load_model('model-128x128x1-centers.h5')
preds_inner = model_centers.predict(X_train, verbose=1)

model_mask = load_model('model-256x256x1-dataAug-2.h5',custom_objects={'mean_iou':mean_iou})
preds_mask = model_mask.predict(X_train_mask, verbose=1)

# Create list of upsampled test masks, so that the masks match the original image sizes
preds_inner_upsampled = []
preds_mask_upsampled = []
for i in range(len(preds_inner)):
	preds_inner_upsampled.append(resize(np.squeeze(preds_inner[i]), 
									   (sizes_test[i][0], sizes_test[i][1]), 
									   mode='constant', preserve_range=True))
									   
	preds_mask_upsampled.append(resize(np.squeeze(preds_mask[i]), 
									   (sizes_test[i][0], sizes_test[i][1]), 
									   mode='constant', preserve_range=True))

#need take the whole mask of the training set and break them down into individual cells
#use skimage to find separate cells, may have trouble if they overlap
outFile = "center-compare-output.txt"
file = open(outFile,'w+')
file.write("averageDistance max min missed total\n")	
for n, id_ in enumerate(train_ids):
		
	if(n%10==0):
		print('checking image',n,id_)
	
	#clean up the mask and make labels
	mask_norm = preds_mask_upsampled[n]/np.amax(preds_inner_upsampled[n])
	mask_thresh = (mask_norm > 0.55).astype(np.uint8)
	mask_labels = label(mask_thresh)
	
	#find predictions for the cell centers
	center_norm = preds_inner_upsampled[n]/np.amax(preds_inner_upsampled[n])
	center_thresh = (center_norm > 0.7).astype(np.uint8)
	center_labels = ndimage.label(center_thresh)[0]
	centerList = []
	centerMap = np.zeros((sizes_test[n][0], sizes_test[n][1]), dtype=np.uint8)
	for i in range(1,center_labels.max()+1):
		if(np.sum(center_labels==i) > 10):
			new_center = ndimage.measurements.center_of_mass(center_labels==i)
			centerList.append((new_center[0],new_center[1]))
			centerMap[int(new_center[0]),int(new_center[1])] = 1
		#else:
		#	print('discard center')

		
	#remove any centers that are outside the mask
	centerMap = np.minimum(centerMap, mask_thresh)
	
	#see if a single mask ever is without centers
	for i in range(1, mask_labels.max()+1):
		current_mask = (mask_labels == i).astype(np.uint8)
		intersections = np.sum(np.multiply(current_mask,centerMap))
		if(intersections == 0 and np.sum(current_mask)>100): 
			#print(n, intersections)
			new_center = ndimage.measurements.center_of_mass(current_mask)
			#print('new center',new_center)
			centerList.append((new_center[0],new_center[1]))
		
	#find centers of given mask
	path = TRAIN_PATH + id_
	maskList = []
	for mask_file in next(os.walk(path + '/masks/'))[2]:
		mask_ = imread(path+'/masks/'+mask_file) #read each mask for the image
		
		if(np.amax(mask_)>0):
			centerMask = ndimage.measurements.center_of_mass(mask_)
			#print(mask_file,centerMask)
			maskList.append(centerMask)
			
	#compare the predicted centers with the mask centers
	#print(centerList)
	#print(maskList)
	aveDist, numDiff, minD, maxD = compareCenters(centerList,maskList)
	file.write(str(aveDist)+ ' '+str(maxD)+' '+str(minD)+' '+str(numDiff)+' '+str(len(maskList))+"\n")	
	
	if(n<5):
		print(id_, numDiff,str(len(maskList)))
		print(np.amax(mask_thresh))
		pltImg = resize(X_train[n], (sizes_test[n][0], sizes_test[n][1]), mode='constant', preserve_range=True)
		plt.subplot(221)
		imshow(np.squeeze(pltImg))
		plt.subplot(222)
		imshow(np.squeeze(center_thresh))
		plt.subplot(223)
		imshow(mask_thresh)
		plt.show()
		
file.close()