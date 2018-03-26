#tutorial on setting up a unet model for kaggle submission
# by kjetil amdal-saevik: https://www.kaggle.com/keegil

import os
import sys
import random
import warnings

import numpy as np

import cell_fcrna as fcrna
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
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1
TEST_PATH = 'videos/'
VIDEO_NAME = 'Cap200_100416_pos3_C0'
#VIDEO_NAME = 'Cap2_100516_pos3_C0'

warnings.filterwarnings('ignore',category=UserWarning,module='skimage')
seed=42
random.seed = seed
np.random.seed = seed

def GetClosestIndex(testX,testY,xList,yList):
	xArray = np.asarray(xList)
	yArray = np.asarray(yList)
	xDis = xArray - testX
	yDis = yArray - testY
	rValues = xDis*xDis + yDis*yDis
	minIndex = np.argmin(rValues)
	return minIndex, rValues[minIndex]

	
#get all folder ids in each set
test_ids = next(os.walk(TEST_PATH))[1]
print(test_ids)
#get all the image data
sys.stdout.flush() #force python to write everything out, not keep in a buffer
	
print('Getting and resizing test images ... ')

#load a movie
container = imread(TEST_PATH + VIDEO_NAME + '.tiff') #(frame, width,height)

#load model
fcrn_model = fcrna.Cell_Model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, 1)
model = Model(inputs=[fcrn_model.inputs], outputs=[fcrn_model.outputs])
model.load_weights('weights-retrain2.h5')
#model.load_weights('weights-retrain1.h5')
#model.load_weights('weights-256x256x1-centers.h5')

model.compile(optimizer='adam', loss='mean_squared_error')

fig = plt.figure()

i = 0
#for n,frame in enumerate(container.decode(video=0)):
#for n in range(container.shape[0]):
timeSet = 5
midTime = 2
X_test = np.zeros((timeSet, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)

for n in range(0,container.shape[0]-190,timeSet):
	all_labels = []
	for aveIndex in range(timeSet):
		img = container[n+aveIndex]
		img = img/np.amax(img)
		img = 255.0*rgb2gray(img)
		img=np.expand_dims(img,axis=-1)
		img = img[:,:,:1]
		X_test[aveIndex] = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
	
	preds_inner = model.predict(X_test[0:timeSet])
	
	for i in range(timeSet):
		center_norm = preds_inner[i]/np.amax(preds_inner[i])
		center_thresh = (center_norm > 0.0).astype(np.uint8) #0.7 for 128,256 models
		all_labels.append(ndimage.label(center_thresh)[0])
		
	
	#find predictions for the cell centers
	center_norm = preds_inner[midTime]/np.amax(preds_inner[midTime])
	center_thresh = (center_norm > 0.0).astype(np.uint8) #0.7 for 128,256 models
	center_labels = ndimage.label(center_thresh)[0]
	xList = []
	yList = []
	
	for j in range(timeSet):
		for i in range(1,all_labels[j].max()+1):
			if(np.sum(all_labels[j]==i) > 10):
				#make a '+' for easy visualization
				new_center = ndimage.measurements.center_of_mass(all_labels[j]==i)
				
				if(j>0):
					index,dis = GetClosestIndex(new_center[1],new_center[0],xList,yList)
					if(dis>15):
						xList.append(new_center[1])
						yList.append(new_center[0])
				else:
					xList.append(new_center[1])
					yList.append(new_center[0])
						
	im = plt.plot(xList,yList,linestyle='None',marker='+',color='white')
	im = plt.imshow(np.squeeze(X_test[midTime])) #plot the middle image
	#plt.savefig('outImages/'+VIDEO_NAME+'_'+str(n)+'.png')
	#plt.cla()
	plt.show()
	
	
	