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
		
#get all folder ids in each set
test_ids = next(os.walk(TEST_PATH))[1]
print(test_ids)
#get all the image data
sys.stdout.flush() #force python to write everything out, not keep in a buffer
	
# allocate memory
X_test = np.zeros((1000, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
img_counter=0

#load a movie
container = imread(TEST_PATH + VIDEO_NAME + '.tiff') #(frame, width,height)

i = 0
#for n,frame in enumerate(container.decode(video=0)):
for n in range(container.shape[0]):
	if(n%1==0):
		img = container[n]
		img = img/np.amax(img)
		img = 255*rgb2gray(img)
		img=np.expand_dims(img,axis=-1)
		img = img[:,:,:1]
		X_test[i] = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
		i = i + 1
print (i,'images')
testLength = i

#load model
fcrn_model = fcrna.Cell_Model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, 1)
model = Model(inputs=[fcrn_model.inputs], outputs=[fcrn_model.outputs])
#model.load_weights('weights-retrain1.h5')
model.load_weights('weights-256x256x1-centers.h5')

model.compile(optimizer='adam', loss='mean_squared_error')
preds_inner = model.predict(X_test[:testLength], verbose=1)

fig = plt.figure()

#need take the whole mask of the training set and break them down into individual cells
for n in range(testLength):
	
	#find predictions for the cell centers
	center_norm = preds_inner[n]/np.amax(preds_inner[n])
	center_thresh = (center_norm > 0.0).astype(np.uint8) #0.7 for 128,256 models
	center_labels = ndimage.label(center_thresh)[0]
	xList = []
	yList = []
	
	for i in range(1,center_labels.max()+1):
		if(np.sum(center_labels==i) > 10):
			#make a '+' for easy visualization
			new_center = ndimage.measurements.center_of_mass(center_labels==i)
			xList.append(new_center[1])
			yList.append(new_center[0])

	im = plt.plot(xList,yList,linestyle='None',marker='+',color='white')
	im = plt.imshow(np.squeeze(X_test[n]))
	plt.savefig('outImages/'+VIDEO_NAME+'_'+str(n)+'.png')
	plt.cla()
	
	
	