#tutorial on setting up a unet model for kaggle submission
# by kjetil amdal-saevik: https://www.kaggle.com/keegil

import os
import sys
import random
import warnings

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm #adds a progress bar for all for loops
from itertools import chain

import av

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
TEST_PATH = 'data/'

warnings.filterwarnings('ignore',category=UserWarning,module='skimage')
seed=42
random.seed = seed
np.random.seed = seed

#get all folder ids in each set
test_ids = next(os.walk(TEST_PATH))[1]
print(test_ids)
#get all the image data
sys.stdout.flush() #force python to write everything out, not keep in a buffer
	
# Get and resize test images
X_test = np.zeros((1000, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
img_counter=0


container = av.open(TEST_PATH + 'Cap200_100416_Pos2_DIC and positions.avi') #(frame, width,height)

i = 0
for frame in container.decode(video=0):
	img = frame.to_image()
	npImg = np.asarray(img)
	#npImg = rgb2gray(npImg)
	#npImg=invert(npImg)
	#npImg=np.expand_dims(npImg,axis=-1)
	#print("inverted image ",img)
	npImg = npImg[:,:,:1]
	X_test[i] = resize(npImg, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
	i = i + 1
print (i,'images')

testLength = i

plt.imshow(np.squeeze(X_test[0]))
plt.show()

model_centers = load_model('model-256x256x1-centers.h5')
preds_inner = model_centers.predict(X_test[:testLength], verbose=1)
									   
#need take the whole mask of the training set and break them down into individual cells
#use skimage to find separate cells, may have trouble if they overlap
fig = plt.figure()
frames =[]#save out images to recreate video with predicted centers
for n in range(testLength):
	
	#find predictions for the cell centers
	center_norm = preds_inner[n]/np.amax(preds_inner[n])
	center_thresh = (center_norm > 0.7).astype(np.uint8) #0.7 for 128,256 models
	center_labels = ndimage.label(center_thresh)[0]
	centerList = []
	centerMap = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
	for i in range(1,center_labels.max()+1):
		if(np.sum(center_labels==i) > 10):
			new_center = ndimage.measurements.center_of_mass(center_labels==i)
			centerList.append((new_center[0],new_center[1]))
			centerMap[int(new_center[0]),int(new_center[1])] = 1
		#else:
		#	print('discard center')

	backgroundMaskData = np.ma.masked_where(center_thresh<0.5,center_thresh)
	combined = np.maximum(X_test[n]/255,center_thresh)
	im = plt.imshow(np.squeeze(combined))
	frames.append([im])
	#plt.imshow(np.squeeze(backgroundMaskData))
	#plt.show()

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
	
ani = animation.ArtistAnimation(fig,frames,interval=50,blit=True)
ani.save('anim_256.mp4',writer=writer)
#plt.show()