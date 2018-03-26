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

from skimage.io import imread, imshow, imread_collection, concatenate_images #image processing
from skimage.transform import resize
from skimage.morphology import label,dilation
from skimage.util import invert
from skimage.color import rgb2gray
from skimage.segmentation import find_boundaries

from scipy import ndimage

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

fig = plt.figure()
frames = []
#need take the whole mask of the training set and break them down into individual cells
for n in range(2,testLength):
	velocity = np.absolute(X_test[n] - X_test[n-1])
	velocity = velocity - np.amin(velocity)
	velocity = velocity/np.amax(velocity)
	#velocity = (velocity > 0.2)
	velocity = 255*velocity
	im = plt.imshow(np.squeeze(velocity), animated=True)
	im_standard = plt.imshow(np.squeeze(X_test[n]),animated=True)
	#plt.show()
	frames.append([im])
	
im_ani = animation.ArtistAnimation(fig, frames, interval=50, repeat_delay=1000, blit=True)
plt.show()
	