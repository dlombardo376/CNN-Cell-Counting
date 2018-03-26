import os
import sys

import numpy as np
from tqdm import tqdm #adds a progress bar for all for loops

from skimage.io import imread, imshow, imread_collection, concatenate_images #image processing
from skimage.transform import resize, rotate
from skimage.morphology import label
from skimage.util import invert
from skimage.color import rgb2gray
from skimage.segmentation import find_boundaries

from scipy import ndimage

import matplotlib.pyplot as plt	
	
def loadImagesCenters(TRAIN_PATH, MASK_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):

	#get all folder ids in each set
	train_ids = next(os.walk(TRAIN_PATH))[2]
	#train_ids = train_ids[:1]
	
	#print(train_ids)
	
	#get all the image data
	#multiply length of ids by 10, multiple frames per video
	X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
	Y_train = np.zeros((len(train_ids),IMG_HEIGHT,IMG_WIDTH,1),dtype=np.uint8)
	
	#make a gaussian centered on the mask, with stdev = 2
	maxD = 3
	sigma = 1
	gaussValues = np.zeros(maxD*2+1)
	for i in range(0,maxD*2+1,1):
		gaussValues[i] = np.exp(-((i-maxD)**2)/ (2*sigma*sigma))
	#print(gaussValues)
	print('Get and resize all train images and masks')
	sys.stdout.flush() #force python to write everything out, not keep in a buffer
	
	trainIndex = 0
	#loop through every movie
	for n,id_ in tqdm(enumerate(train_ids),total=len(train_ids)):
		path = TRAIN_PATH + id_
		
		#load the movie
		#container = av.open(path) #(frame, width,height)

		img = np.loadtxt(path,dtype = float)
		img = img/np.amax(img)
		img = 255*rgb2gray(img)
		img=np.expand_dims(img,axis=-1)
		img = img[:,:,:1]
		
		img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode = 'constant', preserve_range=True)
		X_train[n] = img
		
		#get mask file path for video
		mask = np.zeros((IMG_HEIGHT, IMG_WIDTH,1),dtype=np.float32)
		mpath = MASK_PATH + id_[:-6]

		#load the cell centers for the mask of this frame
		Xlist = np.loadtxt(mpath + "/" + "X-" + str(id_[-5:-4]) + ".txt",dtype=float)
		
		Ylist = np.loadtxt(mpath + "/" + "Y-" + str(id_[-5:-4]) + ".txt",dtype=float)

		#convert centers into density function
		density = np.zeros((IMG_HEIGHT, IMG_WIDTH,1),dtype=np.float32)
		for maskIndex in range(len(Xlist)):
			minX = np.maximum(int(Xlist[maskIndex]) - maxD,0)
			maxX = np.minimum(int(Xlist[maskIndex]) + maxD,IMG_WIDTH-1)
			minY = np.maximum(int(Ylist[maskIndex]) - maxD,0)
			maxY = np.minimum(int(Ylist[maskIndex]) + maxD,IMG_WIDTH-1)
		
			for i in range(minX,maxX+1,1):
				for j in range(minY,maxY+1,1):
					density[i,j,0] = 10*gaussValues[i-int(Xlist[maskIndex])+maxD]*gaussValues[j-int(Ylist[maskIndex])+maxD]
		mask= np.maximum(mask, density) #combine all mask files into one
		# if(n==0):print(np.average(mask))
		Y_train[n] = mask

		# plt.subplot(121)
		# plt.imshow(np.squeeze(X_train[n]))
		# backgroundMaskData = np.ma.masked_where(Y_train[n]<0.1,Y_train[n])
		# plt.imshow(np.squeeze(backgroundMaskData))
		# plt.subplot(122)
		# plt.imshow(np.squeeze(Y_train[n]))
		# plt.show()
		
	return X_train, Y_train
	
loadImagesCenters('snapshots/', 'maskCenters/',256, 256, 1)