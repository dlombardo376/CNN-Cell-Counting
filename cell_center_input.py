import os
import sys

import numpy as np
from tqdm import tqdm #adds a progress bar for all for loops

from skimage.io import imread, imshow, imread_collection, concatenate_images #image processing
from skimage.transform import resize, rotate
from skimage.morphology import label
from skimage.util import invert
from skimage.color import rgb2gray

from scipy import ndimage

import matplotlib.pyplot as plt

def loadImages(TRAIN_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):

	#get all folder ids in each set
	train_ids = next(os.walk(TRAIN_PATH))[1]
	#train_ids = train_ids[:10]
	
	#get all the image data
	X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
	Y_train = np.zeros((len(train_ids),IMG_HEIGHT,IMG_WIDTH,1),dtype=np.uint8)
	
	#make a gaussian centered on the mask, with stdev = 2
	maxD = 10
	sigma = 1
	gaussValues = np.zeros(maxD*2+1)
	for i in range(0,maxD*2+1,1):
		gaussValues[i] = np.exp(-((i-maxD)**2)/ (2*sigma*sigma)) / np.sqrt(2*np.pi*sigma*sigma)
	#print(gaussValues)
	print('Get and resize all train images and masks')
	sys.stdout.flush() #force python to write everything out, not keep in a buffer
	for n,id_ in tqdm(enumerate(train_ids),total=len(train_ids)):
		path = TRAIN_PATH + id_
		img = imread(path + '/images/' + id_ + '.png')[:,:,:3] #always read whole image
		
		if(IMG_CHANNELS == 1): #actually want grayscale
			if(img[0,0,0]!=img[0,0,1]):  #invert contrast on fluoures
				img = 255*rgb2gray(img)
				img=invert(img)
				img=np.expand_dims(img,axis=-1)
				#print("inverted image ",img)
			img = img[:,:,:1]
				
		img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode = 'constant', preserve_range=True)
		X_train[n] = img
		mask = np.zeros((IMG_HEIGHT, IMG_WIDTH,1),dtype=np.float32)
		for mask_file in next(os.walk(path + '/masks/'))[2]:
			mask_ = imread(path+'/masks/'+mask_file) #read each mask for the image
			mask_ = resize(mask_,(IMG_HEIGHT,IMG_WIDTH),mode='constant',preserve_range=True)
			density = np.zeros((IMG_HEIGHT, IMG_WIDTH,1),dtype=np.float32)
			
			if(np.amax(mask_)>0):
				center = np.round(ndimage.measurements.center_of_mass(mask_))
				
				minX = np.maximum(int(center[0]) - maxD,0)
				maxX = np.minimum(int(center[0]) + maxD,IMG_WIDTH-1)
				minY = np.maximum(int(center[1]) - maxD,0)
				maxY = np.minimum(int(center[1]) + maxD,IMG_WIDTH-1)
				
				for i in range(minX,maxX+1,1):
					for j in range(minY,maxY+1,1):
						density[i,j,0] = 100*gaussValues[i-int(center[0])+maxD]*gaussValues[j-int(center[1])+maxD]
				mask= mask + density #combine all mask files into one
		Y_train[n] = mask

	return X_train, Y_train
	