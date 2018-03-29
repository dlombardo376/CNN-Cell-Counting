#tutorial on setting up a unet model for kaggle submission
# by kjetil amdal-saevik: https://www.kaggle.com/keegil


#background currently predicts on nothing, give it something
	#gray scale values of images?
#break up masks by size

import random
import warnings

import numpy as np

from PIL import Image

import video_input3D as vInput
import cell_fcrna3D as fcrna

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras import backend as K

import tensorflow as tf

#define global constants
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1
IMG_FRAMES = 8
IMG_CLASSES = 1
TRAIN_PATH = 'snapshots3D/'
MASK_PATH = 'maskCenters3D/'

warnings.filterwarnings('ignore',category=UserWarning,module='skimage')
seed=42
#random.seed = seed
#np.random.seed = seed

#create subclass to save prediction images per epoch
class predictEpoch(Callback):
    def on_epoch_end(self,epoch, logs={}):
        sample = next(train_generator)
       
        #get sample images and predictions
        ims = sample[0][:10]
        gt = 255*sample[1][:10]
        preds = 255*model.predict(ims)
       
        #combine image, truth, and prediction into array
        nx = np.size(ims,2)
        combinedImg = np.concatenate((ims[...,0].reshape([-1,nx]),
                                      gt[...,0].reshape([-1,nx]),
                                      preds[...,0].reshape([-1,nx])), axis=1)
       
        #save image
        path = 'predictions/epoch_%i.jpg'%(epoch+1)
        Image.fromarray(combinedImg.round().astype(np.uint8)).save(path,
                                       'JPEG', dpi=[300,300], quality=90)
					
					
X_train, Y_train = vInput.loadImagesCenters(TRAIN_PATH, MASK_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_FRAMES, IMG_CHANNELS)
#X_train, Y_train = cell_input.loadImagesBoundaries(TRAIN_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,IMG_CLASSES)

X_val = X_train[(len(X_train)-20):]
Y_val = Y_train[(len(Y_train)-20):]	

#data augmentation
data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
					 rotation_range = 0.,
					 height_shift_range = 0.1,
					 width_shift_range = 0.1,
                     horizontal_flip=True,
					 vertical_flip=True,
					 fill_mode='reflect',cval=0.1)
					 

# plotCounter=0
# for X_batch, Y_batch in train_generator:
	# plt.subplot(221)
	# plt.imshow(np.squeeze(X_batch[0]))
	
	# plt.subplot(222)
	# plt.imshow(np.squeeze(Y_batch[0]))
	
	# plt.show()
	# plotCounter = plotCounter+1;
	# if(plotCounter>3):
		# break
	
fcrn_model = fcrna.Cell_Model(IMG_FRAMES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, IMG_CLASSES)

#use the intersections over union metric for training, as defined by the competition
model = Model(inputs=[fcrn_model.inputs], outputs=[fcrn_model.outputs])
model.compile(optimizer='adam', loss='mean_squared_error')
#model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()

#optimizer is some variant of gradient descent. To prevent overfitting we can stop the fit process
#patience = # of epochs with no improvement to model
earlystopper = EarlyStopping(patience = 5, verbose=1)
predictions = predictEpoch()

#occasionally save the model as h5 output
checkpointer = ModelCheckpoint('model-dsbowl2018-center-1.h5',verbose=1,save_best_only=True)

results = model.fit(x=X_train, y=Y_train,epochs=50, batch_size = 16, callbacks=[earlystopper, checkpointer, predictions], validation_data = (X_val, Y_val))