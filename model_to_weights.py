#simple script to convert saved model into saved weights

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

print('load saved model')
model = load_model('model-256x256x1-centers.h5')

print('save weights')
model.save_weights('weights-256x256x1-centers.h5')
