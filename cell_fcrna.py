from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

class FCRN_A_Class:
	def __init__(self, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, IMG_CLASSES):
		#build the keras model of unet
		#source: U-Net: Convolutional Networks for Biomedical Image Segmentation, by Ronneberger et al
		#make a tensor for the inputs
		self.inputs = Input((IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))

		#keras defines layers of a model, such as lambda, conv2d, etc..
		s = Lambda(lambda x: x/255)(self.inputs) #128x128x1

		#conv2d(filters, kernel_size,options)

		c1 = Conv2D(32,(3,3),activation='relu',kernel_initializer='Orthogonal',padding='same')(s) #9 weights * 1 channel * 32 features + 1 bias * 32 features = 320
		p1 = MaxPooling2D((2,2))(c1)

		c2 = Conv2D(64,(3,3),activation='relu', kernel_initializer='Orthogonal', padding='same') (p1)#9 * 32 * 64 + 1 * 64 = 18496 parameters
		p2 = MaxPooling2D((2, 2)) (c2)

		c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='Orthogonal', padding='same') (p2)
		p3 = MaxPooling2D((2, 2)) (c3)

		fc = Conv2D(512, (3, 3), kernel_initializer='Orthogonal', padding='same') (p3)#16x16x512
		
		u4 = UpSampling2D(size=(2,2))(fc)
		c4 = Conv2DTranspose(128, (3, 3), activation='relu', kernel_initializer='Orthogonal', padding='same') (u4)

		u5 = UpSampling2D(size=(2,2))(c4)
		c5 = Conv2DTranspose(64, (3, 3), activation='relu', kernel_initializer='Orthogonal', padding='same') (u5)

		u6 = UpSampling2D(size=(2,2))(c5)
		c6 = Conv2DTranspose(32, (3, 3), activation='relu', kernel_initializer='Orthogonal', padding='same') (u6)

		self.outputs = Conv2D(IMG_CLASSES, (3, 3), activation='relu', kernel_initializer='Orthogonal', padding='same') (c6)