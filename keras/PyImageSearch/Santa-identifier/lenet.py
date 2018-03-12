from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as k

class LeNet:
    #width, height, # of channels, classes to be recognized
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        #channels last -> TF
        inputShape = (height, width, depth)

        #theano
        if k.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        #CONV => RELU => POOL
        #20 filters of 5x5
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        #stride ~ step
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #CONV->RELU->POOL
        #it's common to see the layers increase the deeper we go
        model.add(Conv2D(50, (5,5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #Now we start flattening
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        #softmax
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
