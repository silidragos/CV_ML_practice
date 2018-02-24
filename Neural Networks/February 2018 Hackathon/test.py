from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.utils import np_utils
import numpy
import os
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

json_file=open("model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('model.h5')
print('loaded model from disk')

loaded_model.compile(loss='categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy'])
score = loaded_model.evaluate(x_test, y_test, verbose = 0)
print('Test loss: ' , score[0])
print('Test accuracy: ' , score[1])