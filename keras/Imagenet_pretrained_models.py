#source: https://www.learnopencv.com/keras-tutorial-using-pre-trained-imagenet-models/

import keras
import numpy as np
from keras.applications import vgg16, inception_v3, resnet50, mobilenet

vgg_model = vgg16.VGG16(weights='imagenet')
inception_model = inception_v3.InceptionV3(weights='imagenet')
resnet_model = resnet50.ResNet50(weights='imagenet')
mobilenet_model = mobilenet.MobileNet(weights='imagenet')

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt

filename = 'images/cat.jpg'
#load in PIL format
original = load_img(filename, target_size=(224, 224))
print('PIL image size', original.size)

numpy_image = img_to_array(original)
print('numpy array size', numpy_image.shape)

image_batch = np.expand_dims(numpy_image, axis=0)
print('image batch size', image_batch.shape)

#vgg
processed_image = vgg16.preprocess_input(image_batch.copy())
predictions = vgg_model.predict(processed_image)
label = decode_predictions(predictions)
print("vgg",label)

#resnet
processed_image = resnet50.preprocess_input(image_batch.copy())
predictions = resnet_model.predict(processed_image)
label_resnet = decode_predictions(predictions, top=3)
print("resnet", label_resnet)

#mobileNet
processed_image = mobilenet.preprocess_input(image_batch.copy())
predictions = mobilenet_model.predict(processed_image)
label_mobilenet = decode_predictions(predictions)
print("mobilenet", label_mobilenet)

#inception_v3
#Has different input size
original = load_img(filename, target_size=(299, 299))
numpy_image = img_to_array(original)
image_batch = np.expand_dims(numpy_image, axis=0)
processed_image = inception_v3.preprocess_input(image_batch.copy())
predictions = inception_model.predict(processed_image)

label_inception = decode_predictions(predictions)
print("inception", label_inception)