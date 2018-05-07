##Source https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/
##Data is gathered using Bing Image search API
###https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/

## VGGNet - very uniform, lots of filters. Preferred choice of the comunitty for extracting features.
import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from silidragos.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output")
ap.add_argument("-l", "--labelbin", required=True, help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="output accuracy/loss plot")
args = vars(ap.parse_args())

EPOCHS = 75
INIT_LR = 1e-3 #default for Adam optymizer
BS = 32
IMAGE_DIMS = (96, 96, 3)

print("[Info] Loading and shuffling images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

data =[]
labels = []

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

    l = label = imagePath.split(os.path.sep)[-2].split("_")
    labels.append(l)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

print("[Info] data matrix: {} images ({:.2f}MB".format(len(imagePaths), data.nbytes / (1024 * 1000.0)))

#binarizing the labels
print("[Info] class labels: ")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

for (i, label) in enumerate(mlb.classes_):
    print("{}.{}".format(i+1, label))


#train/test split
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state = 42)
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, 
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")


#Build model and initialize Adam Optimizer
print("[Info] compiling model...")
model = SmallerVGGNet.build(width = IMAGE_DIMS[1], height=IMAGE_DIMS[0],
    depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
    finalAct="sigmoid")

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

#Compile and train!!
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[Info] training network...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)

print("[Info] Serializing network...")
model.save(args["model"])

print("[Info] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(mlb))
f.close()

#Plot
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper_left")
plt.savefig(args["plot"])








