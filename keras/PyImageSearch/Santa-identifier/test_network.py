from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained model")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

image=cv2.imread(args["image"])
orig=image.copy()

image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
cv2.imshow("im", image)
image = img_to_array(image)
image = np.expand_dims(image, axis = 0) #(1, w, h , 3) assuming channels last ordering

print("[INFO] loading network...")
model = load_model(args["model"])

(notSanta, santa) = model.predict(image)[0]

#labels
label = "Santa" if santa > notSanta else "Not Santa"
proba = santa if santa > notSanta else notSanta
label = "{}: {:.2f}%".format(label, proba*100)

output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Output", output)
cv2.waitKey(0)

