# USAGE
# py -3.5 train_cnn.py
# needs to be v3.5 because of TensorFlow

# import the necessary packages
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import time
import os
import cv2




def save_model(model,labels,filename):
  # serialize model to JSON
  model_json = model.to_json()
  with open(filename+".json", "w") as json_file:
      json_file.write(model_json)
  with open(filename+".labels", "w") as text_file:
      text_file.write("\n".join(labels))
  # serialize weights to HDF5
  model.save_weights(filename+".h5")
  print("Saved model to disk")

def load_model(filename):
  # load json and create model
  json_file = open(filename+'.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()

  text_file = open(filename+'.labels', 'r')
  loaded_labels = text_file.read().split("\n")
  text_file.close()

  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights(filename+".h5")
  print("Loaded model from disk")
  return (loaded_model,loaded_labels)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="shapes",
  help="path to directory containing your dataset")
ap.add_argument("-l", "--loadmodel", type=str, default="model",
  help="name the file of a saved model to load")
ap.add_argument("-s", "--savemodel", type=str, default="model",
  help="name the file to save the model as")
ap.add_argument("-t", "--train", type=str, default="50",
  help="number of times to train")
args = vars(ap.parse_args())

# grab all image paths in the input dataset directory, then initialize
# our list of images and corresponding class labels
print("[INFO] loading images...")
imagePaths = paths.list_images(args["dataset"])
data = []
labels = []


t0 = time.time()



# loop over our input images
for imagePath in imagePaths:
  # load the input image from disk, resize it to 32x32 pixels, scale
  # the pixel intensities to the range [0, 1], and then update our
  # images list

  image = Image.open(imagePath)
  print(imagePath,np.array(image).shape)

  image = ( np.array(image.resize((80,80))) / 255.0 )[:,:,:3]
  data.append(image)

  # extract the class label from the file path and update the
  # labels list
  label = imagePath.split(os.path.sep)[-2]
  labels.append(label)

print(labels)

# encode the labels, converting them from strings to integers
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
class_labels = lb.classes_


# perform a training and testing split, using 75% of the data for
# training and 25% for evaluation
(trainX, testX, trainY, testY) = train_test_split(np.array(data),
  np.array(labels), test_size=.25)



if ( args["loadmodel"] ):
  print("Loading previous model");

  (model,class_labels) = load_model( args["loadmodel"] )

else:
  print("Generating new model");

  # define our Convolutional Neural Network architecture
  model = Sequential()

  model.add(Conv2D(6, (5, 5), padding="same", input_shape=(80,80, 3)))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

  model.add(Conv2D(12, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

  model.add(Conv2D(6, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

  model.add(Flatten())
  model.add(Dense(5))
  model.add(Activation("softmax"))


if int(args["train"])>0:
  # train the model using the Adam optimizer
  print("[INFO] training network...")
  opt = Adam(lr=1e-3, decay=1e-3 / 50)

  model.compile(loss="categorical_crossentropy", optimizer=opt,
      metrics=["accuracy"])
  H = model.fit(trainX, trainY, validation_data=(testX, testY),
      epochs=int(args["train"]), batch_size=10000)


t1 = time.time()
print ("%.4f secs." % (t1-t0) )


if ( args["savemodel"] ):
  print("Saving trained model");
  save_model( model, class_labels, args["savemodel"] )


# show layers of model
print("\n")
wts = model.get_weights()
for layer in wts:
  print ("Weights",layer.shape)
print("\n")


# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=1000)
print(classification_report(testY.argmax(axis=1),
  predictions.argmax(axis=1), target_names=class_labels))


Y = testY.argmax(axis=1)
P = predictions.argmax(axis=1)

for i in range( Y.size ):

  correct = ' ' if ( P[i] == Y[i] ) else 'X'
  print ("%s   %s => %s" % (correct,class_labels[Y[i]],class_labels[P[i]]))

  # reorder the channels
  image = testX[i][...,::-1]
  image = np.uint8( image*255 )

  image = cv2.resize( image, (400,400) )
  label = class_labels[Y[i]]

  cv2.imshow("Out",image)
  if cv2.waitKey(0) == 27: 
    break  # esc to quit
