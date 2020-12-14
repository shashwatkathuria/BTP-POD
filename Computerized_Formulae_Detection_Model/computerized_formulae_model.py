# B. Tech Project
# HANDWRITTEN FORMULAE DETECTION
# Semester 7
# December 2020
# Dr. Gaurav Harit
# Shashwat Kathuria - B17CS050
# Satya Prakash Sharma - B17CS048

# Importing required libraries
import os, cv2, keras, random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from computerized_model_helper import getCoordinatesDictList, getSortedFilenames, getModifiedImagePath, getModifiedAnnotationsPath
from keras.layers import Dense
from keras import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint, EarlyStopping

def getIOU(boundingBox1, boundingBox2):
    '''Function to return calculated IOU value of the two bounding boxes given as input.'''

    # Asserting correct coordinates
    assert boundingBox1['x1'] < boundingBox1['x2']
    assert boundingBox1['y1'] < boundingBox1['y2']
    assert boundingBox2['x1'] < boundingBox2['x2']
    assert boundingBox2['y1'] < boundingBox2['y2']

    # Storing coordinates
    xLeft = max(boundingBox1['x1'], boundingBox2['x1'])
    yTop = max(boundingBox1['y1'], boundingBox2['y1'])
    xRight = min(boundingBox1['x2'], boundingBox2['x2'])
    yBottom = min(boundingBox1['y2'], boundingBox2['y2'])

    # Return zero if no intersection
    if xRight < xLeft or yBottom < yTop:
        return 0.0

    # Calculating intersection area
    intersectionArea = (xRight - xLeft) * (yBottom - yTop)

    # Calculating areas of the bounding boxes separately
    boundingBox1Area = (boundingBox1['x2'] - boundingBox1['x1']) * (boundingBox1['y2'] - boundingBox1['y1'])
    boundingBox2Area = (boundingBox2['x2'] - boundingBox2['x1']) * (boundingBox2['y2'] - boundingBox2['y1'])

    # Calculating IOU
    iou = intersectionArea / float(boundingBox1Area + boundingBox2Area - intersectionArea)

    # Asserting IOU >= 0 and <= 1
    assert iou >= 0.0
    assert iou <= 1.0

    # Returning IOU value
    return iou

# Selective search
selectiveSearch = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# Initializing training variables as required
trainImages = []
trainLabels = []

# Keeping track of count
count = 0
# Looping through all the filenames of annnotations and images
for filename in getSortedFilenames():
    try:
        # Incrementing counter
        count += 1
        print('-----------------------------')
        print(count, filename)

        # Initializing cv2 image of input image file
        image = cv2.imread(getModifiedImagePath(filename))

        # Getting coordinates list of the input annotation file
        coordinatesDictList = getCoordinatesDictList(filename)

        # Setting base image for selective search and setting to fast mode
        selectiveSearch.setBaseImage(image)
        selectiveSearch.switchToSelectiveSearchFast()

        # Getting region proposals
        regionProposals = selectiveSearch.process()

        # Looping through all the formula coordinates and labelling them
        for coordinateDict in coordinatesDictList:
            print('Looping through coordinate', coordinateDict)
            # Copying image
            imout = image.copy()
            # Getting the subset of image with bounding box of the coordinates mentioned
            timage = imout[coordinateDict['y1'] : coordinateDict['y2'], coordinateDict['x1'] : coordinateDict['x2']]

            if timage.shape[0] != 0 and timage.shape[1] != 0:
                # Resizing image
                resized = cv2.resize(timage, (224, 224), interpolation = cv2.INTER_AREA)
                # Adding to training labels and images
                trainImages.append(resized)
                trainLabels.append(1)

        # Initializing variables required
        imout = image.copy()
        counter = 0
        falseCounter = 0
        ious = []

        # Looping through all the region proposals
        for e, result in enumerate(regionProposals):
            maxLoopIOU = 0
            # Analyzing a max of 2000 regions
            if e < 2000:
                # Getting the IOU of the region proposed with all the annotated formula coordinates
                x, y, w, h = result
                for coordinateDict in coordinatesDictList:
                    iou = getIOU(coordinateDict,{ "x1": x,"x2": x + w, "y1": y,"y2": y + h })
                    ious.append(iou)
                    maxLoopIOU = max(maxLoopIOU, iou)

                # Adding to training images and labels as a formula region if IOU > 0.25
                if maxLoopIOU > 0.25:
                    print({ 'x1': x, 'x2': x + w, 'y1': y, 'y2': y + h }, 'proposed region is added to training labels and images!')
                    # Getting the subset of image with bounding box of the coordinates mentioned
                    timage = imout[y: y + h, x: x + w]
                    # Resizing image
                    resized = cv2.resize(timage, (224, 224), interpolation = cv2.INTER_AREA)
                    # Adding to training labels and images
                    trainImages.append(resized)
                    trainLabels.append(1)
                    # Incrementing counter
                    counter += 1
                # Adding to training images and labels as a non-formula region if IOU < 0.05
                # With a max of 20 such inputs per input image
                elif maxLoopIOU < 0.05 and falseCounter < 20:
                    # Getting the subset of image with bounding box of the coordinates mentioned
                    timage = imout[y: y + h, x: x + w]
                    # Resizing image
                    resized = cv2.resize(timage, (224, 224), interpolation = cv2.INTER_AREA)
                    # Adding to training labels and images
                    trainImages.append(resized)
                    trainLabels.append(0)
                    # Incrementing false counter
                    falseCounter += 1

        # Printing max IOU in the loop for the input image
        print('Max IOU:', max(ious))

    # Continuing if error in file
    except Exception as e:
        print("Error occured in ", filename, ':', e)
        continue

# Getting numpy array of training images and labels
X_train = np.array(trainImages)
Y_train = np.array(trainLabels)
X_train.shape

# Initializing VGG 16 Model
vggmodel = VGG16(weights = 'imagenet', include_top = True)
vggmodel.summary()

# Printing model summary
for layers in (vggmodel.layers)[:15]:
    print(layers)
    layers.trainable = False

# Removing last 2 layer of vgg model and using softmax in their place
X = vggmodel.layers[-2].output
predictions = Dense(2, activation = "softmax")(X)

# Final model
finalModel = Model(inputs = vggmodel.input, outputs = predictions)

# Using adam optimizer
adamOptimizer = Adam(lr = 0.0001)

# Compiling model with loss function as categorical crossentropy and adam optimizer
finalModel.compile(loss = keras.losses.categorical_crossentropy, optimizer = adamOptimizer, metrics = ["accuracy"])
finalModel.summary()

# Class for one hot encoding
class OneHotEncoder(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        # Binary one hot encoding
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1 - Y))
        else:
            return Y
    def inverse_transform(self, Y, threshold = None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)

# One hot encoding
lenc = OneHotEncoder()
Y = lenc.fit_transform(Y_train)

# Splitting data
X_train, X_test , Y_train, Y_test = train_test_split(X_train, Y, test_size = 0.10)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# Flipping the image to generate more images in the model to make prediction better
trdata = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, rotation_range = 90)
traindata = trdata.flow(x = X_train, y = Y_train)
tsdata = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, rotation_range = 90)
testdata = tsdata.flow(x = X_test, y = Y_test)

# Setting model checkpoint, early stopping and fit generator
checkpoint = ModelCheckpoint("ieeercnn_vgg16_1.h5", monitor = 'val_loss', verbose = 1, save_best_only = True, save_weights_only = False, mode = 'auto', period = 1)
early = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 100, verbose = 1, mode = 'auto')
hist = finalModel.fit_generator(generator = traindata, steps_per_epoch = 10, epochs = 100, validation_data = testdata, validation_steps = 2, callbacks = [checkpoint, early])

# Plotting the model loss v/s epoch
plt.clf()
# plt.plot(hist.history["acc"])
# plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Loss","Validation Loss"])
plt.show()
plt.savefig('chart loss.png')

# Predicting 15 images randomly
testFilenames = getSortedFilenames()
testFilenames = random.sample(testFilenames, 15)

# Looping through images
for filename in testFilenames:

    print('---------------------------------------------')
    print('Predicting file:', filename)

    # Initializing cv2 image of input image file
    img = cv2.imread(getModifiedImagePath(filename))

    # Setting base image for selective search and setting to fast mode
    selectiveSearch.setBaseImage(img)
    selectiveSearch.switchToSelectiveSearchFast()

    # Getting region proposals
    regionProposals = selectiveSearch.process()

    imout = img.copy()
    # Looping through all the region proposals
    for e, result in enumerate(regionProposals):
        # Analyzing a max of 2000 regions
        if e < 2000:
            x, y, w, h = result
            # Getting the subset of the image bounding box region proposal
            timage = imout[y: y + h, x: x + w]
            # Resizing the image
            resized = cv2.resize(timage, (224, 224), interpolation = cv2.INTER_AREA)
            # Predicing the region, whether or not it is a formula
            img = np.expand_dims(resized, axis = 0)
            out = finalModel.predict(img)
            print(filename, e, out[0][0])
            # If prediction value > 0.3, then it is a formula region
            if out[0][0] > 0.3:
                # Marking region bounding box in image
                print('Obtained rectangular area:', out[0][0])
                cv2.rectangle(imout, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

    # Saving image with the predictions marked
    plt.clf()
    plt.figure()
    plt.imshow(imout)
    plt.savefig(filename + '.jpg')
