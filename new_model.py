import os,cv2,keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from program import getCoordinatesDictList, getSortedFilenames, getModifiedImagePath, getModifiedAnnotationsPath
import random

def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

train_images = []
train_labels = []

count = 0
for filename in getSortedFilenames()[:200]:
    try:
        count += 1
        print('-----------------------------')
        print(count, filename)
        image = cv2.imread(getModifiedImagePath(filename))
        coordinatesDictList = getCoordinatesDictList(filename)
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        for gtval in coordinatesDictList:
            print('Looping through coordinate', gtval)
            imout = image.copy()
            timage = imout[gtval['y1']: gtval['y2'], gtval['x1'] : gtval['x2']]
            if timage.shape[0] != 0 and timage.shape[1] != 0:
                resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                train_images.append(resized)
                train_labels.append(1)

        imout = image.copy()
        counter = 0
        falsecounter = 0
        ious = []

        # random.shuffle(ssresults)
        for e,result in enumerate(ssresults):
            max_loop_iou = 0
            if e < 2000:
                x, y, w, h = result
                for gtval in coordinatesDictList:
                    iou = get_iou(gtval,{ "x1": x,"x2": x + w, "y1": y,"y2": y + h })
                    ious.append(iou)
                    max_loop_iou = max(max_loop_iou, iou)

                if max_loop_iou > 0.25:
                    print({ 'x1': x, 'x2': x + w, 'y1': y, 'y2': y + h }, 'is labelled!')
                    timage = imout[y: y + h, x: x + w]
                    resized = cv2.resize(timage, (224, 224), interpolation = cv2.INTER_AREA)
                    train_images.append(resized)
                    train_labels.append(1)
                    counter += 1
                elif max_loop_iou < 0.05 and falsecounter < 20:
                    timage = imout[y: y + h, x: x + w]
                    resized = cv2.resize(timage, (224, 224), interpolation = cv2.INTER_AREA)
                    train_images.append(resized)
                    train_labels.append(0)
                    falsecounter += 1

        print('Max IOU:', max(ious))

    except Exception as e:
        print("Error occured in ", filename, ':', e)
        continue

X_new = np.array(train_images)
y_new = np.array(train_labels)

X_new.shape

from keras.layers import Dense
from keras import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

vggmodel = VGG16(weights='imagenet', include_top=True)
vggmodel.summary()

for layers in (vggmodel.layers)[:15]:
    print(layers)
    layers.trainable = False

X= vggmodel.layers[-2].output

predictions = Dense(2, activation="softmax")(X)

model_final = Model(inputs = vggmodel.input, outputs = predictions)

from keras.optimizers import Adam
opt = Adam(lr=0.0001)


model_final.compile(loss = keras.losses.categorical_crossentropy, optimizer = opt, metrics=["accuracy"])

model_final.summary()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1-Y))
        else:
            return Y
    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)

lenc = MyLabelBinarizer()
Y =  lenc.fit_transform(y_new)

X_train, X_test , y_train, y_test = train_test_split(X_new,Y,test_size=0.10)


print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


trdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
traindata = trdata.flow(x=X_train, y=y_train)
tsdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
testdata = tsdata.flow(x=X_test, y=y_test)

from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("ieeercnn_vgg16_1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

hist = model_final.fit_generator(generator= traindata, steps_per_epoch= 10, epochs= 10, validation_data= testdata, validation_steps=2, callbacks=[checkpoint,early])


import matplotlib.pyplot as plt
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

im = X_test[-1]
plt.imshow(im)
img = np.expand_dims(im, axis=0)
out = model_final.predict(img)
if out[0][0] > out[0][1]:
    print("Formula region found")
else:
    print("Formula region not found")

testFilenames = getSortedFilenames()[600:]
testFilenames = random.sample(testFilenames, 10)

for filename in testFilenames:

    print('---------------------------------------------')
    print('Predicting file:', filename)

    img = cv2.imread(getModifiedImagePath(filename))
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()
    imout = img.copy()
    for e, result in enumerate(ssresults):
        if e < 2000:
            x,y,w,h = result
            timage = imout[y: y + h, x: x + w]
            resized = cv2.resize(timage, (224, 224), interpolation = cv2.INTER_AREA)
            img = np.expand_dims(resized, axis = 0)
            out = model_final.predict(img)
            print(filename, e, out[0][0])
            if out[0][0] > 0.3:
                print('Obtained rectangular area:', out[0][0])
                cv2.rectangle(imout, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
    plt.clf()
    plt.figure()
    plt.imshow(imout)
    plt.savefig(filename + '.jpg')
