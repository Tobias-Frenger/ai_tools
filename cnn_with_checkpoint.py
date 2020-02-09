# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 21:21:55 2020

@author: Tobias
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import os
from glob import glob
# LOAD train_x, train_y
#Training data
pickle_in = open("C:/path-to-training-set/training_set_X.pickle", "rb")
X = pickle.load(pickle_in)

#Answers to training data
pickle_in = open("C:/path-to-training-set-facit/training_set_y.pickle", "rb")
y = pickle.load(pickle_in)

#Normalize data -- scale the data
X = X/255.0
print(X.shape)

model = Sequential()
#layer_1
model.add(Conv2D(64, kernel_size=5, strides=1, input_shape = (128,128,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding="valid"))
#Layer_2
model.add(Conv2D(128, kernel_size=3, strides=1,))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding="valid"))
#layer_3
model.add(Conv2D(64, kernel_size=3, strides=1))
model.add(BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001))
model.add(Activation("relu"))
#layer_4
model.add(Flatten())
model.add(Dense(64))
model.add(BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001))
#Output_layer
model.add(Dropout(0.6))
model.add(Dense(3))
model.add(BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001))
model.add(Activation('softmax'))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

#train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
y = np.array(y)
print(X.shape, X.dtype)
print(y.shape, y.dtype)

#only save the best model based on what is being monitored
checkpoint = ModelCheckpoint("C:/where-to-save-model-weights/best_model.h5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)

history = model.fit(X, y, batch_size=16, validation_split=0.15, epochs = 75, callbacks=[checkpoint])

def show(history):
    val_acc = history['val_accuracy']
    val_loss = history['val_loss']
    acc = history['accuracy']
    loss = history['loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, loss, 'c', label='Training loss')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.legend()
    plt.figure()
    plt.show()

show(history.history)

#PREDICT A SINGLE IMAGE
img = cv2.imread('C:/image-dir/img.jpg', cv2.IMREAD_UNCHANGED)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

dim = (128,128)

img_resized = cv2.resize(img_rgb, dim, interpolation = cv2.INTER_CUBIC)

img_resized = np.expand_dims(img_resized, axis=0)
img_resized = tf.cast(img_resized, tf.float32)
y_prob = model.predict(img_resized)

print(y_prob)


# PRINT AND EVALUATE THE ERROR RATE OF THE TRAINING SET
tot = 0
error = 0
def incrementTot():
    global tot
    tot += 1
    
def incrementError():
    global error
    error += 1
    
def getTot():
    global tot
    return tot

def getError():
    global error
    return error

def resetErrorRate():
    global tot
    global error
    tot = 0
    error = 0

def predict_image(model, image, target):

    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = np.squeeze(img, axis=None)
    dim = (128,128)
    #img = cv2.resize(img,dim, interpolation = cv2.INTER_CUBIC)
    
    img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
    img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = tf.cast(img_resized, tf.float32)

    y_prob = model.predict(img_resized)
    class1 = np.array([[1.,0.,0.]])
    Class2 = np.array([[0.,1.,0.]])
    Class3 = np.array([[0.,0.,1.]])
    
    if (np.array_equal(y_prob, knight) or (y_prob.item(0) > y_prob.item(1) and y_prob.item(2))):
        if (target == "class1"):
            incrementTot()
            print("Class1 " + str(y_prob))
        else:
            print("ERROR - " + str(y_prob) + " - wrong prediction")
            incrementTot()
            incrementError()
    elif(np.array_equal(y_prob, orc) or (y_prob.item(0) and y_prob.item(2) < y_prob.item(1))):
        if (target == "class2"):
            incrementTot()
            print("Class2 " + str(y_prob))
        else:
            print("ERROR - " + str(y_prob) + " - wrong prediction")
            incrementTot()
            incrementError()
    elif(np.array_equal(y_prob, sorceress) or (y_prob.item(1) and y_prob.item(0) < y_prob.item(2))):
        if (target == "class3"):
            incrementTot()
            print("Class3 " + str(y_prob))
        else:
            print("ERROR - " + str(y_prob) + " - wrong prediction")
            incrementTot()
            incrementError()            
    else:
        print("elif did not work")

def printResultsOnClass1(model):
    Data_dir=np.array(glob('C:/path-to-first-class/class1/*'))
    i = 0
    for images in Data_dir[i:i+2000]:
        i+=1
        predict_image(model, images, "class1")
    errorRate = getError()/getTot()
    print ("Prediction error: " + str(errorRate))

def printResultsOnClass2(model):
    Data_dir=np.array(glob('C:/path-to-second-class/class2/*'))
    i = 0
    for images in Data_dir[i:i+2000]:
        i+=1
        predict_image(model, images, "class2")
    errorRate = getError()/getTot()
    print ("Prediction error: " + str(errorRate))
    
def printResultsOnClass3(model):
    Data_dir=np.array(glob('C:/path-to-third-class/class3/*'))
    i = 0
    for images in Data_dir[i:i+2000]:
        i+=1
        predict_image(model, images, "class3")
    errorRate = getError()/getTot()
    print ("Prediction error: " + str(errorRate))

resetErrorRate()
print("Class1")
printResultsOnClass1(model)
print("Class2")
printResultsOnClass2(model)
print("Class3")
printResultsOnClass3(model)
