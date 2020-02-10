# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 21:21:55 2020

@author: Tobias
"""

import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
if (not tf.executing_eagerly()): 
    tf.compat.v1.enable_eager_execution(
        config=None,
        device_policy=None,
        execution_mode=None
    )
tf.executing_eagerly()
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
#from tensorflow.keras.callbacks import
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import os
from glob import glob
# LOAD train_x, train_y

#TRAINING DATA
pickle_in = open("C:/Users/Tobias/CNN/training_set_X.pickle", "rb")
train_x = pickle.load(pickle_in)
#Answers to training data
pickle_in = open("C:/Users/Tobias/CNN/training_set_y.pickle", "rb")
train_y = pickle.load(pickle_in)

#TEST DATA
pickle_in = open("C:/Users/Tobias/CNN/test_set_X.pickle", "rb")
test_x = pickle.load(pickle_in)
#Answers to training data
pickle_in = open("C:/Users/Tobias/CNN/test_set_y.pickle", "rb")
test_y = pickle.load(pickle_in)

#Normalize data -- scale the data
train_x = train_x/255.0
test_x = test_x/255.0
print(train_x.shape)
print(test_x.shape)

def CNN_Model():
    model = Sequential()
    #layer_1
    model.add(Conv2D(64, kernel_size=5, strides=1, input_shape = (128,128,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding="valid"))
    #Layer_2
    model.add(Conv2D(64, kernel_size=3, strides=1,))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding="valid"))
    #layer_3
    #model.add(Conv2D(64, kernel_size=3, strides=1))
    #model.add(BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001))
    #model.add(Activation("relu"))
    #layer_4
    model.add(Flatten())
    model.add(Dense(64))
    model.add(BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001))
    #Output_layer
    model.add(Dropout(0.6))
    model.add(Dense(5))
    model.add(BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001))
    model.add(Activation('softmax'))
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model

three_layer_cnn = CNN_Model()
#train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
train_y = np.array(train_y)
test_y = np.array(test_y)
print(train_x.shape, train_x.dtype)
print(train_y.shape, train_y.dtype)
print(test_x.shape, train_x.dtype)
print(test_y.shape, train_y.dtype)

#only save the best model based on what is being monitored
checkpoint_loss = ModelCheckpoint("C:/Users/Tobias/CNN/PICKLE{epoch:02d}-{loss:.2f}_best_model.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
checkpoint_val_loss = ModelCheckpoint("C:/Users/Tobias/CNN/PICKLE{epoch:02d}-{val_loss:.2f}_best_val_model.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.001)

history = three_layer_cnn.fit(train_x, train_y, batch_size=16, validation_split=0.15, epochs = 2, callbacks=[checkpoint_loss, checkpoint_val_loss, reduce_lr])


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
print(history.history['accuracy'])
model.evaluate(test_x, test_y, batch_size=16, verbose=2)

three_layer_cnn.load_weights('C:/Users/Tobias/CNN/02-10_PICKLE18-0.10_best_model.h5')
three_layer_cnn.evaluate(test_x, test_y, batch_size=16, verbose=2)
print(three_layer_cnn.metrics_names)
#PREDICT A SINGLE IMAGE
img = cv2.imread('C:/Users/Tobias/CNN/Images/Trafik/trafik_tva_1.jpg', cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dim = (128,128)

img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)

img_resized = np.expand_dims(img_resized, axis=0)
img_resized = tf.cast(img_resized, tf.float32)
img_resized = img_resized/255
y_prob = three_layer_cnn.predict(img_resized)

print(y_prob)

#model.save_weights('C:/Users/Tobias/CNN/PICKLE_best_E20_model.h5')
# PRINT AND EVALUATE THE ERROR RATE OF THE TRAINING SET
three_layer_cnn.load_weights('C:/Users/Tobias/CNN/02-10_PICKLE18-0.10_best_model.h5')
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
    dim = (128,128)
    
    img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
    
    img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = tf.cast(img_resized, tf.float32)
    img_resized = img_resized/255
    y_prob = model.predict(img_resized)
#    class1 = np.array([[1.,0.,0.,0.,0.]])
#    class2 = np.array([[0.,1.,0.,0.,0.]])
#    class3 = np.array([[0.,0.,1.,0.,0.]])
#    class4 = np.array([[0.,0.,0.,1.,0.]])
#    class5 = np.array([[0.,0.,0.,0.,1.]])
    if (target == "class1"):
        if ((y_prob.item(1) and y_prob.item(2) and y_prob.item(3) and y_prob.item(4)) < y_prob.item(0)):
            incrementTot()
            print("Class1 " + str(y_prob))
        else:
            print("ERRORc1 - " + str(y_prob) + " - wrong prediction")
            incrementTot()
            incrementError()
    elif (target == "class2"):
        if((y_prob.item(0) and y_prob.item(2) and y_prob.item(3) and y_prob.item(4)) < y_prob.item(1)):
            incrementTot()
            print("Class2 " + str(y_prob))
        else:
            print("ERRORc2 - " + str(y_prob) + " - wrong prediction")
            incrementTot()
            incrementError()
    elif (target == "class3"):
        if((y_prob.item(0) and y_prob.item(1) and y_prob.item(3) and y_prob.item(4)) < y_prob.item(2)):
            incrementTot()
            print("Class3 " + str(y_prob))
        else:
            print("ERRORc3 - " + str(y_prob) + " - wrong prediction")
            incrementTot()
            incrementError()
    elif (target == "class4"):
        if((y_prob.item(0) and y_prob.item(1) and y_prob.item(2) and y_prob.item(4)) < y_prob.item(3)):
            incrementTot()
            print("Class4 " + str(y_prob))
        else:
            print("ERRORc4 - " + str(y_prob) + " - wrong prediction")
            incrementTot()
            incrementError()
    elif (target == "class5"):
        if((y_prob.item(0) and y_prob.item(1) and y_prob.item(2) and y_prob.item(3)) < y_prob.item(4)):
            incrementTot()
            print("Class5 " + str(y_prob))
        else:
            print("ERRORc5 - " + str(y_prob) + " - wrong prediction")
            incrementTot()
            incrementError()  
    else:
        print("elif did not work: T-> " + target)

def printResultsOnClass1(model):
    Data_dir=np.array(glob('C:/Users/Tobias/CNN/Images/Hallplats/*'))
    i = 0
    for images in Data_dir[i:i+2000]:
        i+=1
        predict_image(model, images, "class1")
    errorRate = getError()/getTot()
    print ("Prediction error: " + str(errorRate))

def printResultsOnClass2(model):
    Data_dir=np.array(glob('C:/Users/Tobias/CNN/Images/Johan/*'))
    i = 0
    for images in Data_dir[i:i+2000]:
        i+=1
        predict_image(model, images, "class2")
    errorRate = getError()/getTot()
    print ("Prediction error: " + str(errorRate))
    
def printResultsOnClass3(model):
    Data_dir=np.array(glob('C:/Users/Tobias/CNN/Images/Rulltrappa/*'))
    i = 0
    for images in Data_dir[i:i+2000]:
        i+=1
        predict_image(model, images, "class3")
    errorRate = getError()/getTot()
    print ("Prediction error: " + str(errorRate))
    
def printResultsOnClass4(model):
    Data_dir=np.array(glob('C:/Users/Tobias/CNN/Images/Tobias/*'))
    i = 0
    for images in Data_dir[i:i+2000]:
        i+=1
        predict_image(model, images, "class4")
    errorRate = getError()/getTot()
    print ("Prediction error: " + str(errorRate))
    
def printResultsOnClass5(model):
    Data_dir=np.array(glob('C:/Users/Tobias/CNN/Images/Trafik/*'))
    i = 0
    for images in Data_dir[i:i+2000]:
        i+=1
        predict_image(model, images, "class5")
    errorRate = getError()/getTot()
    print ("Prediction error: " + str(errorRate))

resetErrorRate()
print("Class1")
printResultsOnClass1(three_layer_cnn)
print("Class2")
printResultsOnClass2(three_layer_cnn)
print("Class3")
printResultsOnClass3(three_layer_cnn)
print("Class4")
printResultsOnClass4(three_layer_cnn)
print("Class5")
printResultsOnClass5(three_layer_cnn)

##################################
###     HEAT MAPS
##################################
from tensorflow.keras import backend as K
from tensorflow.keras import models
from tensorflow import reshape
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
image_size = (128, 128)

base_model = three_layer_cnn
base_model.load_weights('C:/Users/Tobias/CNN/02-10_PICKLE18-0.10_best_model.h5')
#print(model.layers[-3])
#print(model.layers[-3].get_weights()[0])
print (three_layer_cnn.layers[-13])
weights = three_layer_cnn.layers[-3].get_weights()[0]
print(base_model.inputs)
print(three_layer_cnn.input)
conv_layer = three_layer_cnn.layers[-10]
model2 = Model(three_layer_cnn.input, [conv_layer.output, three_layer_cnn.output])
#plt.figure(figsize=(12, 10))
Data_dir=np.array(glob('C:/Users/Tobias/CNN/Images/Hallplats/*'))
for image in Data_dir[:10]:
    image_size = 128
    
    # Load pre-trained Keras model and the image to classify
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dim = (128,128)
    img_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_tensor = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
    
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor = img_tensor/255
    
    #image = np.random.random((image_size, image_size, 3))
    #img_tensor = preprocessing.image.img_to_array(image)
    #img_tensor = np.expand_dims(img_tensor, axis=0)
    #img_tensor = preprocess_input(img_tensor)
    
    conv_layer = three_layer_cnn.layers[-10]
    heatmap_model = models.Model([three_layer_cnn.inputs], [conv_layer.output, three_layer_cnn.output])
    print(heatmap_model)
    # Get gradient of the winner class w.r.t. the output of the (last) conv. layer
    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(img_tensor)
        loss = predictions[:, np.argmax(predictions[0])]
        grads = gtape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
    
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
   # heatmap = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = reshape(heatmap, (60,60), 3)
    heatmap = np.maximum(heatmap, 0)

    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    
    heatmap_img = cv2.applyColorMap(img_orig, cv2.COLORMAP_JET)
    plt.matshow(heatmap_img)
    plt.show()
    
    
######
# HEATMAP V2
####
from tensorflow.keras import backend as K
from tensorflow.keras import models
from tensorflow import reshape
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
image_size = (128, 128)

base_model = three_layer_cnn
base_model.load_weights('C:/Users/Tobias/CNN/02-10_PICKLE18-0.10_best_model.h5')
#print(model.layers[-3])
#print(model.layers[-3].get_weights()[0])
print (three_layer_cnn.layers[-13])
weights = three_layer_cnn.layers[-3].get_weights()[0]
print(base_model.inputs)
print(three_layer_cnn.input)




conv_layer = three_layer_cnn.layers[-10]
model2 = Model(three_layer_cnn.input, [conv_layer.output, three_layer_cnn.output])
#plt.figure(figsize=(12, 10))

weights = three_layer_cnn.layers[-3].get_weights()[0]
Data_dir=np.array(glob('C:/Users/Tobias/CNN/Images/Johan/*'))
for image in Data_dir[:1]:
    
    
    # Load pre-trained Keras model and the image to classify
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dim = (128,128)
    
    img_tensor = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
    
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor = img_tensor/255
    
    [base_model_outputs, prediction] = model2.predict(img_tensor)
    prediction = prediction[0]
    print(prediction)
    base_model_outputs = base_model_outputs[0]
    print(base_model_outputs)
    print(np.matmul(base_model_outputs, weights))
    cam = (prediction - 0.5) * np.matmul(base_model_outputs, weights)

    cam -= cam.min()
    cam /= cam.max()
    cam -= 0.2
    cam /= 0.8
    
    cam = cv2.resize(cam, (128, 128))
    #heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam <= 0.2)] = 0
    
    out = cv2.addWeighted(img, 0.8, heatmap, 0.4, 0)
    
    plt.axis('off')
    plt.imshow(out[:,:,::-1])
