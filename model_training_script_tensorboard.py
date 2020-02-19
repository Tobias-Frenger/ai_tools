# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 00:16:49 2020

@author: Tobias
"""

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

def train_model(model_location, train_x, train_y, test_x, test_y):
    num_versions = 30
    print("Training " + str(num_versions) + " models")
    for i in range(30):
        version = "V" + str(i+1) + "_"
        model = load_model(model_location)
        model_name = model.name

        #log_dir=os.path.join('logs')
        IMG_SIZE = 128
        train_x = np.array(train_x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        train_y = np.array(train_y)
        test_x = np.array(test_x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        test_y = np.array(test_y)
        
        #checkpoint_loss = ModelCheckpoint('C:/Users/Tobias/CNN/Thesis_Models/' + str(model_name) + "/saved_models/" + version + model_name + "_{epoch:02d}-{loss:.2f}_best_model.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
        checkpoint_val_loss = ModelCheckpoint('C:/Users/Tobias/CNN/Thesis_Models/' + str(model_name) + "/saved_models/" + version + model_name + "_{epoch:02d}-{val_loss:.2f}_best_val_model.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1)
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = 100000000)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        history = model.fit(train_x, train_y, batch_size=16, validation_split=0.15, epochs = 1, callbacks=[checkpoint_val_loss, early_stopping, tensorboard_callback])
        print("MODEL EVALUATION:")
        model.evaluate(test_x, test_y, batch_size=16, verbose=2)

    return history
