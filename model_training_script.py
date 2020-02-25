# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 00:16:49 2020

@author: Tobias
"""

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import csv
import os



def show(history, version, model_name):
    val_acc = history['val_accuracy']
    val_loss = history['val_loss']
    acc = history['accuracy']
    loss = history['loss']
    epochs = range(len(acc))
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(epochs, acc, 'b', label='Training acc')
    ax.plot(epochs, loss, 'c', label='Training loss')
    ax.plot(epochs, val_acc, 'r', label='Validation acc')
    ax.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.ylim(top=1.5, bottom=0.0)
    plt.title("Training History")
    ax.legend()
    fig.savefig('C:/Users/Tobias/CNN/Thesis_Models/' + str(model_name) + '/' + version + str(model_name) + '.png')

def train_model(model_location, train_x, train_y, test_x, test_y):
    num_versions = 30
    print("Training " + str(num_versions) + " models")
    for i in range(30):
        version = "V" + str(i+1) + "_"
        model = load_model(model_location)
        model_name = model.name
        IMG_SIZE=model.input.shape[1]
        
        folder_name = os.path.join('C:/Users/Tobias/CNN/Thesis_Models/' + str(model_name) + '/saved_models/' + version + '/')
        if (not os.path.exists(folder_name)):
            os.mkdir(folder_name)
            
        #log_dir=os.path.join('logs')
        train_x = np.array(train_x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        train_y = np.array(train_y)
        test_x = np.array(test_x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        test_y = np.array(test_y)
        
        csv_logger = CSVLogger(version + str(model_name) + '_training.csv', separator=';')
        checkpoint_val_loss = ModelCheckpoint(folder_name + version + model_name + "_{epoch:02d}-{val_loss:.2f}_best_val_model.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=1)
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = 100000000)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        history = model.fit(train_x, train_y, batch_size=16, validation_split=0.15, epochs = 1000, callbacks=[checkpoint_val_loss, csv_logger, early_stopping])
        #show(history.history,version,model_name)
        print("MODEL EVALUATION:")
        eval_metrics = model.evaluate(test_x, test_y, batch_size=16, verbose=2)
        file = open(version + str(model_name) +'_evaluation.csv', 'a')
        with file as eval_file:
            metrics = ['loss', 'accuracy']
            writer = csv.DictWriter(eval_file, delimiter=";", fieldnames=metrics) 
            writer.writeheader()
            writer.writerow({'loss' : eval_metrics[0], 'accuracy' : eval_metrics[1]})
    return history