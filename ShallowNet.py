
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten
from tensorflow.keras.models import Model
    
from tensorflow.keras.datasets import cifar100, cifar10
#(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

def init_shallownet(in_dim, classes):
    input_layer = Input(shape = in_dim, dtype='float32', name='in')  
    l1 = layers.Conv2D(32, (3,3), padding='same', name='conv_1')(input_layer)
    f1 = Flatten()(l1)
    d1 = Dense(classes, activation='softmax')(f1)
    return Model(inputs = input_layer, outputs = d1, name="model")
    
shallowNet = init_shallownet((32,32, 3), 10)

shallowNet.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
shallowNet.fit(x_train, y_train, batch_size=16, epochs=10, validation_split=0.15)
shallowNet.evaluate(x_test, y_test, batch_size=16, verbose=2)
