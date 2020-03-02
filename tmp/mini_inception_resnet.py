from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation, Add, BatchNormalization, MaxPooling2D, Dropout
import tensorflow as tf

def conv_block(X, filters, kernel_size, strides):
    shortcut = X
    if strides == 1:
        while (kernel_size > 1): # replacing the kernel_size with 3x3 kernels to cover the same area while saving in computational cost
            X = Conv2D(filters, kernel_size=3, strides=1, padding = 'same')(X)
            kernel_size = ((kernel_size - 3)/1) + 1
    else:
        X = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding = 'same')(X)
    X = Add()([shortcut, X])
    X = common_layers(X)
    return X
    
def common_layers(X):
    X = BatchNormalization()(X)
    X = tf.keras.layers.LeakyReLU()(X)
    return X
    
def conv_layer(X, filters, kernel_size, strides):
    if strides == 1:
        while (kernel_size > 1): # replacing the kernel_size with 3x3 kernels to cover the same area while saving in computational cost
            X = Conv2D(filters, kernel_size=3, strides=1, padding = 'same')(X)
            kernel_size = ((kernel_size - 3)/1) + 1
            X = common_layers(X)
    else:
        X = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding = 'same')(X)
        X = common_layers(X)
    
    return X

def maxPool_layer(X, pool_size, strides):
    X = MaxPooling2D(pool_size = pool_size, strides = strides, padding = "valid")(X)
    return X

def inception_residual_block(X, filters):
    shortcut = X
    split_4 = int(filters/4)
    
    l1 = Conv2D(32, 1, 1, padding='same')(X)
    l1 = common_layers(l1)
    l1 = conv_layer(l1, split_4, 7, 1)
    
    l2 = Conv2D(32, 1, 1, padding='same')(X)
    l2 = common_layers(l2)
    l2 = conv_layer(l2, split_4, 5, 1)
    
    l3 = Conv2D(32, 1, 1, padding='same')(X)
    l3 = common_layers(l3)
    l3 = conv_layer(l2, split_4, 3, 1)
    
    l4 = Conv2D(16, 1, 1, padding='same')(X)
    l4 = common_layers(l4)
    
    l5 = MaxPooling2D(pool_size = 3, strides = 1, padding = "same")(X)
    l5 = Conv2D(16, 1, 1, padding='same')(l5)
    l5 = common_layers(l5)
    
    X = tf.keras.layers.concatenate([l1,l2,l3,l4,l5])
    X = Add()([shortcut, X])
    return X

def dense_block(X, categories):
    X = Flatten()(X)
    #X = Dense(64, name="dense_pre_out")(X)
    #X = tf.keras.layers.LeakyReLU()(X)
    #X = Dropout(0.2, name="dout_final")(X)
    X = Dense(categories, name="dense_out")(X)
    return X

def init_model(dimensions, categories):
    inputs = tf.keras.Input(shape=dimensions)
    X = Conv2D(128, 3, 1, padding='same')(inputs)
    X = common_layers(X)
    X = maxPool_layer(X, 3, 2)
    X = Dropout(0.2, name="dout_init")(X)
    X = inception_residual_block(X, 128)
    X = common_layers(X)
    X = maxPool_layer(X, 3, 2)
    X = Dropout(0.2, name="dout_mid")(X)
    X = inception_residual_block(X, 128)
    X = common_layers(X)
    X = maxPool_layer(X, 3, 2)
    X = Dropout(0.2, name="dout_final")(X)
    X = dense_block(X, categories)
    outputs = Activation("softmax", name="softmax")(X)
    model = tf.keras.Model(inputs = inputs, outputs = outputs, name ="mini_inception_resnet")
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model
