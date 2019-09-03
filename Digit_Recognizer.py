import numpy as np 
import pandas as pd 
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, BatchNormalization, Activation, Dropout, DepthwiseConv2D
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils import np_utils
from keras.datasets import mnist

import os
tf.compat.v1.disable_eager_execution()

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
val = pd.read_csv('/content/drive/My Drive/train.csv').values
test = pd.read_csv('/content/drive/My Drive/test.csv').values.astype('float32')

X_train = np.vstack((X_train, X_test))
Y_train = np.hstack((Y_train, Y_test))
X_val = val[:,1:].astype('float32')
Y_val = val[:,0].astype('int32')

X_train = X_train.astype('float32')
Y_train = Y_train.astype('int32')

X_train = X_train.reshape(-1,28,28,1)
X_val = X_val.reshape(-1,28,28,1)
X_test = test.reshape(-1,28,28,1)
print(X_train.shape,X_val.shape,Y_train.shape,Y_val.shape)

X_train /= 255
X_val /= 255
X_test /= 255

Y_train = to_categorical(Y_train, 10)
Y_val = to_categorical(Y_val, 10)
'''
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
'''
def create_model():
    
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = 5, activation = 'relu', padding = 'same', input_shape = (28,28,1)))
#     model.add(BatchNormalization())
    model.add(Conv2D(filters = 32, kernel_size = 5, activation = 'relu', padding = 'same'))
#     model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = 2))
#     model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
#     model.add(BatchNormalization())
    model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
#     model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = 2, strides = 2))
#     model.add(Dropout(rate=0.5))

    
    model.add(Flatten())
    model.add(Dense(20, activation = 'relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(rate=0.5))
    model.add(Dense(10, activation = 'softmax'))
    
    
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model

model = create_model()
model.summary()

reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_acc', patience = 3, verbose = 1, factor = 0.3, min_lr = 0.00001)
checkpoint = ModelCheckpoint('mnist_weights.h5', monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 1e-10, patience = 10, verbose = 1, restore_best_weights = True)

callbacks = [reduce_learning_rate, checkpoint, early_stopping]

history = model.fit(
#                     datagen.flow(X_train, Y_train, batch_size = 32), 
                    X_train, 
                    Y_train, 
                    batch_size = 32,
                    epochs = 30,
                    validation_data = (X_val, Y_val),  
                    callbacks = callbacks,
                    verbose = 1, 
                    shuffle = True)
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

test_labels = model.predict_classes(X_test, verbose = 2)

sub = pd.read_csv('/content/drive/My Drive/sample_submission.csv')
sub['Label'] = test_labels
sub.to_csv('/content/drive/My Drive/sample_submission.csv',index = False)