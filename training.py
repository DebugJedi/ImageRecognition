# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 22:46:46 2021

@author: Priyank Rao
"""
import os
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from tensorflow.keras import backend as K
import gc
from keras.preprocessing.image import ImageDataGenerator

IMG_WIDTH,IMG_HEIGHT = 150,150
TRAIN_DATA_DIR = 'train' #where we have our training data
VALIDATION_DATA_DIR = 'validation'
TRAIN_SAMPLES = 20 #number training examples
VALIDATION_SAMPLES = 20
EPOCHS = 50 #number of cycles
BATCH_SIZE = 5

def build_model():
    
    # making sure the image is in right shape
    if K.image_data_format() =='channels_first':
        input_shape = (3, IMG_WIDTH, IMG_HEIGHT)
    else:
        input_shape = (IMG_WIDTH, IMG_HEIGHT,3)
        
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape= input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(32, (3,3), input_shape= input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(32, (3,3), input_shape= input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    # and we are compiling the final model with a loss function..
    model.compile(loss='binary_crossentropy',
                        optimizer='rmsprop',
                        metrics = ['accuracy'])
    
    return model

def train_model(model):
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
                    TRAIN_DATA_DIR,
                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                    batch_size=BATCH_SIZE,
                    class_mode = 'binary')
    validation_generator = test_datagen.flow_from_directory(
                VALIDATION_DATA_DIR,
                target_size=(IMG_WIDTH,IMG_HEIGHT),
                batch_size=BATCH_SIZE,
                class_mode='binary')
    model.fit_generator(
                    train_generator,
                    steps_per_epoch=VALIDATION_SAMPLES//BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data = validation_generator,
                    validation_steps = VALIDATION_SAMPLES//BATCH_SIZE)
    
    return model

def save_model(model):
    model.save('saved_model.h5')

def main():
    myModel = None
    tf.keras.backend.clear_session() #housekeeping making sure everythin loads neatly 
    gc.collect()
    myModel = build_model()
    myModel = train_model(myModel)
    save_model(myModel)

main()

