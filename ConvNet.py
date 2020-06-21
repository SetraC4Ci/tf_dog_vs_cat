#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np

x = np.array(pickle.load(open("x.pickle", "rb")))
y = np.array(pickle.load(open("y.pickle", "rb")))

x = x/255.0

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
optimizer="adam",
metrics=['accuracy'])

model.fit(x=x, y=y, batch_size=32, epochs=15, validation_split=0.1)
model.save("dog_cat_classifier_model.tf")

