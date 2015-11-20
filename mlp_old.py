import numpy as np
import pickle

import cv2
import projection as proj

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

with open('classes.pickle', 'rb') as f:
    classes = pickle.load(f)
with open('pages.pickle', 'rb') as f:
    pages = pickle.load(f)

#targets = np.zeros((len(classes),1))
#training = np.zeros((len(classes),997))
targets = np.zeros((10, 2))
training = np.zeros((10, 997))
cpt = 0

# size
nb_examples, data = training.shape

print("hpc_low/-"+pages[5]+".pgm")

x_test = np.zeros((1,997))
x_test[:][0] = proj.projection(cv2.imread("hpc_low/-"+pages[5]+".pgm"))
y_test = np.zeros((1,2))
y_test[0][int(classes[5])] = 1



for i in range(40, 50):
    print("Working on page : "+pages[i]+", class : "+str(classes[i]))
    img = cv2.imread("hpc_low/-"+pages[i]+".pgm")
    training[:][cpt] = proj.projection(img)
    targets[cpt][int(classes[i])] = 1
    cpt += 1


model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape
model.add(Dense(64, input_dim=997, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(training, targets, nb_epoch=40, batch_size=2)
score = model.evaluate(x_test, y_test, batch_size=16)

print(score)
print(model.predict_classes(x_test))
print(model.predict(x_test))