import cv2
import numpy as np
import pickle

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

with open('classes.pickle', 'rb') as f:
    classes = pickle.load(f)
with open('data.pickle', 'rb') as f:
    datas = pickle.load(f)

if datas is not None:
    dim = len(datas[0])
else:
    dim = -1

# training = datas[0:80]
# target = classes[0:80]

# x_test = datas[80:100]
# test_target = classes[80:100]

# print("Running MLP >>>")

# model = Sequential()
# # Dense(64) is a fully-connected layer with 64 hidden units.
# # in the first layer, you must specify the expected input data shape
# model.add(Dense(128, input_dim=dim, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(128, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(128, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(128, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(2, init='uniform'))
# model.add(Activation('softmax'))

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='mean_squared_error', optimizer=sgd)

# model.fit(training, target, nb_epoch=500, batch_size=200)
# score = model.evaluate(x_test, test_target, batch_size=200)

# print("score : "+str(score))
# print(model.predict_classes(x_test))


nbWindow = 10
nbDatas = len(datas)
scores = np.zeros(nbWindow)
for x in range(0, nbWindow):
    training = np.zeros(((nbWindow-1)*nbDatas/nbWindow, len(datas[0])))
    target = np.zeros(((nbWindow-1)*nbDatas/nbWindow, len(classes[0])))
    training[0:x*nbDatas/nbWindow] = datas[0:x*nbDatas/nbWindow]
    target[0:x*nbDatas/nbWindow] = classes[0:x*nbDatas/nbWindow]
    x_test = datas[x*nbDatas/nbWindow:(x+1)*nbDatas/nbWindow]
    test_target = classes[x*nbDatas/nbWindow:(x+1)*nbDatas/nbWindow]
    training[x*nbDatas/nbWindow:(nbWindow-1)*nbDatas/nbWindow] = datas[(x+1)*nbDatas/nbWindow:nbDatas]
    target[x*nbDatas/nbWindow:(nbWindow-1)*nbDatas/nbWindow] = classes[(x+1)*nbDatas/nbWindow:nbDatas]

    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape
    model.add(Dense(128, input_dim=dim, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(128, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(128, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(128, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(2, init='uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    model.fit(training, target, nb_epoch=500, batch_size=200)
    scores[x] = model.evaluate(x_test, test_target, batch_size=200)
    print(scores[x])
    print(model.predict_classes(x_test))
    for x in range(0, len(test_target)):
        if test_target[x][0] == 1:
            if x == 0:
                print "[0",
            else:
                if x == len(test_target) - 1:
                    print "0]",
                else:
                    print "0",
        else:
            if x == 0:
                print "[1",
            else:
                if x == len(test_target) - 1:
                    print "1]",
                else:
                    print "1",
        pass
    pass
    print " <- Good Classes"

print "Scores :", scores, np.mean(scores)
