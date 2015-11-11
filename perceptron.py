# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.optimizers import SGD
import numpy as np

# model = Sequential()
# # Dense(64) is a fully-connected layer with 64 hidden units.
# # in the first layer, you must specify the expected input data shape:
# # here, 20-dimensional vectors.
# model.add(Dense(64, input_dim=20, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(64, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(2, init='uniform'))
# model.add(Activation('softmax'))

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='mean_squared_error', optimizer=sgd)

X_all = np.array([np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4])])
print X_all
y_all = np.array([np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4])])
cross = 2
tot = len(X_all)
pas = tot/cross


for x in range(0, cross):
    X_train = np.zeros(tot - pas)
    y_train = np.zeros(tot - pas)
    X_test = np.zeros(pas)
    y_test = np.zeros(pas)
    for i in range(0, pas*x):
        X_train[i] = X_all[i]
        y_train[i] = y_all[i]
    for i in range(pas*x, pas*(x+1)):
        X_test[i-pas*x] = X_all[i]
        y_test[i-pas*x] = y_all[i]
    for i in range(pas*(x+1), tot):
        X_train[i-pas] = X_all[i]
        y_train[i-pas] = y_all[i]

    print X_train
    print y_train
    print X_test
    print y_test
    pass

# model.fit(X_train, y_train, nb_epoch=20, batch_size=16)
# score = model.evaluate(X_test, y_test, batch_size=16)
