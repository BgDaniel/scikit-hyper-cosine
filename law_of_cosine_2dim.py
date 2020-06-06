from keras.models import Sequential
from keras.layers import Dense
from data import samples
from sklearn import model_selection as ms
import numpy as np
from data import samples as sp
import matplotlib as mplt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from data.trigonometric_laws.hyper_triangles import *
from sklearn import preprocessing
from matplotlib import pyplot as plt
import seaborn as sbn
import pandas as pd

DEVIATION = 10e-5
cross_check = False

# Create a Sequential model
model = Sequential()
model.add(Dense(6, input_shape=(3,), activation='sigmoid'))
model.add(Dense(36, activation='sigmoid'))
model.add(Dense(36, activation='sigmoid'))
model.add(Dense(6, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['mse'])

model.summary()

data, X, y = sp.get_data('triangles.csv', 'A', 'B', 'Gamma', target='C')

# cross check data
if cross_check:
    for i, _X in enumerate(X):
        deviation = abs(LAW_OF_COSINE_I3(_X[0], _X[1], _X[2], y[i]) - .0)
        assert deviation < DEVIATION, 'Deviation too high for triangle number {0}: {1}'.format(i, deviation) 

# get ranges
a_min, a_max = min(X[:,0]), max(X[:,0])
b_min, b_max = min(X[:,1]), max(X[:,1])
gamma_min, gamma_max = min(X[:,2]), max(X[:,2])
c_min, c_max = min(y), max(y)

# inspect dependencies
sbn.pairplot(data, diag_kind="kde")
plt.show()


# scale data to range [0,1]
X_scaler = preprocessing.MinMaxScaler(feature_range=(- 5.0, + 5.0))
X_scaler.fit(X)
X_scaled = X_scaler.transform(X)

y = y.reshape(-1, 1)
y_scaler = preprocessing.MinMaxScaler()
y_scaler.fit(y)
y_scaled = y_scaler.transform(y)


# split into traning and test sets
X_train, X_test, y_train, y_test = ms.train_test_split(X_scaled, y_scaled, test_size = 0.1, random_state=31)

# define early stopping by looking at loss function ('mse')
monitor_val_acc = EarlyStopping(monitor = 'binary_crossentropy', patience=3, min_delta=10e-6)

# train model
history = model.fit(X_train, y_train, epochs=100, callbacks=[monitor_val_acc], validation_data=(X_test, y_test))

# Plot train vs test accuracy/loss during training
#  "Accuracy"
plot = False

if plot:
    plt.plot(history.history['binary_crossentropy'])    
    plt.plot(history.history['val_binary_crossentropy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show(block=False)
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

# make prediction
y_pred = model.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred)

print(y_pred)



