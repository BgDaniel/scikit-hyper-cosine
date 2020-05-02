from keras.models import Sequential
from keras.layers import Dense
from data import samples
from sklearn import model_selection as ms
import numpy as np
from data import samples as sp
import matplotlib as mplt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from data.trigonometric_laws.hyper_triangles import *

DEVIATION = 10e-5

# Create a Sequential model
model = Sequential()
model.add(Dense(256, input_shape=(6,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(loss='mse', optimizer='adam', metrics = ['mse'])

model.summary()

X, y = sp.get_data('triangles.csv', 'A', 'B', 'Gamma', target='C')

# cross check data
for i, _X in enumerate(X):
    deviation = abs(LAW_OF_COSINE_I3(_X[0], _X[1], _X[2], y[i]) - .0)
    assert deviation < DEVIATION, 'Deviation too high for triangle number {0}: {1}'.format(i, deviation) 



X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = 0.1, random_state=31)


monitor_val_acc = EarlyStopping(monitor = 'val_mse', patience=3)

history = model.fit(X_train, y_train,epochs=200, callbacks=[monitor_val_acc], validation_data=(X_test, y_test))


y_pred = model.predict(X_test)

# Plot train vs test loss during training
mplt.plot_loss(history.history['loss'], history.history['val_loss'])

# Plot train vs test accuracy during training
mplt.plot_accuracy(history.history['acc'], history.history['val_acc'])

accuracy = model.evaluate(X_test, y_test)
print('Accuracy:', accuracy)

# Evaluate your model 
#print("Final lost value:",model.evaluate(time_steps, y_positions))

