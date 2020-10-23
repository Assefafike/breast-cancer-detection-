import numpy as np
import pandas as pd
import tensorflow as tf
#importing data
dataset = pd.read_excel('dataset.xlsx')
x = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values
#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#build ANN
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
#train model
ann.compile(optimizer='adam', loss='mean_squared_error')
ann.fit(x_train, y_train, batch_size=22,  epochs=100)
y_pred = ann.predict(x_test)
np.set_printoptions(precision=2)
np.concatenate(y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1), 1)
#print(y_pred)
#print(y_test)

