import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
dataset = np.loadtxt('/users/programming/downloads/red-wine-quality.csv', delimiter=';')
x = dataset[:,0:11]
y = dataset[:,11]
train_data, test_data, train_label, test_label = train_test_split(x, y, test_size = 0.2, random_state = 1)
print(len(train_data))
print(len(train_label))

model = keras.Sequential()
model.add(keras.layers.Dense(10, input_dim=11, activation="relu"))
model.add(keras.layers.Dense(15, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(50, activation="relu"))
model.add(keras.layers.Dense(25, activation="relu"))
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, train_label, epochs=500, batch_size=10, validation_split=0.05)
result = model.evaluate(test_data, test_label)
print(result)

predict = model.predict(test_data)
times = 100
correct = 0
for i in range(0,times):
    if np.argmax(predict[i]) == test_label[i]:
        print("Correct! / " + "Prediction: " + str(np.argmax(predict[i])) + " / Actual: " + str(test_label[i]))
        correct += 1
    else:
        print("Wrong! / " + "Prediction: " + str(np.argmax(predict[i])) + " / Actual: " + str(test_label[i]))
print(correct/times)