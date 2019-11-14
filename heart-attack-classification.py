from sklearn.utils import resample
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
print(len(class_names))
df = pd.read_csv("/users/programming/Desktop/datasets/letter-recognition.data")
print(df.head())
df = df.replace("T", 1)
for i in range(0,26):
    df = df.replace(class_names[i].upper(),i)
print(df.head())
print(df.tail())
predict = "Letter"
x = np.array(df.drop([predict], 1)) # Features
y = np.array(df[predict]) # Labels
print(x[0])
print(y[0])

train_data, test_data, train_label, test_label = train_test_split(x, y, test_size = 0.1, random_state = 1)
model = keras.Sequential()
model.add(keras.layers.Dense(85, input_dim=16, activation="selu"))
model.add(keras.layers.Dropout(0.15))
model.add(keras.layers.Dense(60, activation="selu"))
model.add(keras.layers.Dense(45, activation="selu"))
model.add(keras.layers.Dense(10, activation="selu"))
model.add(keras.layers.Dense(6, activation="selu"))
model.add(keras.layers.Dense(26, activation="softmax"))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data, train_label, epochs=32, batch_size=10, validation_split=0.05)

results = model.evaluate(test_data, test_label)

print(results)

prediction = model.predict(test_data)
times = 100
correct = 0
for i in range(0,times):
    if np.argmax(prediction[i]) == test_label[i]:
        print(str(np.argmax(prediction[i])) + " Correct! ")
        correct += 1
    else:
        print("Pred: " + str(np.argmax(prediction[i])) + " Correct! Actual: " + str(test_label[i]))
custom_acc = correct/times
print(custom_acc)

