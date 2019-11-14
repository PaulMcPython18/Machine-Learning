from sklearn.utils import resample
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

df = pd.read_csv("/users/programming/Desktop/datasets/tic-tac-toe.data")
df = df.replace("x", 1)
df = df.replace("o", 0)
df = df.replace("b", 2)
df = df.replace("positive",1)
df = df.replace("negative", 0)

df_majority = df[df.X10==1]
df_minority = df[df.X10==0]
print(df_majority.head())
print(df_minority.head())

df_minority_upsampled = resample(df_minority,
                                 replace=True,  # sample with replacement
                                 n_samples=626,  # to match majority class
                                 random_state=123)  # reproducible results

# Combine majority class with upsampled minority class
predict = "X10"
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
x = np.array(df_upsampled.drop([predict], 1)) # Features
y = np.array(df_upsampled[predict]) # Labels
train_data, test_data, train_label, test_label = train_test_split(x, y, test_size = 0.1, random_state = 1)

model = keras.Sequential()
model.add(keras.layers.Dense(85, input_dim=9, activation="selu"))
model.add(keras.layers.Dropout(0.15))
model.add(keras.layers.Dense(60, activation="selu"))
model.add(keras.layers.Dense(45, activation="selu"))
model.add(keras.layers.Dense(10, activation="selu"))
model.add(keras.layers.Dense(6, activation="selu"))
model.add(keras.layers.Dense(2, activation="softmax"))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, train_label, epochs=500, batch_size=10, validation_split=0.05)
print(model.evaluate(test_data, test_label))

predictions = model.predict(test_data)

times = 95

correct = 0

for i in range(0,times):
    if np.argmax(predictions[i]) == test_label[i]:
        print("Prediction: " + str(np.argmax(predictions[i])) + "/ Correct! / Real: " + str(test_label[i]))
        correct += 1
    else:
        print("Prediction: " + str(np.argmax(predictions[i])) + "/ Wrong! / Real: " + str(test_label[i]))

print("Accuracy: " + str(correct/times))
print(model.evaluate(test_data, test_label))
