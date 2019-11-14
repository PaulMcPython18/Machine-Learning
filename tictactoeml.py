import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
# print(dataset.head())
# '''Turning string values into integer values and applying to pandas dataframe'''
# le = preprocessing.LabelEncoder()
# le.fit(dataset['X2'])
# le.transform(dataset['X2'])
# print(list(le.inverse_transform([2, 2, 1])))
# print(dataset.head())
# lb_style = LabelBinarizer()
# lb_results = lb_style.fit_transform(dataset["X2"])
# pd.DataFrame(lb_results, columns=lb_style.classes_).head()
# print(dataset.head())
df = pd.read_csv("/users/programming/Desktop/datasets/tic-tac-toe.data")
df = df.replace("x", 1)
df = df.replace("o", 0)
df = df.replace("b", 2)
df = df.replace("positive",1)
df = df.replace("negative", 0)

predict = "X10"
x = np.array(df.drop([predict], 1)) # Features
y = np.array(df[predict]) # Labels



train_data, test_data, train_label, test_label = train_test_split(x, y, test_size = 0.1, random_state = 1)
print(train_data)
print(train_label)
model = keras.Sequential()
model.add(keras.layers.Dense(10, input_dim=9, activation="relu"))
model.add(keras.layers.Dense(15, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
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


