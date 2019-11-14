import tensorflow as tf
from tensorflow import keras
import numpy as np



data = keras.datasets.reuters

(train_data, train_label), (test_data, test_label) = data.load_data(num_words=10000)

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=0, padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=0, padding="post", maxlen=250)

print(train_data)
print(train_label)


word_index = data.get_word_index()


print(word_index)


model = keras.Sequential()
model.add(keras.layers.Embedding(10000,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(180, activation="relu"))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(46, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# Max Pooling if you want to see something in your data, global average pooling if your entire sequence matters.
# Softmax Multi Class
# Sigmoid 2 class
# categorical = multi-class (more than 2)
# binary = not multi-class
xdata1 = train_data[:10000]
xdata2 = train_data[10000:]

ylabel1 = train_label[:10000]
ylabel2 = train_label[10000:]

fitModel = model.fit(xdata1, ylabel1, batch_size=512, epochs=60,verbose=1, validation_split=0.1)

results = model.evaluate(test_data, test_label)
print(results)
predict = model.predict([test_data])
print(np.argmax(predict[1]))
print(test_label[1])
for i in range(0,120):
    if np.argmax(predict[i]) == test_label[i]:
        print("Guess: " + str(np.argmax(predict[i]))+ "  / Actual: " + str(test_label[i]) + " / Correct: Yes!")
    else:
        print("Guess: " + str(np.argmax(predict[i])) + " / Actual: " + str(test_label[i]) + " / Correct: Bad!")