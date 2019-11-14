import tensorflow as tf
from tensorflow import keras
import numpy as np
# import matplotlib.pyplot as plt

#print(train_images[0])
#print("Network Accuracy: " + str(test_acc))


mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images/255
test_images = test_images/255
# plt.imshow(train_images[0], cmap=plt.cm.binary) #greyscale
# # plt.imshow(train_images[0]) #neon
# plt.show()
# print(train_images)
# print(test_images[0])
# print(test_labels[0])
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print(train_images[0])
model = keras.Sequential([
    # tf.keras.layers.Conv1D(56, 4, activation='relu', input_shape=(28, 28)),
    # tf.keras.layers.MaxPooling1D(3, 3),
    # tf.keras.layers.Conv1D(56, 4, activation='relu'),
    # tf.keras.layers.MaxPooling1D(3, 3),
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(784, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_labels[0])
prediction = model.predict(test_images)
answer = np.argmax(prediction[0])
print(answer)

