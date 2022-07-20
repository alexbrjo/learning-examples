import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
import csv

# Using the UCI Iris dataset. The Hello world of ML
# 1. Iris-versicolor
# 2. Iris-virginica
# 3. Iris-setosa
X = [] # parameters
y = [] # labels
class_names = ['Iris-versicolor', 'Iris-virginica', 'Iris-setosa']

with open('iris.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        X.append(np.array([float(row[0]),float(row[1]),float(row[2]),float(row[3])]))
        y.append(class_names.index(row[4]))
print("Imported data")

# Separating the data into training, testing and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

shape = X_train[0].shape
model = keras.Sequential([
    keras.layers.Dense(9, input_shape=shape),
    keras.layers.Dense(3)
])

print("Compiling model")
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy']);

print("Fitting model")
model.fit(np.array(X_train), np.array(y_train), epochs=100)

predictions = model.predict(np.array(X_test))

correct = 0
for i in range(len(predictions)):
    predicted = np.argmax(predictions[i])
    actual = y_test[i]
    if predicted == actual:
        correct += 1
    print("Predicted class: %d actual: %d" % (predicted, actual))
print("Accuracy: %f" % (correct / len(predictions)))
