import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import csv

# Using the UCI Iris dataset. The Hello world of ML
# 1. Iris-versicolor
# 2. Iris-virginica
# 3. Iris-setosa
data = []
labels = []
class_names = ['Iris-versicolor', 'Iris-virginica', 'Iris-setosa']

with open('iris.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        data.append(np.array([float(row[0]),float(row[1]),float(row[2]),float(row[3])]))
        labels.append(class_names.index(row[4]))
print("Imported data")

# Separating the data into training, testing and validation
training_data = []
training_labels = []
testing_data = []
testing_labels = []

for i in range(len(data)):
    if i % 4 < 3:
        training_data.append(data[i])
        training_labels.append(labels[i])
    else:
        testing_data.append(data[i])
        testing_labels.append(labels[i])
print("Separated data")

shape = training_data[0].shape
model = keras.Sequential([
    keras.layers.Dense(9, input_shape=shape),
    keras.layers.Dense(3)
])

print("Compiling model")
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy']);

print("Fitting model")
model.fit(np.array(training_data), np.array(training_labels), epochs=100)

predictions = model.predict(np.array(testing_data))

correct = 0
for i in range(len(predictions)):
    predicted = np.argmax(predictions[i])
    actual = testing_labels[i]
    if predicted == actual:
        correct += 1
    print("%d %d" % (predicted, actual))
print("Accuracy: %f" % (correct / len(predictions)))

#print(str(pre) + " " + str(testing_labels[i]))
