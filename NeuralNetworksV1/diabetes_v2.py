import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from matplotlib import pyplot as plt
from mlxtend.evaluate import accuracy_score, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import train_test_split

data = pd.read_csv('diabetes.csv')

all_inputs = data[['pregnant-times', 'glucose-concentr', 'blood-pressure'
                    ,'skin-thickness', 'insulin', 'mass-index',
                   'pedigree-func', 'age']].values

data['class'] = data['class'].apply(lambda x: 1 if x == 'tested_positive' else 0)
all_classes = data[['class']].values

# (train_set, test_set, train_classes, test_classes) = \
#     tf.keras.utils.split_dataset(all_inputs, all_classes,
#                              left_size=0.7, right_size=0.3,
#                              shuffle=False, seed=22044)

train_set, test_set, train_classes, test_classes = \
    train_test_split(all_inputs, all_classes,
                     train_size=0.7, random_state=22044)

# train_utils = np_utils.to_categorical(train_classes)
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model = Sequential()
model.add(Dense(6, input_dim = 8, activation = 'relu'))
model.add(Dense(3, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_set, train_classes, validation_split=0.1, epochs=50, batch_size=4)

# calculate accuracy
predictions_train = model.predict(train_set)
predictions_train = np.argmax(predictions_train, axis=1)
train_score = accuracy_score(predictions_train, train_classes)
print("score on train_set: ", train_score)
predictions_test = model.predict(test_set)
predictions_test = np.argmax(predictions_test, axis=1)
test_score = accuracy_score(predictions_test, test_classes)
print("score on test_set: ", test_score)


plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss curve')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('loss_curve_diabetes_v2.png')







