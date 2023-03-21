import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

iris = pd.read_csv('iris.csv')

all_inputs = iris[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
all_classes = iris['variety'].values
(train_set, test_set, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=22044)

clf1 = MLPClassifier(solver='lbfgs',
                    alpha=1e-5,
                    hidden_layer_sizes=(2,),
                    random_state=22044)

clf1.fit(train_set, train_classes)
clf1.score(train_set, train_classes)
print('input: 4 neurons; hidden: 2 neurons; output: 1 neuron')
predictions_train = clf1.predict(train_set)
predictions_test = clf1.predict(test_set)
train_score = accuracy_score(predictions_train, train_classes)
print("score on train_set: ", train_score)
test_score = accuracy_score(predictions_test, test_classes)
print("score on test_set: ", test_score)

clf2 = MLPClassifier(solver='lbfgs',
                    alpha=1e-5,
                    hidden_layer_sizes=(3,),
                    random_state=22044)

clf2.fit(train_set, train_classes)
clf2.score(train_set, train_classes)
print('\ninput: 4 neurons; hidden: 3 neurons; output: 1 neuron')
predictions_train = clf2.predict(train_set)
predictions_test = clf2.predict(test_set)
train_score = accuracy_score(predictions_train, train_classes)
print("score on train_set: ", train_score)
test_score = accuracy_score(predictions_test, test_classes)
print("score on test_set: ", test_score)

clf3 = MLPClassifier(solver='lbfgs',
                    alpha=1e-5,
                    hidden_layer_sizes=(3,3),
                    random_state=22044)

clf3.fit(train_set, train_classes)
clf3.score(train_set, train_classes)
print('\ninput: 4 neurons; hidden1: 3 neurons; hidden2: 3 neurons output: 1 neuron')
predictions_train = clf3.predict(train_set)
predictions_test = clf3.predict(test_set)
train_score = accuracy_score(predictions_train, train_classes)
print("score on train_set: ", train_score)
test_score = accuracy_score(predictions_test, test_classes)
print("score on test_set: ", test_score)

