import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

iris = pd.read_csv('iris.csv')
iris['setosa'] = iris['variety'].apply(lambda x: 1 if x == 'Setosa' else 0)
iris['versicolor'] = iris['variety'].apply(lambda x: 1 if x == 'Versicolor' else 0)
iris['virginica'] = iris['variety'].apply(lambda x: 1 if x == 'Virginica' else 0)

pd.set_option('display.max_columns', None)
print(iris.head(10))

all_inputs = iris[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
all_classes = iris[['setosa', 'versicolor', 'virginica']].values
(train_set, test_set, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=22044)

clf = MLPClassifier(solver='lbfgs',
                    alpha=1e-5,
                    hidden_layer_sizes=(3,),
                    random_state=22044)

clf.fit(train_set, train_classes)
clf.score(train_set, train_classes)
print('input: 4 neurons; hidden: 3 neurons; output: 3 neurons')
predictions_train = clf.predict(train_set)
predictions_test = clf.predict(test_set)
train_score = accuracy_score(predictions_train, train_classes)
print("score on train_set: ", train_score)
test_score = accuracy_score(predictions_test, test_classes)
print("score on test_set: ", test_score)