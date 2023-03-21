import pandas as pd
from matplotlib import pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('diabetes.csv')

all_inputs = data[['pregnant-times', 'glucose-concentr', 'blood-pressure'
                    ,'skin-thickness', 'insulin', 'mass-index',
                   'pedigree-func', 'age']].values
all_classes = data['class'].values
(train_set, test_set, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=22044)

clf = MLPClassifier(solver='lbfgs',
                    alpha=1e-5,
                    activation='relu',
                    hidden_layer_sizes=(6,3),
                    max_iter = 500,
                    random_state=22044)

clf.fit(train_set, train_classes)
clf.score(train_set, train_classes)
print('input: 8n; hidden1: 6n; hidden2: 3n output: 1n')
predictions_train = clf.predict(train_set)
predictions_test = clf.predict(test_set)
train_score = accuracy_score(predictions_train, train_classes)
print("score on train_set: ", train_score)
test_score = accuracy_score(predictions_test, test_classes)
print("score on test_set: ", test_score)

plt.figure()
cm = plot_confusion_matrix(conf_mat=confusion_matrix(train_classes, predictions_train),
                            show_absolute=True,
                            show_normed=True,
                            colorbar=True)
plt.title('Confusion matrix for train_set')
plt.savefig('confusion_matrix_train.png')