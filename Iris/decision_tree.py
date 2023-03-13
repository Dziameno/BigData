import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv("iris.csv")

all_inputs = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
all_classes = df['variety'].values
(train_set, test_set, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=22044)

print('Train Set\n',train_set)
print('\nTest Set\n',test_set)

print('\nAny NULL values?\n',df.isnull().any())
print('\nData types of dataset:\n',df.dtypes)
print('\nStatistics of dataset:\n',df.describe())

# df['petal.width'].plot.hist()
# plt.show

sns.pairplot(df, hue='variety')

dtc = tree.DecisionTreeClassifier()
dtc.fit(train_set, train_classes)

tree.plot_tree(dtc,
               feature_names=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'],
                class_names=['Setosa', 'Versicolor', 'Virginica'],
                filled=True)
plt.savefig('train_decision_tree.png')

dts = tree.DecisionTreeClassifier(random_state=0)
dts.fit(test_set, test_classes)

tree.plot_tree(dts,
                feature_names=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'],
                class_names=['Setosa', 'Versicolor', 'Virginica'],
                filled=True)
plt.savefig('test_decision_tree.pdf')


print('\nAcc(score): ', dtc.score(test_set, test_classes))
print('\nAcc(predict): ', np.mean(dtc.predict(test_set) == test_classes))


cm = confusion_matrix(test_classes, dtc.predict(test_set))
print('\n',cm)




