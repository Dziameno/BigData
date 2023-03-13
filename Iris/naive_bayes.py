import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("iris.csv")

all_inputs = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
all_classes = df['variety'].values
(train_set, test_set, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=22044)

gnb = GaussianNB()
gnb.fit(train_set, train_classes)
y_pred = gnb.predict(test_set)

print('\nAcc:',accuracy_score(test_classes, y_pred))

cm = confusion_matrix(test_classes, y_pred)
print('\n',cm)

# klasyfikator bayesowski jest najbardziej lub
# równie dokładny jak klasyfikator drzew decyzyjnych i
# k najbliższych sąsiadów


