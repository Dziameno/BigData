import matplotlib.pyplot as plt
import pandas as pd


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("iris.csv")

all_inputs = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
all_classes = df['variety'].values
(train_set, test_set, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=22044)

k_range = (3,11)
scores = {}
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_set, train_classes)
    y_pred = knn.predict(test_set)
    scores[k] = metrics.accuracy_score(test_classes, y_pred)
    scores_list.append(metrics.accuracy_score(test_classes,y_pred))

plt.plot(k_range,scores_list)
plt.xlabel('K')
plt.ylabel('Acc%')
plt.savefig('acc_to_k.png')
