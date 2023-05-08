import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('winequality-white.csv', sep=';')
all_inputs = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
                    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']].values
all_classes = df['quality'].values

(train_input, test_input, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=22044)

#print(df['quality'].value_counts())

# Decision Tree
tree_model = tree.DecisionTreeClassifier()
tree_model.fit(train_input, train_classes)
tree.plot_tree(tree_model)
plt.savefig('1.1-DecisionTree.pdf', dpi=1000)
plt.clf()
tree_accuracy = tree_model.score(test_input, test_classes)
print("1.1DT acc: ", tree_accuracy)


tree_model_small = tree.DecisionTreeClassifier(max_depth=4)
tree_model_small.fit(train_input, train_classes)
tree.plot_tree(tree_model_small,
               feature_names=['fixed acidity', 'volatile acidity', 'citric acid',
                              'residual sugar','chlorides', 'free sulfur dioxide',
                              'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'],
                class_names=['3', '4', '5', '6', '7', '8', '9'],
               filled=True)
plt.savefig('1.2-DecisionTreeSmall.pdf', dpi=300)
plt.clf()
tree_accuracy_small = tree_model_small.score(test_input, test_classes)
print("1.2DTS acc: ", tree_accuracy_small)


y_pred = tree_model.predict(test_input)
cm = confusion_matrix(test_classes, y_pred)
fig, ax = plt.subplots(figsize=(7, 7))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
plt.xticks(range(7), ['3', '4', '5', '6', '7', '8', '9'])
plt.yticks(range(7), ['3', '4', '5', '6', '7', '8', '9'])
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('Actual', fontsize=18)
plt.title('Decision Tree Confusion Matrix', fontsize=18)
plt.accuracy = tree_accuracy
plt.savefig('1.3-DecisionTreeConfusionMatrix.png')
plt.clf()

#Naive Bayes
gnb = GaussianNB()
gnb.fit(train_input, train_classes)
gnb_accuracy = gnb.score(test_input, test_classes)
print("2.NB acc: ", gnb_accuracy)

y_pred = gnb.predict(test_input)
cm = confusion_matrix(test_classes, y_pred)
fig, ax = plt.subplots(figsize=(7, 7))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
plt.xticks(range(7), ['3', '4', '5', '6', '7', '8', '9'])
plt.yticks(range(7), ['3', '4', '5', '6', '7', '8', '9'])
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('Actual', fontsize=18)
plt.title('Naive Bayes Confusion Matrix', fontsize=18)
plt.accuracy = gnb_accuracy
plt.savefig('2.2-NaiveBayesConfusionMatrix.png')
plt.clf()

#KNN
k_range = range(5, 25)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_input, train_classes)
    scores.append(knn.score(test_input, test_classes))
print("3.KNN acc: ", max(scores), " for K:", k_range[scores.index(max(scores))])
plt.figure()
plt.title('KNN accuracy for different K values')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.scatter(k_range, scores)
plt.xticks([5, 10, 15, 20, 25])
plt.savefig('3.1-KNNBestKValue.png')
plt.clf()

y_pred = knn.predict(test_input)
cm = confusion_matrix(test_classes, y_pred)
fig, ax = plt.subplots(figsize=(7, 7))
ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center', size='xx-large')
plt.xticks(range(7), ['3', '4', '5', '6', '7', '8', '9'])
plt.yticks(range(7), ['3', '4', '5', '6', '7', '8', '9'])
plt.xlabel('Predicted', fontsize=18)
plt.ylabel('Actual', fontsize=18)
plt.title('KNN Confusion Matrix', fontsize=18)
plt.accuracy = max(scores)
plt.savefig('3.2-KNNConfusionMatrix.png')
plt.clf()





