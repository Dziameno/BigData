import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")

(train_set, test_set) = train_test_split(df.values, train_size= 0.7, random_state= 22044)

def classify_iris(sl, sw, pl, pw):
    if pw <= 0.6:
        return("Setosa")
    elif sl >= 5.8 and pl >= 4.9:
        return("Virginica")
    else:
        return("Versicolor")

good_predictions_counter = 0
len = test_set.shape[0]

for i in range(len):
    if classify_iris(test_set[i, 0], test_set[i, 1], test_set[i, 2], test_set[i, 3]) == (test_set[i, 4]):
        good_predictions_counter += 1

print(good_predictions_counter)
print(good_predictions_counter/len*100, "%")

df.sort_values(["petal.length"],
                    ascending=[False],
                    inplace=True)
# print(df.values)