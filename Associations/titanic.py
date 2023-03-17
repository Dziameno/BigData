import random

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import mlxtend as ml

df = pd.read_csv('titanic.csv')
df = df.drop(df.columns[0], axis=1)

people = []
cols = ['Class','Sex','Age','Survived']
for col in cols:
    people.append(pd.get_dummies(df[col]))

titanic_people = pd.concat(people, axis=1)
# print(titanic_people)

def convert(x):
    if x == 0:
        return False
    else:
        return True

titanic_people = titanic_people.applymap(convert)

titanic_itemsets = apriori(titanic_people, min_support=0.005, use_colnames=True)

rules = association_rules(titanic_itemsets, metric='confidence', min_threshold=0)

rules = rules[rules['confidence'] > 0.8]

rules = rules[~rules['antecedents'].astype(str).str.contains('Yes')]
rules = rules[~rules['antecedents'].astype(str).str.contains('No')]

rules = rules[rules['consequents'].astype(str).str.contains('Yes')]

rules['consequents'] = rules['consequents'].apply(lambda x: x - {'Adult'})

pd.set_option('display.max_columns', None)
print(rules)

plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence')
plt.savefig('titanic.png')


















