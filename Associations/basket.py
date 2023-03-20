import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def convert(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

encoding = 'unicode_escape'
df = pd.read_csv('basket.csv', encoding= 'unicode_escape')
df = df.drop(['StockCode','InvoiceDate','UnitPrice','CustomerID'], axis=1)

df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]

basket = (df[df['Country'] == 'Poland']
            .groupby(['InvoiceNo', 'Description'])['Quantity']
            .sum().unstack().reset_index().fillna(0)
            .set_index('InvoiceNo'))


basket_sets = basket.applymap(convert)
basket_sets.drop('POSTAGE', inplace=True, axis=1)

poland_items = apriori(basket_sets, min_support=0.11, use_colnames=True)
rules = association_rules(poland_items, metric="confidence", min_threshold=0)
# lift >6 and confidence >= 0.8
rules = rules[
    (rules['lift'] >= 6) &
    (rules['confidence'] >= 0.8)]

# print(rules['antecedents'].value_counts())
# print(rules['consequents'].value_counts())
rules = rules[rules['antecedents'].astype(str).str.contains('CERAMIC BOWL WITH STRAWBERRY DESIGN')]
rules = rules[rules['consequents'].astype(str).str.contains('CERAMIC STRAWBERRY CAKE MONEY BANK')]

pd.set_option('display.max_columns', None)
print(rules)



