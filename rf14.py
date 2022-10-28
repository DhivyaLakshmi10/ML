# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 07:33:41 2022

@author: dhivya
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("D:/PSG/PSG-Z/SEM 5/20XW58 MACHINE LEARNING LAB/ws8-random-forest/datasets/tic-tac-toe-endgame.csv")
df.head()

from sklearn import preprocessing
features = df.columns[0:9]
features

le1 = preprocessing.LabelEncoder()
    
for i in features:
    df[i] = le1.fit_transform(df[i])
df.head()

le2 = preprocessing.LabelEncoder()

df['V10'] = le2.fit_transform(df['V10'])
df.head()

X = df[features]
Y = df['V10']
from sklearn.model_selection import train_test_split

X_train, X_test,Y_train,Y_test = train_test_split(X.to_numpy(),Y.to_numpy(),test_size=0.33)
clf = RandomForestClassifier(criterion="entropy")

clf = clf.fit(X_train,Y_train)

Y_pred = clf.predict(X_test)

count = 0

for i in range(len(Y_pred)):
    if(Y_pred[i]==Y_test[i]):
        count+=1
        
print(f"Accuracy : {count/len(Y_pred)}")

# from sklearn.tree import plot_tree

# plt.figure(figsize=(100,50))
# plot_tree(clf.estimators_[0])
# plt.savefig('randomforest.png')
# dir(clf)
