# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 07:46:16 2022

@author: dhivya
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def findPriors(train, output):
    # print(train.groupby(output).size().div(len(train)))
    return train.groupby(output).size().div(len(train))
def constructCPT(train, features, output):
    class_probs = findPriors(train, output)
    cpt = dict()
    for i in features:
        cpt[i] = train.groupby([i, output]).size().div(len(train)).div(class_probs)
        print(i,cpt[i])
    return cpt
def predict(test, cpt, label):
    Ypred = []
    class_probs = findPriors(train, output)
    for i in range(len(test)):
        X = test[features].iloc[i].tolist()
        probs = []
        for j in label:
            prob = 1
            for i in range(len(X)):
                _cpt = cpt[features[i]]
                if (X[i], j) in _cpt.index:
                    prob *= _cpt[X[i]][j]
            prob *= class_probs[j]
            probs.append((prob, j))

        Ypred.append(max(probs)[1])
    return Ypred 
def calcMisclass(Y, Ypred):
    res = 0
    for i in range(len(Y)):
        if Y[i]!=Ypred[i]:
            res+=1
    return res
    
    
df = pd.read_csv("breast-cancer-wisconsin.csv")

df = df.replace("?", np.nan) 
df = df.dropna() 

df = df.apply(pd.to_numeric)
# df['bare_nuclei'] = pd.to_numeric(df['bare_nuclei'])

cols = list(df.columns)
print(cols)


features = cols[1:-1]
output = "class"
label = df[output].unique()

print(label)
d = len(features)

train, test = train_test_split(df, test_size=0.25)

c1 = df['class'].value_counts()[2]
c2 = df['class'].value_counts()[4]
print(c1,c2)
cpt = constructCPT(train, features, output)
Y = test[output].tolist()
Ypred = predict(test, cpt, label)

print("[Size of testing set] ", len(test))
print("[Misclassifications ] ", calcMisclass(Y, Ypred))    