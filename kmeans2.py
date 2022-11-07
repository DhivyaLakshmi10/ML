import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve

def dist(x, y):
    n = len(x)
    dist=0
    for i in range(n):
        dist += np.square(x[i]-y[i])
    return np.sqrt(dist)

def KMeans(data, n_clusters=3):
    
    n = len(data[0])
    
    cluster_centres_old = [data[random.randint(0, len(data)-1)] for i in range(n_clusters)]
    
    cluster_centres_new = list(cluster_centres_old)
    
    while True: 
        labels =[]
        count = np.zeros(n_clusters)
        
        for i in range(len(data)):
            distances = []
            for j in range(len(cluster_centres_old)):
                distances.append(dist(data[i], cluster_centres_old[j]))
            label = distances.index(min(distances))
            labels.append(label)
            count[label]+=1
            
        
        for i in range(n_clusters):
            cluster_centres_new[i]  = np.zeros(n)
        
        for i in range(len(data)):
            cluster_centres_new[labels[i]] += data[i]/count[labels[i]]
        
        if np.array_equal(cluster_centres_old, cluster_centres_new):
            break
        else:
            cluster_centres_old  = list(cluster_centres_new)
            
    return { "labels": labels, "cluster_centres" : cluster_centres_old }
df = pd.read_csv('Iris.csv')
cols = df.columns.to_list()
op = cols[5]
X = df.iloc[:,0:5]
Y = df[op].to_numpy()
print(KMeans(X.to_numpy(), 3))
