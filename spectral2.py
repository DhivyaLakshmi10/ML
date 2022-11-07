import pandas as pd
import numpy as np

df = pd.read_csv("Iris.csv")
del df['Id']
# df = df[0:1000]

df_arr = df[df.columns[:-1]].to_numpy()
#print(df_arr)
A = []

for i in range(len(df_arr)):
    rowe = []
    for j in range(len(df_arr)):
        dist = 0
        if (i == j):
            rowe.append(dist)
        else:
            for k in range(len(df_arr[i])):
                dist += (df_arr[i][k] - df_arr[j][k]) ** 2
            rowe.append(dist ** 0.5)
    A.append(rowe)
A = np.array(A)
#print(A)
D = []

for i in range(len(df_arr)):
    rowe = []
    for j in range(len(df_arr)):
        deg = 0
        if (i == j):
            deg = len(A[0]) - 1
            rowe.append(deg)
        else:
            rowe.append(deg)
    D.append(rowe)
D = np.array(D)
#print(D)

L = D - A
val, vec = np.linalg.eig(L)
print(val,vec)
val=np.argsort(val)
i=0
for i in range(len(val)):
    if i>0:
        break
print(i)
f=vec[i]
f=list(f)
f=[1 if i>0 else 0 for i in f]
print(f)
