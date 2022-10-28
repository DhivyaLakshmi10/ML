
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt


ETA= 0.007

def ConfusionMatrix(X, Y, W, theta=0.5):
    Ypred = predict(X, W, theta)
    tp, fn, fp, tn= 0, 0, 0, 0
    for i in range(len(Y)):
        if Y[i][0]==1:
            if Ypred[i][0]==1:
                tp+=1
            else:
                fn+=1
        else:
            if Ypred[i][0]==0:
                tn+=1
            else:
                fp+=1
    
    return [[tp, fn], [fp, tn]]

    
def accuracy(c):
    n = sum([i for x in c for i in x])
    return (c[0][0] + c[1][1])/n

def precision(c):
    return c[0][0]/(c[0][0] + c[1][0])

def recall(c):
    return c[0][0]/(c[0][0] + c[0][1])

def f_measure(c):
    p = precision(c)
    r = recall(c)
    return 2*p*r/(p+r)
    
def TPR(c):
    return recall(c)

def FPR(c):
    return c[1][0]/(c[1][0]+c[1][1])

def logisticRegression(x, y):
    Y = np.reshape(y, (len(y), 1))

    Xvector = np.c_[np.ones((len(x), 1)), x]

    n = len(x)
    W = np.zeros((len(Xvector[0]), 1))
    
    for _ in range(10000000):
        gradient = 2/n*Xvector.transpose().dot(sigmoid(Xvector.dot(W)) - Y)
        if np.all((gradient == 0)):
            break
        
        W = W - ETA*gradient
    
    return (W, SSE(Xvector, Y, W))
    
    
    
def sigmoid(z):
    return 1/(1+np.exp(-z))


def predict(X, W, theta=0.5):
    Ypredicted = np.array([[1] if i[0]>=theta else [0] for i in sigmoid(X.dot(W))])
    return Ypredicted


def SSE(X, Y, W):
    error = Y - predict(X, W)
    return error.transpose().dot(error)[0][0]


df = pd.read_csv("data.csv")

cols = list(df.columns)
print(cols)

for i in cols:
    print(df[i].describe())


features = cols[2:-1]
output = "diagnosis"


label_encoder = preprocessing.LabelEncoder()
df[output]= label_encoder.fit_transform(df[output])


final_features = list(features)


p = df[features].corr().values.tolist()


for i in range(len(p)):
    for j in range(i+1, len(p)):
        if abs(p[i][j]) > 0.7 and features[i] in final_features and features[j] in final_features:
            final_features.remove(features[i])
print("\n\nFeatures after removing multicollinearity:\n", final_features, len(final_features))


d = len(final_features)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(df[final_features], df[output], test_size=0.2)

# print(logisticRegression(Xtrain, Ytrain.tolist()))


#obtained using gradient descent with eta =0.007 and iterations = 10000000
W = np.array([[-14.13967032],
           [  7.95389434],
           [ -1.09796794],
           [  5.13651019],
           [ -1.44968948],
           [ 14.10065082],
           [ -2.88448603],
           [  0.28892423],
           [ -5.05859146],
           [ 79.52821553],
           [  0.547714  ],
           [-46.33737769]])


Y = np.reshape(Ytest.tolist(), (len(Ytest), 1))
Xvector = np.c_[np.ones((len(Xtest), 1)), Xtest]

c = ConfusionMatrix(Xvector, Y, W)
print(c)
print("[ACCURACY]", accuracy(c))
print("[PRECISION]", precision(c))
print("[RECALL]", recall(c))
print("[F-MEASURE]", f_measure(c))
print("[TPR]", TPR(c))
print("[FPR]", FPR(c))


space = np.linspace(0, 1, 1000)
points= []
for i in space:
    points.append((FPR(ConfusionMatrix(Xvector, Y, W, i)), TPR(ConfusionMatrix(Xvector, Y, W, i))))
    
points.sort()
fpr = [x for (x, y) in points]
tpr = [y for (x, y) in points]

plt.plot(fpr, tpr)
plt.show()

auc = np.trapz(tpr, fpr)
print("[AUC]", auc)


