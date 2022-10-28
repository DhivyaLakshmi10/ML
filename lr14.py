

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split

#import the csv
df = pd.read_csv("wcb.csv")
df.head()

#Cleaning the dataset + preprocessing
l = LabelEncoder()
l.fit(df['diagnosis'])
df['diagnosis'] = l.transform(df['diagnosis'])

X = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
Y = df['diagnosis'].values
Xnames = X.columns
#X is normalized
# X = pd.DataFrame(normalize(X.values), columns = Xnames)


#Reducing multicollinearity
final_features = [x for x in Xnames]
p = df[Xnames].corr().values.tolist()
for i in range(len(p)):
    for j in range(i+1, len(p)):
        if abs(p[i][j]) > 0.7 and Xnames[i] in final_features:
            final_features.remove(Xnames[i])
print("\n\nFeatures before removing multicollinearity: ", Xnames)
print("\n\nFeatures after removing multicollinearity:\n", final_features)

# Outlier Treatment
def outlier_treatment(df, feature):
    q1, q3 = np.percentile(df[feature], [25, 75])
    IQR = q3 - q1 
    lower_range = q1 - (3 * IQR) 
    upper_range = q3 + (3 * IQR)
    to_drop = df[(df[feature]<lower_range)|(df[feature]>upper_range)]
    df.drop(to_drop.index, inplace=True)

outlier_treatment(df, 'diagnosis')

#Splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

def sigmoid(Z):
    Z = np.array(Z, dtype='float64')
    return 1/(1 + np.exp(-Z))

def logisticRegression(X, Y, thold, learningRate, iterations):
    X = np.vstack((np.ones((X.shape[0],)), X.T)).T
    wT=np.zeros((X.shape[1], 1)).T
    costs = []
    for i in range(iterations):
        wTx = np.dot(wT, X.T )
        A = sigmoid(wTx)
        wPred = np.array([1 if x >= thold else 0 for x in A[0]])
        costs.append(np.sum(np.square(wPred - Y)))
        dW = np.dot(X.T, (wPred - Y)) / (Y.size) 
#         dW = 
        wT = wT - learningRate * dW
    return wT, np.array(costs)


W, costs = logisticRegression(X_train, Y_train, 0.5, 0.00001,100000)
W

#Plotting the cost function
plt.plot(costs)

def predict(X, Y, W, tHold):
    X = np.vstack((np.ones((X.shape[0],)), X.T)).T
    wTx = np.dot(W, X.T)
    A = sigmoid(wTx)
    wPred = np.array([1 if x >= tHold else 0 for x in A[0]])
#     print("Accuracy: ", np.sum(wPred == Y)/Y.size)    
    return wPred

dP = predict(X_test, Y_test, W, 0.5)
def parseDiagnosis(x):
    return np.array(["Malignant" if i == 0 else "Benign" for i in x])
parseDiagnosis(dP)


def analyseMeasures(dP, Y_test):
    trueOrFalse = []
    for i in zip(dP,dP == Y_test):
        trueOrFalse.append(i)
    trueOrFalse = np.array(trueOrFalse)
    TP = FP = TN = FN = 0
    for i,j in trueOrFalse:
        if i == 1 and j == 1:
            TP+=1
        elif i == 0 and j == 1:
            TN+=1
        elif i == 1 and j == 0:
            FN+=1
        elif i == 0 and j == 0:
            FP+=1
        else:
            print(f"{i},{j} <-- idk what to do with this")
    confusionArr = [TP, FP, TN, FN]
    measures = {}
    measures["Accuracy"] = (TP + TN)/(TP + TN + FP + FN) 
    measures["Precision"] = (TP/(TP + FP), TN/(TN + FN))
    measures["Recall"] = (TP/(TP+FN), TN/(TN+FP))
    measures["F-Measure"] = (2*(TP/(TP+FP))*(TP/(TP+FN))/((TP/(TP+FN))+(TP/(TP+FP))), 2*(TN/(TN+FN))*(TN/(TN+FP))/((TN/(TN+FP))+(TN/(TN+FN))))
    measures["TPR"] = TP/(TP+FN)
    measures["FPR"] = FP/(TN+FP)
    measures["TNR"] = TN/(TN+FP)
    measures["FNP"] = FN/(TP+FN)
    return measures, confusionArr

measures, confusionArr = analyseMeasures(dP, Y_test)
print(measures, confusionArr)

#print(FP+FN) # No. of miscalculations

predicts = []
for i in range(0, 11, 1):
    try:
        predicts.append(analyseMeasures(predict(X_test, Y_test, W, i/10), predict(X_test, Y_test, W, i/10) == Y_test))
    except ZeroDivisionError:
            pass

