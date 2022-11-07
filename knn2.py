import pandas as pd
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

df = pd.read_csv('diabetes.csv')
zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']

for i in zero:
    df[i] = df[i].replace(0,np.NaN)
    mean = int(df[i].mean(skipna=True))
    df[i] = df[i].replace(np.NaN,mean)
df
cols= df.columns.tolist()
ip = cols[0:8]
op = cols[8]

X = df.iloc[:,0:8]

Y = df[op].to_numpy()
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

df.corr()

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    
        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority voye
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
clf = KNN(k=11)
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)

print(predictions)

acc = np.sum(predictions == Y_test) / len(Y_test)
print(acc)

cm = confusion_matrix(Y_test,predictions)
print(cm)
acc = accuracy_score(Y_test,predictions)
print(acc)
f1_score = f1_score(Y_test,predictions)
print(f1_score)

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Y_test, predictions)

plt.subplots(1, figsize=(10,10))
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
 