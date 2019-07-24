import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
iris = load_iris()
x = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors = 5)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.20)

knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

cnf = metrics.confusion_matrix(y_test,y_pred)
accuracy = metrics.accuracy_score(y_test,y_pred)

print cnf
print accuracy



