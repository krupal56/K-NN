import pandas as pd
import numpy as np
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

##Error rate
error = []

for i in range(1,26):
	knn = KNeighborsClassifier(n_neighbors=i)
	knn.fit(x_train,y_train)
	pred_i = knn.predict(x_test)
	error.append(np.mean(pred_i !=y_test))

import matplotlib.pyplot as plt
plt.plot(range(1,26),error,color = 'red')
plt.xlabel('K value')
plt.ylabel('Mean Error')
plt.show()





