import numpy as np
import matplotlib as plt
import csv
from sklearn import preprocessing, cross_validation, neighbors,svm
import pandas as pd
from sklearn.cross_validation import train_test_split
import random
df= pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999, inplace=True)

df.drop(df.columns[0], axis=1,inplace=True)

X = np.array(df.drop(df.columns[9], axis=1))
y = np.array(df.iloc[:, 9])
print(X)
print(y)	
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = svm.SVC()

clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)

example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
if(prediction==[2]):
	print("malignant")
else:
	print("benign")

