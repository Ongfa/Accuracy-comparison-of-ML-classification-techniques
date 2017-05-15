import csv
import random
import math
import operator
import numpy as np
#from sklearn import preprocessing, cross_validation, neighbors, svm
#import pandas as pd
#from sklearn.cluster import KMeans

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, newline='') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		#print instance1[x] - instance2[x]
		a=instance1[x]
		b=instance2[x]
		a=int(a)
		b=int(b)
		distance += math.pow((a - b), 2)
		print (distance)
		print (math.sqrt(distance))
	return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
	
def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.67
	loadDataset('breast-cancer-wisconsin.data', split, trainingSet, testSet)

	print ('Train set: ' + repr(len(trainingSet)))
	print ('Test set: ' + repr(len(testSet)))
	# generate predictions
	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	
main()
"""
	df = pd.read_csv('data/breast-cancer-wisconsin.data.txt')
	df.replace('?',-99999, inplace=True)
	df.drop(['id'], 1, inplace=True)

	X = np.array(df.drop(['class'], 1))
	y = np.array(df['class'])
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
"""

"""





clf_knn = neighbors.KNeighborsClassifier()
clf_knn.fit(X_train, y_train)
accuracy = clf_knn.score(X_test, y_test)
print(accuracy)

# Example of kNN implemented from Scratch in Python

"""