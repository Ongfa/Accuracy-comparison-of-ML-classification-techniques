
import csv
import random
import math

def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def trainTestSplit(dataset, splitRatio):
	trainLength = int(len(dataset) * splitRatio)
	trainData = []
	copy = list(dataset)
	while len(trainData) < trainLength:
		index = random.randrange(len(copy))
		trainData.append(copy.pop(index))
	return [trainData, copy]

def separateByClass(dataset):
	seperate = {}
	for i in range(len(dataset)):
		vec = dataset[i]
		if (vec[-1] not in seperate):
			seperate[vec[-1]] = []
		seperate[vec[-1]].append(vec)
	return seperate

def mean(values):
	return sum(values)/float(len(values))

def stdev(values):
	avg = mean(values)
	variance = sum([pow(x-avg,2) for x in values])/float(len(values)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	seperate = separateByClass(dataset)
	#print seperate
	summaries = {}
	for classValue, instances in seperate.iteritems(): 
		summaries[classValue] = summarize(instances)
	return summaries

def findProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def findClassProbabilities(summaries, inputvec):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputvec[i]
			probabilities[classValue] *= findProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputvec):
	probabilities = findClassProbabilities(summaries, inputvec)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testData):
	predictions = []
	for i in range(len(testData)):
		result = predict(summaries, testData[i])
		predictions.append(result)
	return predictions

def getAccuracy(testData, predictions):
	correct = 0
	for i in range(len(testData)):
		if testData[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testData))) * 100.0

def main():
	       #filename = 'pima-indians-diabetes.data.csv'
	       i=0.67
	       filename='data/cancer.data.csv'
	       splitRatio = i
	       dataset = loadCsv(filename)
	       trainingSet, testData = trainTestSplit(dataset, splitRatio)
	       summaries = summarizeByClass(trainingSet)
               predictions = getPredictions(summaries, testData)
	       accuracy = getAccuracy(testData, predictions)
	       print('Accuracy: {0}%').format(accuracy)

main()