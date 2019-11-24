from scipy.io import loadmat;
import os;
from sklearn.linear_model import LogisticRegression;
import statistics;
import math;
import random;
import numpy as np;
import matplotlib.pyplot as plt;

""" GLOBALS """

k = 10;	 # k is the number of samples in each batch selected from unlabeled data 
N = 50;  # N is the number of iterations in the algorithm
numberOfMultipleRuns = 3;  # Number of multiple runs to find average


""" CLASSES """

class Data():
	"""Data class to store training, testing and unlabeled data for two datasets"""
	def __init__(self, typeOfDataset, typeOfSampling):
		self.datasetName = typeOfDataset;
		self.samplingType = typeOfSampling;
		self.data = dict();

	# load the data
	def loadData(self):
		labels = ["testing", "training", "unlabeled"];
		typesOfDS = ["Matrix", "Labels"];
		dataDirectory = "Data";
		parentDir = os.path.dirname(os.getcwd());
		datasetDir = os.path.join(parentDir, dataDirectory, self.datasetName);
		
		for label in labels:
			if label not in self.data:
				self.data[label] = dict();
			for typeOfDS in typesOfDS:
				if typeOfDS not in self.data[label]:
					self.data[label][typeOfDS] = [];

				for x in range(numberOfMultipleRuns):
					if self.datasetName == "MindReading":
						loaded = loadmat(os.path.join(datasetDir, label+typeOfDS+'_'+self.datasetName+str(x+1)+'.mat'));
					else:
						loaded = loadmat(os.path.join(datasetDir, label+typeOfDS+'_'+str(x+1)+'.mat'));

					listt = loaded[label+typeOfDS];

					if typeOfDS == 'Labels':
						listt = [item for sublist in listt for item in sublist];

					self.data[label][typeOfDS].append(listt);


class Results():
	"""Results class to store the accuracy results for two datasets"""
	def __init__(self, typeOfDataset, typeOfSampling):
		super(Results, self).__init__()
		self.datasetName = typeOfDataset;
		self.accuracies = dict();
		self.samplingType = typeOfSampling;
		self.averageAccuracies = [];
		
	# store the given accuracy for a specific runNumber
	def storeAccuracy(self, accuracy, runNumber):
		if runNumber not in self.accuracies:
			self.accuracies[runNumber] = [];
		self.accuracies[runNumber].append(accuracy);

	# calculate average of accuracies for different runs of algorithm
	def calculateAvgAccuracy(self):
		for x in range(N):
			self.averageAccuracies.append(statistics.mean([self.accuracies[0][x], 
				self.accuracies[1][x], self.accuracies[2][x]]));


""" FUNCTIONS """

# active learning algorithm for multiple runs to calculate mean
def activeLearningMultiple(Data, Results):

	for runNumber in range(numberOfMultipleRuns):
		activeLearning(Data, Results, runNumber);

	Results.calculateAvgAccuracy();


# active learning algorithm for a single run
def activeLearning(Data, Results, runNumber):
	
	for numIter in range(N):

		# get logistic regression model
		lrModel = LogisticRegression(C=0.01, solver='newton-cg', max_iter=200, multi_class='multinomial');

		# train the model on given training data
		lrTrainedModel = lrModel.fit(Data.data['training']['Matrix'][runNumber], Data.data['training']['Labels'][runNumber]);
		
		# get accuracy of the model on testing data
		accuracy = lrTrainedModel.score(Data.data['testing']['Matrix'][runNumber], Data.data['testing']['Labels'][runNumber]);

		# store the results
		Results.storeAccuracy(accuracy, runNumber);

		# calculate the probabilities of all unlabeled samples being in each class
		probsUnlabeledData = lrTrainedModel.predict_proba(Data.data['unlabeled']['Matrix'][runNumber]);

		# check the type of sampling
		if Data.samplingType == 'random':
			kValuesSelected = randomSampling(probsUnlabeledData);
		else:	
			kValuesSelected = uncertaintyBasedSampling(probsUnlabeledData);

		kValuesSelected = sorted(kValuesSelected, reverse=True);

		for x in kValuesSelected:
			# adjust samples -> delete some samples from unlabeled data, add to training data
			adjustSamples(Data, x, runNumber);
	

# adjust samples
def adjustSamples(Data, sampleNum, runNumber):

	toDelete = Data.data['unlabeled']['Matrix'][runNumber][sampleNum];
	
	# add some samples from unlabeled data to training data
	Data.data['training']['Matrix'][runNumber] = np.append(Data.data['training']['Matrix'][runNumber], [toDelete], axis=0);
	
	# delete those samples from unlabeled data
	Data.data['unlabeled']['Matrix'][runNumber] = np.delete(Data.data['unlabeled']['Matrix'][runNumber], sampleNum, axis=0);

	toDelete = Data.data['unlabeled']['Labels'][runNumber][sampleNum];
	
	# add corresponding labels for samples added from unlabeled data to training data
	Data.data['training']['Labels'][runNumber] = np.append(Data.data['training']['Labels'][runNumber], [toDelete]);
	
	# delete those labels from unlabeled data
	Data.data['unlabeled']['Labels'][runNumber] = np.delete(Data.data['unlabeled']['Labels'][runNumber], sampleNum);


# random sampling of 'k' samples from unlabeled data
def randomSampling(probabilityData):

	numSamples = len(probabilityData)
	return [random.randint(0, numSamples-1) for x in range(k)];


# uncertainty based sampling of 'k' samples from unlabeled data 
def uncertaintyBasedSampling(probabilityData):
	
	d = dict();
	numClasses = len(probabilityData[0]);
	
	# calculate classification entropy for each sample
	for x in range(len(probabilityData)):
		sum = 0;
		for y in range(numClasses):
			sum += probabilityData[x][y]*(math.log(probabilityData[x][y], 2));
		sum *= -1;
		d[sum] = x;
	
	x = 0;
	topKValues = [];
	
	# select 'k' samples with highest entropy
	for key in sorted(d, reverse=True):
		topKValues.append(d[key]);
		x += 1;
		if x == k:
			return topKValues;

# draw a graph for random and uncertainty based sampling for given dataset
def drawGraph(resultsRandom, resultsUncertaintyBased):
	
	iterations = range(1, N+1);
	
	# random sampling 
	plt.plot(iterations, resultsRandom.averageAccuracies, label='accuracy for random sampling');
	
	# uncertainty based sampling
	plt.plot(iterations, resultsUncertaintyBased.averageAccuracies, label='accuracy for uncertainty based sampling');
	
	# setting plot data and showing it
	plt.xlabel('Iterations');
	plt.ylabel('Accuracy');
	plt.title('Accuracy on ' + resultsUncertaintyBased.datasetName + ' dataset');
	plt.legend();
	plt.show();


def main():

	# Load data for MindReading dataset to run for random sampling
	mindReadingDataRandom = Data("MindReading", "random");
	mindReadingDataRandom.loadData();

	# Create results object for MindReading data for random sampling
	mindReadingResultsRandom = Results("MindReading", "random");
	
	# Apply active learning algorithm for MindReading data for random sampling
	activeLearningMultiple(mindReadingDataRandom, mindReadingResultsRandom);

	
	# Load data for MindReading dataset to run for uncertainty based sampling
	mindReadingDataUncertaintyBased = Data("MindReading", "uncertaintyBased");
	mindReadingDataUncertaintyBased.loadData();

	# Create results object for MindReading data for uncertainty based sampling
	mindReadingResultsUncertaintyBased = Results("MindReading", "uncertaintyBased");
	
	# Apply active learning algorithm for MindReading data for uncertainty based sampling
	activeLearningMultiple(mindReadingDataUncertaintyBased, mindReadingResultsUncertaintyBased);

	
	# Draw graph for MindReading data
	drawGraph(mindReadingResultsRandom, mindReadingResultsUncertaintyBased);
	
	# ---------------------------------------------------------------------------------------------

	
	# Load data for MMI dataset to run for random sampling
	MMIDataRandom = Data("MMI", "random");
	MMIDataRandom.loadData();

	# Create results object for MMI data for random sampling
	MMIResultsRandom = Results("MMI", "random");

	# Apply active learning algorithm for MMI data for random sampling
	activeLearningMultiple(MMIDataRandom, MMIResultsRandom);
	

	# Load data for MMI dataset to run for uncertainty based sampling
	MMIDataUncertaintyBased = Data("MMI", "uncertaintyBased");
	MMIDataUncertaintyBased.loadData();

	# Create results object for MMI data for uncertainty based sampling
	MMIResultsUncertaintyBased = Results("MMI", "uncertaintyBased");

	# Apply active learning algorithm for MMI data for uncertainty based sampling
	activeLearningMultiple(MMIDataUncertaintyBased, MMIResultsUncertaintyBased);
	

	# Draw graph for MMI data
	drawGraph(MMIResultsRandom, MMIResultsUncertaintyBased);
	

if __name__ == '__main__':
	main();