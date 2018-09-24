import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from datetime import datetime
import time
fmt = '%H:%M:%S'


def get_current_time():
    time.ctime()
    return time.strftime(fmt)


r = range(42, 753)

trainX = np.genfromtxt(fname='trainData.csv', delimiter=',', dtype=int, skip_header=1, usecols=r)
trainY = np.genfromtxt(fname='trainLabels.csv', dtype=int, skip_header=1)
# testX = np.genfromtxt(fname='testData.csv', delimiter=',', dtype=int, skip_header=1, usecols=r)

testX = np.genfromtxt(fname='kaggleTestSubset.csv', delimiter=',', dtype=int, skip_header=1, usecols=r)
testY = np.genfromtxt(fname='kaggleTestSubsetLabels.csv', dtype=int, skip_header=1)

trainX = preprocessing.normalize(trainX, axis=1, copy=True, return_norm=False)
testX = preprocessing.normalize(testX, axis=1, copy=True, return_norm=False)

# trainX = preprocessing.scale(trainX, axis=1, with_mean=False)
# testX = preprocessing.scale(testX, axis=1, with_mean=False)

first = get_current_time()

print('Fitting Data')
classifier = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='auto', p=3,
                                  metric='minkowski', n_jobs=-1)
classifier.fit(trainX, trainY)

second = get_current_time()
print("Time taken to train(sec):", datetime.strptime(second, fmt) - datetime.strptime(first, fmt))

print('Predicting Data')
testpredy = classifier.predict(testX)

third = get_current_time()
print("Time taken to predict(sec):", datetime.strptime(third, fmt) - datetime.strptime(second, fmt))

np.savetxt('result.csv', np.dstack((np.arange(1, testpredy.size+1), testpredy))[0], "%d,%d", header="ID,Label",
           comments='')

print("Accuracy is", accuracy_score(testY, testpredy)*100)
