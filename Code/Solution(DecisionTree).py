import numpy as np
from datetime import datetime
import time
from sklearn import tree
from sklearn.preprocessing import normalize, scale
from sklearn.metrics import accuracy_score
fmt = '%H:%M:%S'


def get_current_time():
    time.ctime()
    return time.strftime(fmt)


r = range(42, 753)

trainX = np.genfromtxt(fname='trainData.csv', delimiter=',', dtype=int, skip_header=1, usecols=r)
trainY = np.genfromtxt(fname='trainLabels.csv', dtype=int, skip_header=1)
# test = np.genfromtxt(fname='testData.csv', delimiter=',', dtype=int, skip_header=1, usecols=r)

testX = np.genfromtxt(fname='kaggleTestSubset.csv', delimiter=',', dtype=int, skip_header=1, usecols=r)
testY = np.genfromtxt(fname='kaggleTestSubsetLabels.csv', dtype=int, skip_header=1)

print('Data loaded')

trainX = normalize(trainX, axis=1, copy=True, return_norm=False)
testX = normalize(testX, axis=1, copy=True, return_norm=False)

# trainX = scale(trainX, axis=1, with_mean=False)
# testX = scale(testX, axis=1, with_mean=False)

first = get_current_time()

classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
classifier.fit(trainX, trainY)

second = get_current_time()
print("Time taken to train(sec):", datetime.strptime(second, fmt) - datetime.strptime(first, fmt))

testpredy = classifier.predict(testX)

third = get_current_time()
print("Time taken to predict(sec):", datetime.strptime(third, fmt) - datetime.strptime(second, fmt))

# np.savetxt('result_tree.csv', np.dstack((np.arange(1, testpredy.size+1), testpredy))[0], "%d,%d", header="ID,Label",
#            comments='')

print("Accuracy is", accuracy_score(testY, testpredy)*100)
