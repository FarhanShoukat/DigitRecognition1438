# DigitRecognition1438

## Abstract:
In this project, I developed three methods to classify digits 1, 3, 4 and 8. The classification is done using kNN, Decision Tree and Convolutional Neural Network (CNN). First, image data is either normalized or scaled. Then it is fed to classifier.

## Methadology:
First, training images (pixel values) and their labels are read. Then, classifier is trained using training data and labels. Finally, test data is used to evaluate training.

### 1) Data Set Selection:
Data Set used here is a subset of MNIST data set which include images of 0-9 digits. In this project, I only used 1, 3, 4 and 8 for digit recognition. This dataset's main purpose is to compare different approaches of image classification. The code can be run on any dataset.

### 2) Feature Selection:
it was found that many of the starting and ending bits fo a number are zero. So, those bits were removed. It helped decrease feature length. It reduced predicting time for kNN by 10-15%.

### 3) Data Pre-processing:
After reading data and removing features in feature selection, data was preprocessed. In this case, normalization and scaling were tested.

### 4) Machine Learning Algorithm:
As I am using supervised learning approach to classify, I used kNN, Decision Tree and Convolutional Neural Network (CNN).
