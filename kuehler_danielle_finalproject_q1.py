#Danielle Kuehler
#ITP 449 Summer 2020
#Final Project
#Q1

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

#Question One- Wine Quality classification using KNN

#A.	Load the data from the file winequality.csv
data = pd.read_csv("winequality.csv") #Read in csv file
pd.set_option("display.max_columns", None) #Display all columns

#B. Standardize all variables other than Quality
X = data.iloc[:,0:11] #Feature matrix
y = data.iloc[:,11] #Target vector

scaler = StandardScaler() #Standardize function from sklearn
scaler.fit(X) #Fit to feature matrix- computes mean and std dev for later scaling
X_scaled = pd.DataFrame(scaler.transform(X), columns = X.columns) #Standardize with fitting and scaling, then convert fitted data X into dataframe

#C. Partition the dataset into train and test sets
X_train, X_test, y_train, y_test = \
    train_test_split(X_scaled, y, test_size = 0.3, random_state = 2019, stratify = y)

#D. Build a KNN classification model to predict Quality based on all the remaining numeric variables
neighbors = np.arange(1,11) #For graphing
train_accuracy = np.empty(10)
test_accuracy = np.empty(10)

#E. Iterate on K ranging from 1 to 10. Plot the accuracy for the train and test datasets.
#Iterate for number of neighbors
for k in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=k) #Neighbors in range 1-10
    knn.fit(X_train, y_train) #Fit knn to training variables
    y_pred = knn.predict(X_test) #Predict
    cf = metrics.confusion_matrix(y_test, y_pred) #Confusion matrix
    train_accuracy[k-1] = knn.score(X_train, y_train) #Store accuracy of training variables in list
    test_accuracy[k-1] = knn.score(X_test, y_test) #Store accuracy of testing variables in list
#Plot the accuracy
plt.figure() #Create figure
plt.title("E. KNN: Varying Number of Neighbors\nKuehler_Danielle_FinalProject_Q1") #Title
plt.plot(neighbors, test_accuracy, label = "Testing Accuracy") #Plot testing accuracy
plt.plot(neighbors, train_accuracy, label = "Training Accuracy") #Plot traning accuracy
plt.legend() #Display legend of each line color
#Formatting
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show() #Display

#F.	Which value of k produced the best combined accuracy in the train and test data set?
print("F. Best value of k is 8")

#G.	Generate predictions for the test partition with the chosen value of k. Print the crosstab of the actual vs predicted wine quality
X_train, X_test, y_train, y_test = \
    train_test_split(X_scaled, y, test_size = 0.3, random_state = 2019, stratify = y)
knn = KNeighborsClassifier(n_neighbors = 8)  #K = 8
knn.fit(X_train, y_train) #Fit data
y_pred = knn.predict(X_test) #Predict values
y_pred2 = knn.predict(X_train)

#Cross tab
print("G. Crosstab of test partition- actual versus predicted wine quality:\n",
      pd.crosstab(y_test, y_pred, rownames=["Quality"], colnames=["PredictedQuality"]))

#Insert actual and predicted values into dataframe
X_test.insert(11, "Quality", y_test, True)
X_test.insert(12, "PredictedQuality", y_pred, True)

#H. Print the test dataframe with the added columns “Quality” and “Predicted Quality”
print("\nH. WineQuality.csv with actual and predicted values: \n", X_test)


