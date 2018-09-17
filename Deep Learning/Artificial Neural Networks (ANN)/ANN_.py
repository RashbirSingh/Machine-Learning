#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 04:34:47 2018

@author: apple
"""
#part 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading csv data file
DataSet = pd.read_csv('Churn_Modelling.csv')
X = DataSet.iloc[:, 3:13].values
y = DataSet.iloc[:, 13].values


#encoding changing string lable of country and purchased to numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X_1 = LabelEncoder()
X[:, 1] = labelEncoder_X_1.fit_transform(X[:, 1])
labelEncoder_X_2 = LabelEncoder()
X[:, 2] = labelEncoder_X_2.fit_transform(X[:, 2])
oneHotEncoder = OneHotEncoder(categorical_features = [1])
X = oneHotEncoder.fit_transform(X).toarray()
X = X[:, 1:]


#splitting data set into traing and test data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Scaling the values into same range 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#part 2

#importing keras library
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialising the ANN
classifier = Sequential()

#adding the first layer and first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

#addind 2nd hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

#adding output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#compiling the ANN
classifier.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ['accuracy'])

#firring the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 5, epochs = 500)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)