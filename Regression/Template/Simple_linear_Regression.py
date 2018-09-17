#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 07:12:07 2018

@author: apple
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading csv data file
DataSet = pd.read_csv('Salary_Data.csv')
X = DataSet.iloc[:, :-1].values
y = DataSet.iloc[:, 1].values

'''
#using mean to input mising value
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#encoding changing string lable of country and purchased to numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()
labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)
'''

#splitting data set into traing and test data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

'''
#Scaling the values into same range 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''

#Fitting the linear regression model th the training set
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
Regressor.fit(X_train, y_train)

#Predicting the test results
y_pred = Regressor.predict(X_test)

#visualize the training set result
def PlotTrain():
    plt.scatter(X_train, y_train, color = 'red')
    plt.plot(X_train, Regressor.predict(X_train))
    plt.xlabel("Years of experience")
    plt.ylabel("Salary")
    plt.title("Years of Experience V/s Salary(Training Set)")
    plt.show()
    
#visualize the test set result
def PlotTest():
    plt.scatter(X_test, y_test, color = 'red')
    plt.plot(X_train, Regressor.predict(X_train))
    plt.xlabel("Years of experience")
    plt.ylabel("Salary")
    plt.title("Years of Experience V/s Salary(Test Set)")
    plt.show()















