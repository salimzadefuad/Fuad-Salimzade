#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 18:10:05 2020

@author: fuadsalimzade
"""

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("loan-2-notcurrent.csv")

X = dataset.iloc[:,0:89].values
y = dataset.iloc[:, -1].values

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values= np.nan, strategy='constant', 
                      fill_value = 0)
X=imputer.fit_transform(X)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#Credit grade transformation
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[4])],
                                       remainder = 'passthrough',
                                      n_jobs = -1)
X = np.array(ct.fit_transform(X))

X = X[:,1:]

#House ownership transformation
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[11])],
                                       remainder = 'passthrough',
                                      n_jobs = -1)
X = np.array(ct.fit_transform(X))

X = X[:,1:]

#Purpose transformation
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[16])],
                                       remainder = 'passthrough',
                                      n_jobs = -1)
X = np.array(ct.fit_transform(X))

X = X[:,1:]

#State transformation
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[21])],
                                       remainder = 'passthrough',
                                      n_jobs = -1)
X = np.array(ct.fit_transform(X))

X = X[:,1:]

#Date transformation
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[69])],
                                       remainder = 'passthrough',
                                      n_jobs = -1)
X = np.array(ct.fit_transform(X))

X = X[:,1:]

#transforming y to categorical value
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

print(np.bincount(y))

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y.reshape(-1, 1)).toarray()


#Scalling the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, 
                                                    test_size = 0.20)



from sklearn.linear_model import LogisticRegression
LogiR = LogisticRegression(C = 100, random_state = 0, n_jobs = -1)
LogiR.fit(X_train, y_train)
LogiR.score(X_train, y_train)
LogiR.score(X_test, y_test)

from sklearn.model_selection import cross_val_score
cvs =cross_val_score(estimator = LogiR, X= X_train,y = y_train ,cv = 5)
print(cvs.mean())


from sklearn.model_selection import GridSearchCV
c_param = {'C': [0.01, 0.1, 1, 10]}
GS = GridSearchCV(LogiR,param_grid= c_param)
GS.fit(X_test, y_test)
GS.score(X_train, y_train)
GS.score(X_test, y_test)
best_p=GS.best_params_

"*********************Implementing Neural Networks""*********************"""

import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#implementing the model

classifier = Sequential()

#Adding input and first hidden layer

classifier.add(Dense(units = 117, activation = 'relu', input_dim = 173,
                     kernel_initializer = 'glorot_uniform'))

#Adding hidden layer
classifier.add(Dense(units = 117, activation = 'relu', 
                     kernel_initializer = 'glorot_uniform'))

classifier.add(Dense(units = 117, activation = 'relu', 
                     kernel_initializer = 'glorot_uniform'))

#Adding other hidden layer
classifier.add(Dense(units=3, activation='softmax'))


#Compiler
classifier.compile(optimizer='adam', loss='categorical_crossentropy', 
                   metrics=['accuracy'])

#fitting the model

classifier.fit(X_train, y_train, batch_size=64, epochs=100)

classifier.summary()
 

#predicting y on X_test
y_pred=classifier.predict(X_test)


#Creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.argmax(axis = 1), y_pred.argmax(axis = 1))


"*********************Predicting new dataset on the model""****************"""
                  
                  
current_data = pd.read_csv("loan-2-onlycurrent.csv")

X_new = current_data.values

print(X_new[0])

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values= np.nan, strategy='constant', 
                      fill_value = 0)
X_new=imputer.fit_transform(X_new)

print(X_new[0])


#Credit grade transformation
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[4])],
                                       remainder = 'passthrough',
                                      n_jobs = -1)
X_new = np.array(ct.fit_transform(X_new))

X_new = X_new[:,1:]

print(X_new[0])


#House ownership transformation
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[11])],
                                       remainder = 'passthrough',
                                      n_jobs = -1)
X_new = np.array(ct.fit_transform(X_new))

X = X[:,1:]

print(X_new[0])

#Purpose transformation
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[16])],remainder = 'passthrough',
                                      n_jobs = -1)
X_new = np.array(ct.fit_transform(X_new))

X_new = X_new[:,1:]

print(X_new[0])

#State transformation
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[21])],remainder = 'passthrough',
                                      n_jobs = -1)
X_new = np.array(ct.fit_transform(X_new))

X_new = X_new[:,1:]

print(X_new[0])

#Date transformation
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[69])],remainder = 'passthrough',
                                      n_jobs = -1)
X_new = np.array(ct.fit_transform(X_new))

X_new = X_new[:,1:]

print(X_new[0])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_new = sc.fit_transform(X_new)

#Predicted result
y_new=classifier.predict(X_new)


A = y_new[:, 0] 
B = y_new[:, 1]
C = y_new[:, 2]

D = []
for x in range(602244):
    if A[x] > 0.5:
        A[x] = 1
    else:
        A[x] = 0
    D.append(A[x])
    
for x in range(602244):
    if B[x] > 0.5:
        B[x] = 1
    else:
        B[x] = 0
    D.append(A[x])
    
for x in range(602244):
    if C[x] > 0.5:
        C[x] = 1
    else:
        C[x] = 0
    D.append(A[x])
     
    
y_new = np.array(list(zip(A,B,C)))
y_new = pd.DataFrame(y_new)

y_transformed = pd.DataFrame()

for y in range(602244):
    for x in range(3):
        if (y_new[y][x] == 1 and x == 0):
            y_transformed.insert(loc = 0, column = "Fully Paid", value = "Yes")
                
        elif (y_new[y][x] == 1 and x == 1):
            y_transformed.insert(loc = 1, column = "Default" , value = "Yes")
            
        elif (y_new[y][x] == 1 and x == 2):
            y_transformed.insert(loc = 2, column = "Late" , value = "Yes")
            
        
#Predicting y_new
y_new = classifier.predict(X_new)

y_new = pd.DataFrame(y_new)

y_transformed = pd.DataFrame()

y_transformed = y_transformed.append(y_new)

y_transformed.columns = ["Fully Paid", "Default", "Late"]

y_transformed.to_csv("Predicted_results")

            