# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:17:27 2017

@author: Florent
"""

from pandas import read_csv
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPRegressor

import time


imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)


#
#Importe les données CSV en remplaçant les -1 par np.nan
def initialise_dataset(lien, n):

    dataset = read_csv(lien, header=None)

    header=[dataset[j][0] for j in range(n)]
    dataset_sansheader=[dataset[j][1:].astype(float) for j in range(n)]
    dataset_sansheader=np.array(dataset_sansheader)
    dataset_sansheader = pd.DataFrame(dataset_sansheader)
    dataset_sansheader=dataset_sansheader.replace(-1.0000,np.nan)
    return dataset_sansheader,header

def analyse(test,train):
    #K nearest neighbours
    target=train[:,1]
    test_epure=train[:,2:]
    for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = KNeighborsClassifier(10, weights=weights)
        clf.fit(test_epure, target)
    Z = clf.predict(test[:,1:])
    print(Z)

def analyse_MLP(test, train):
    t = time.time()
    entries = train[:, 2:]
    results = train[:,1]
    print(entries[0])
    print(len(entries))
    print(sum(results))
    print(len(results))
    clf = MLPRegressor(solver='lbfgs', alpha=1e-5,activation='logistic', hidden_layer_sizes=(10,10,10))
    clf.fit(entries, results)

    Z = clf.predict(test[:,1:])
    print(sum(Z))
    print(str(time.time()-t)+" seconds to achieve MLP.")
    return [[int(test[i,0]), Z[i]] for i in range(len(Z))]


def export_csv(result):
    np.savetxt('result.csv', result, fmt='%i,%f', header="id,target")

print("Loading data ...")

#Mise en forme des données et imputation
dtst_test=initialise_dataset('test.csv', 58)[0]
imputer = imputer.fit(dtst_test)
dtst_test=np.transpose(imputer.transform(dtst_test))
print("Test loaded.")

#
dtst_train=initialise_dataset('train.csv', 59)[0]
imputer = imputer.fit(dtst_train)
dtst_train=np.transpose(imputer.transform(dtst_train))
print("Train loaded.")

print("Analyzing with MLP ...")
result = analyse_MLP((dtst_test),(dtst_train))
print("Exporting result in csv ...")
export_csv(result)
print("Result exported.")
