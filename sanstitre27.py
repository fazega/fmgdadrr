# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:17:27 2017

@author: Florent
"""

from pandas import read_csv
import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)


#Importe les données CSV en remplaçant les -1 par np.nan
def initialise_dataset(lien):
        
    dataset = read_csv(lien, header=None)
    
    header=[dataset[j][0] for j in range(58)]
    dataset_sansheader=[dataset[j][1:].astype(float) for j in range(58)]
    dataset_sansheader=np.array(dataset_sansheader)
    dataset_sansheader = pd.DataFrame(dataset_sansheader)
    dataset_sansheader=dataset_sansheader.replace(-1.0000,np.nan)
    return dataset_sansheader,header

dtst=initialise_dataset('test.csv')[0]
imputer = imputer.fit(dtst)
dtst=imputer.transform(dtst)
dtst=pd.DataFrame(dtst)

reseau_de_neurones(dtst)