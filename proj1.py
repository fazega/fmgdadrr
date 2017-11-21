# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import csv
import numpy as np

test=[]
with open('test.csv', newline='') as csvfile:
   spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
   for row in spamreader:
       test.append(', '.join(row))

res=[]

for l in test:
    res.append(l.split(","))
    
print(np.asmatrix(res))
    