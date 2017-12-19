# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:06:24 2017

@author: mmenoret
"""
import numpy as np
from sklearn import utils

fold_g = 'F:/IRM_Marche/'
label=np.loadtxt(fold_g+'label.txt','S12')

### Permutation Null Score
#nb_p=10000
#
#null_result=np.zeros(nb_p)
#for i in range(nb_p):
#    y_train_random=np.random.permutation(y_train)
#    pipeline_anova.fit(roi_train, y_train_random)
#    prediction = pipeline_anova.predict(roi_test) 
#    null_result[i]=accuracy_score(prediction,y_test)
#
#sign=(null_result>=result).sum()/nb_p
#print(sign)

### Permutation block
#nb_p=1000
#null_result=np.zeros(nb_p)
#for i in range(nb_p):
#    # shuffle block number
#    session=np.zeros(y_train.shape)
#    d=0
#    ses_bool=y_train==b'foot'
#    for x in range(ses_bool.size):
#        if ses_bool[x]!=ses_bool[x-1]:
#            d=d+1
#        session[x]=d
#        
#    x=np.unique(session)
#    xperm=np.random.permutation(x)
#    y_train_random=np.zeros(y_train.size,dtype='S12')
#    for z in range(x.size):
#        y_train_random[session==xperm[z]]=y_train[session==x[z]]
#        
#    pipeline_anova.fit(roi_train, y_train_random)
#    prediction = pipeline_anova.predict(roi_test) 
#    null_result[i]=accuracy_score(prediction,y_test)
#
#sign=(null_result>=result).sum()/nb_p
#print(sign)

## Permutation block
nb_p=1000
null_result=np.zeros(nb_p)

ncond=['hand','foot']
for i in range(nb_p):
    y_train_random=np.zeros((0,1),dtype='S12')
    # shuffle block number
    for suj in range(22):
        xncond=np.random.permutation(ncond)
        suj_train_random=np.append(np.full(57,xncond[0],dtype='S12'),np.full(57,xncond[1],dtype='S12'))
        y_train_random=np.append(y_train_random,suj_train_random)

        
    pipeline_anova.fit(roi_train, y_train_random)
    prediction = pipeline_anova.predict(roi_test) 
    null_result[i]=accuracy_score(prediction,y_test)

sign=(null_result>=result).sum()/nb_p
print(sign)