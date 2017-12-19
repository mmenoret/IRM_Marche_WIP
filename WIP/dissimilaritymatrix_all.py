# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:17:51 2017

@author: mmenoret
"""

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt

# Behavioral DATA
fold_g = 'F:/IRM_Marche/'
smt='ss'       
names='ap','as','boh','bh','bi','cmp','cas','cs','cb','gm','gn','gbn','mv','ms','pm','pc','ph','pa','pv','pom','rdc','ti','vs'
label=np.loadtxt(fold_g+'label_main.txt','S12')
block=np.loadtxt(fold_g+'block_main.txt','int')
motor_region=np.fromfile('F:/IRM_Marche/masquesROI/reg_whole70_basc444asym.np','int')

# Remove data not analysed
mask_block=block==block
for x in range(label.shape[0]):
    if label[x,2]!=label[x-1,2]:
        mask_block[x]=False
    elif label[x,2]!=label[x-2,2]:
        mask_block[x]=False
c_des_out=np.logical_not(label[:,2]== b'des')
tmp_out= np.logical_and(c_des_out,mask_block)
c_rest_out=np.logical_not(label[:,0]== b'rest')
cond_out= np.logical_and(tmp_out,c_rest_out)
y=label[cond_out,2]
labels=np.unique(y)
# Prepare correlation
estimator = LedoitWolf()
scaler=StandardScaler()
# Create np array
result_matrix = np.empty([len(names),motor_region.shape[0],labels.shape[0],labels.shape[0]])

#Analysis for each subject
for i,n in enumerate(sorted(names)):
    roi_name=fold_g+'mni4060/asymroi_'+smt+'_'+n+'.npz'   
    roi=np.load(roi_name)['roi'][cond_out]
    roi=roi[:,motor_region-1] 
    for j in range(motor_region.shape[0]):
        roi_j=roi[:,j]
        roi_mat=np.zeros(((y==b'imp').sum(),len(labels)))
        for z,lab in enumerate(sorted(labels)):
            roi_mat[:,z]=roi_j[y==lab]           
        roi_sc=scaler.fit_transform(roi_mat) 
        estimator.fit(roi_sc)
        matrix=estimator.covariance_ 
        result_matrix[i,j]=1-matrix

np.savez_compressed('F:/IRM_Marche/dismatrix.npz',result_matrix)


