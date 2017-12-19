# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:55:03 2017

@author: mmenoret
"""

import numpy as np
from scipy.stats import binom_test
from sklearn.pipeline import Pipeline   
from sklearn.svm import SVC
from sklearn import preprocessing 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from nilearn.datasets import load_mni152_brain_mask
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from sklearn.cross_validation import LeaveOneLabelOut, cross_val_score, permutation_test_score
from sklearn.externals.joblib import Memory
from nilearn.plotting import plot_stat_map
import sys
sys.path[0]='Z:/GitHub/gsp-learn/'
from gsplearn.GSPTransform import GraphTransformer
from gsplearn.GSPPlot import plot_selectedregions


motor_label=np.fromfile('F:/IRM_Marche/harv_sma_basc444asym.np','int')
#motor_label=np.arange(410,444)
fold='F:/IRM_marche/all_mni_imp_imag/'
smt='ss'       
names=('ap','as','bh','bi','boh','cmp','cas','cs','cb','gm','gn','gbn','mv',
       'ms','pm','pc','ph','pa','pv','pom','rdc','ti','vs',
       'an','bm','cc','ci','cjf','dm','fb','fm','gem','gmc','hnc','lm','mac',
       'marc','marm','om','pic','pr','qs','ris','sn','tj','va',
       'af','ba','be','br','ds','ea','fj','gc','gv','hc','hn',
       'lbc','lc','lp','my','mc','pj','pf','rs','wl',#'dc',     
      )

scaler = preprocessing.StandardScaler()
svm= SVC(C=1., kernel="linear")  
pipeline = Pipeline([('scale', scaler),('svm', svm)])
k=50 #best 52
feature_selection = SelectKBest(f_classif, k=k)   
pipeline_anova = Pipeline([('anova', feature_selection), ('scale', scaler),('svm', svm)])

block=np.loadtxt(fold+'block.txt','int')
label=np.loadtxt(fold+'label.txt','S12')

mask_block=block==block
for x in range(label.shape[0]):
    if label[x]!=label[x-1]:
        mask_block[x]=False
    elif label[x]!=label[x-2]:
        mask_block[x]=False

condition_imp = np.logical_or(label == b'restimp', label == b'imp')
mask_imp= np.logical_and(condition_imp,mask_block)
condition_imag = np.logical_or(label == b'restimag', label == b'imag')
mask_imag= np.logical_and(condition_imag,mask_block)

y_imp = label[mask_imp]
y_imag = label[mask_imag]
block_cond = block[mask_imp]
cv = LeaveOneLabelOut(block_cond)
roi_imp_all=np.zeros([0,len(motor_label)])
roi_imag_all=np.zeros([0,len(motor_label)])
y_imp_all=np.zeros(0)
y_imag_all=np.zeros(0)
groups=np.zeros(0)
for i,n in enumerate(sorted(names)):
    roi_name=fold+'asymroi_'+smt+'_'+n+'.npz'              
    roi=np.load(roi_name)['roi']
    roi=roi[:,motor_label-1]
    roi_imp=roi[mask_imp]
    roi_imag=roi[mask_imag]
    roi_imp_all=np.vstack((roi_imp_all,roi_imp))
    roi_imag_all=np.vstack((roi_imag_all,roi_imag))
    y_imp_all=np.append(y_imp_all,y_imp)
    y_imag_all=np.append(y_imag_all,y_imag)
    groups=np.append(groups,np.ones(len(y_imp))*i)
result_cv_tr_imp=[] 
result_cv_tr_imag=[]   
pipeline = Pipeline([('scale', scaler),('svm', svm)])    
from sklearn.model_selection import LeaveOneGroupOut
logo = LeaveOneGroupOut()
for train_index, test_index in logo.split(roi_imp_all, y_imp_all, groups):
    X_train, X_test = roi_imp_all[train_index], roi_imag_all[test_index]
    y_train, y_test = y_imp_all[train_index], y_imp_all[test_index]
    pipeline.fit(X_train,y_train)
    prediction = pipeline.predict(X_test)  
    result_cv_tr_imp.append(accuracy_score(prediction,y_test))
    
    X_train, X_test = roi_imag_all[train_index], roi_imp_all[test_index]
    y_train, y_test = y_imp_all[train_index], y_imp_all[test_index]
    pipeline.fit(X_train,y_train)
    prediction = pipeline.predict(X_test)  
    result_cv_tr_imag.append(accuracy_score(prediction,y_test))

from scipy.stats import ttest_1samp
tt,p=ttest_1samp(np.array(result_cv_tr_imag),0.5)