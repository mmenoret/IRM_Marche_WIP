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
from nilearn.plotting import plot_stat_map
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold



# DEFINITION FOR PERMUTATION SCORE
def group_shuffle(y, groups):
    """Return a shuffled copy of y eventually shuffle among same groups."""

    indices = np.arange(len(groups))
    for group in np.unique(groups):
        this_mask = (groups == group)
        indices[this_mask] = np.random.permutation(indices[this_mask])
    return y[indices]

def test_score(estimator,X_cond_train,X_cond_test,y,groups,cv):
    result_score = []
    for train, test in cv.split(X_cond_train, y, groups):
        estimator.fit(X_cond_train[train], y[train])
        result_score.append(estimator.score(X_cond_test[test], y[test]))
    score=np.array(result_score).mean()
    return score,result_score, 


def permutation_score(estimator,X_cond_train,X_cond_test,y,groups,cv,n_permutations=100):
    permutation_scores=[]
    score,result_score=test_score(clone(estimator),X_cond_train,X_cond_test,y,groups,cv)
    kf=GroupKFold(n_splits=3)
    for n in range(n_permutations):
        tmp_scores,w = test_score(clone(estimator), X_cond_train,X_cond_test, 
                              group_shuffle(y, groups), groups, kf)
        permutation_scores.append(tmp_scores)

    permutation_scores = np.array(permutation_scores)        
    pvalue = np.sum(permutation_scores >= score) / n_permutations
    return score, permutation_scores, pvalue

# PARAMETERS
motor_label=np.fromfile('F:/IRM_Marche/masquesROI/reg_motorC_basc444asym.np','int')
motor_label=np.arange(444)
fold = 'F:/IRM_Marche/'
smt='ss'       
names=('ap','as','bh','bi','boh','cmp','cas','cs','cb','gm','gn','gbn',
       'mv','ms','pm','pc','ph','pa','pv','pom','rdc','ti','vs'    
      )

# Prepare classifier
scaler = preprocessing.StandardScaler()
svm= SVC(C=1., kernel="linear")  
pipeline = Pipeline([('scale', scaler),('svm', svm)])
logo = LeaveOneGroupOut()
#logo=GroupKFold(n_splits=3)

# CREATE LABEL / COND
label=np.loadtxt(fold+'label_main3.txt','S12')
block=np.loadtxt(fold+'block_main.txt','int')

mask_block=block==block
for x in range(label.shape[0]):
    if label[x,2]!=label[x-1,2]:
        mask_block[x]=False
    elif label[x,2]!=label[x-2,2]:
        mask_block[x]=False

c_des_out=np.logical_not(label[:,2]== b'restdes')
tmp_out= np.logical_and(c_des_out,mask_block)
c_rest=(label[:,0]== b'rest')
cond_out= np.logical_and(tmp_out,c_rest)

cond_foot=np.logical_or(label[:,2] == b'restimp',label[:,2] == b'restpimag')
cond_foot=np.logical_and(cond_out,cond_foot)
cond_hand=np.logical_or(label[:,2] == b'restmboug',label[:,2] == b'restmimag')
cond_hand=np.logical_and(cond_out,cond_hand)
cond_imag=np.logical_or(label[:,2] == b'restmimag',label[:,2] == b'restpimag')
cond_imag=np.logical_and(cond_out,cond_imag)
cond_stim=np.logical_or(label[:,2] == b'restimp',label[:,2] == b'restmboug')
cond_stim=np.logical_and(cond_out,cond_stim)

y_foot=label[cond_foot,3]
y_hand=label[cond_hand,3]
y_imag=label[cond_imag,4]
y_stim=label[cond_stim,4]

# LOAD DATA AND GROUP ALL SUBJETS
roi_foot_all=np.zeros([0,len(motor_label)])
roi_hand_all=np.zeros([0,len(motor_label)])
roi_imag_all=np.zeros([0,len(motor_label)])
roi_stim_all=np.zeros([0,len(motor_label)])

y_foot_all=np.zeros(0)
y_hand_all=np.zeros(0)
y_imag_all=np.zeros(0)
y_stim_all=np.zeros(0)
groups=np.zeros(0)

for i,n in enumerate(sorted(names)):
    roi_name=fold+'mni4060/asymroi_'+smt+'_'+n+'.npz'            
    roi=np.load(roi_name)['roi']
    roi=roi[:,motor_label-1]
    roi_foot=roi[cond_foot]
    roi_hand=roi[cond_hand]
    roi_imag=roi[cond_imag]
    roi_stim=roi[cond_stim]

    roi_foot_all=np.vstack((roi_foot_all,roi_foot))
    roi_hand_all=np.vstack((roi_hand_all,roi_hand))
    roi_imag_all=np.vstack((roi_imag_all,roi_imag))
    roi_stim_all=np.vstack((roi_stim_all,roi_stim))
    
    y_foot_all=np.append(y_foot_all,y_foot)
    y_hand_all=np.append(y_hand_all,y_hand)
    y_imag_all=np.append(y_imag_all,y_imag)
    y_stim_all=np.append(y_stim_all,y_stim)
    
    groups=np.append(groups,np.ones(len(y_foot))*i)
    
# CROSS MODAL CLASSIFICATION (cross-validated)
#n_p=0
#result_cv_tr_foot,permutation_scores_tr_foot, p_foot=permutation_score(pipeline,roi_foot_all,roi_hand_all,y_foot_all,groups,logo,n_p) 
#print('Train FOOT - IMAG VS STIM',np.array(result_cv_tr_foot).mean(),p_foot)
#result_cv_tr_hand,permutation_scores_tr_hand, p_hand=permutation_score(pipeline,roi_hand_all,roi_foot_all,y_hand_all,groups,logo,n_p)
#print('Train HAND - IMAG VS STIM',np.array(result_cv_tr_hand).mean(),p_hand)
#result_cv_tr_imag,permutation_scores_tr_imag, p_imag=permutation_score(pipeline,roi_imag_all,roi_stim_all,y_imag_all,groups,logo,n_p)   
#print('Train IMAG - HAND VS FOOT',np.array(result_cv_tr_imag).mean(),p_imag)
#result_cv_tr_stim,permutation_scores_tr_stim, p_stim=permutation_score(pipeline,roi_stim_all,roi_imag_all,y_stim_all,groups,logo,n_p)
#print('Train STIM - HAND VS FOOT',np.array(result_cv_tr_stim).mean(),p_stim)


# Prepare ploting
basc = datasets.fetch_atlas_basc_multiscale_2015(version='asym')['scale444']
brainmask = load_mni152_brain_mask()
masker = NiftiLabelsMasker(labels_img = basc, mask_img = brainmask, 
                           memory_level=1, verbose=0,
                           detrend=True, standardize=False,  
                           high_pass=0.01,t_r=2.28,
                           resampling_target='labels'
                           )
masker.fit()

pipeline.fit(roi_foot_all,y_foot_all)
coef_foot = pipeline.named_steps['svm'].coef_
weight_f = masker.inverse_transform(coef_foot)
plot_stat_map(weight_f, title='Train Imp',display_mode='z',cmap='bwr',threshold=0.4)


pipeline.fit(roi_hand_all,y_hand_all)
coef_hand = pipeline.named_steps['svm'].coef_
weight_h = masker.inverse_transform(coef_hand)
plot_stat_map(weight_h, title='Train Imp',display_mode='z',cmap='bwr',threshold=0.1)


from sklearn.preprocessing import normalize
#nh=normalize(coef_imp)
#nf=normalize(coef_imag)
#coef_common=nh.T*nf.T
coef_common=coef_hand.T*coef_foot.T
z = coef_common[:]
for i,x in enumerate(coef_common):
    if x<=0:
        z[i]=0
    else:
        z[i]=x

for i,x in enumerate(coef_common):
    if coef_foot.T[i]<0:
        z[i]=-z[i]
z=z.T

weight = masker.inverse_transform(z)
plot_stat_map(weight, title='Regions',display_mode='z',cut_coords=[-50,-32,-20,16,46,61,70],
              cmap='bwr',threshold=0.005,vmax=0.03
             )

#result_foot,w=test_score(pipeline,roi_foot_all,roi_foot_all,y_foot_all,groups,logo) 
#print('FOOT - IMAG VS STIM',result_foot)
#result_hand,w=test_score(pipeline,roi_hand_all,roi_hand_all,y_hand_all,groups,logo)
#print('HAND - IMAG VS STIM',result_hand)
#result_imag,w=test_score(pipeline,roi_imag_all,roi_imag_all,y_imag_all,groups,logo)   
#print('IMAG - HAND VS FOOT',result_imag)
#result_stim,w=test_score(pipeline,roi_stim_all,roi_stim_all,y_stim_all,groups,logo)
#print('STIM - HAND VS FOOT',result_stim)

#result_foot,permutation_scores_tr_foot, p_foot=permutation_score(pipeline,roi_foot_all,roi_foot_all,y_foot_all,groups,logo,n_p) 
#print('FOOT - IMAG VS STIM',np.array(result_cv_tr_foot).mean(),p_foot)
#result_hand,permutation_scores_tr_hand, p_hand=permutation_score(pipeline,roi_hand_all,roi_hand_all,y_hand_all,groups,logo,n_p)
#print('HAND - IMAG VS STIM',np.array(result_cv_tr_hand).mean(),p_hand)
#result_imag,permutation_scores_tr_imag, p_imag=permutation_score(pipeline,roi_imag_all,roi_imag_all,y_imag_all,groups,logo,n_p)   
#print('IMAG - HAND VS FOOT',np.array(result_cv_tr_imag).mean(),p_imag)
#result_stim,permutation_scores_tr_stim, p_stim=permutation_score(pipeline,roi_stim_all,roi_stim_all,y_stim_all,groups,logo,n_p)
#print('STIM - HAND VS FOOT',np.array(result_cv_tr_stim).mean(),p_stim)
#



