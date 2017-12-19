# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:32:01 2017

@author: mmenoret
"""
def get_masker_coord():
    
    basc = datasets.fetch_atlas_basc_multiscale_2015(version='sym')['scale444']    
    nib_basc444 = nib.load(basc)
    labels_data = nib_basc444.get_data()   
    #fetch all possible label values 
    all_labels = np.unique(labels_data)
    # remove the 0. value which correspond to voxels out of ROIs
    all_labels = all_labels[1:]
    bari_labels = np.zeros((all_labels.shape[0],3))
    ## go through all labels 
    for i,curlabel in enumerate(all_labels):
        vox_in_label = np.stack(np.argwhere(labels_data == curlabel))
        bari_labels[i] = vox_in_label.mean(axis=0)
    return  bari_labels 

#################################################
from nilearn import datasets
import nibabel as nib
import numpy as np
from gsplearn.GSPTransform import GraphTransformer
from sklearn.pipeline import Pipeline   
from sklearn.svm import SVC
from sklearn import preprocessing 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import LeaveOneLabelOut, cross_val_score
#wdir='Z:/Python/Scripts'
from sklearn.metrics import accuracy_score
names='af','ba','be','br','ds','ea','fj','gc','gv','hc','hn','lbc','lc','lp','my','mc','pj','pf','rs','wl'

smt='ss'
fold_g = 'F:/IRM_Marche/'
blocks=np.loadtxt(fold_g+'block.txt','int')
label=np.loadtxt(fold_g+'label.txt','S12')
labelgen=np.loadtxt(fold_g+'label_reststim.txt','S12')
condition_mask = np.logical_or(label == b'restconf', label == b'conf')
y=label[condition_mask]
block=blocks[condition_mask]
cv = LeaveOneLabelOut(block)    

coords=get_masker_coord()
k=100
feature_selection = SelectKBest(f_classif, k=k)
scaler = preprocessing.StandardScaler()
svm= SVC(C=1., kernel="linear")  

result=[]
for n in names:
    sim_filename=fold_g+'mni/roi_'+smt+'_'+n+'.npz'
    rest_filename=  fold_g+'mni/roirest_'+smt+'_'+n+'.npz'       
    
    roi=np.load(sim_filename)['roi']
    rest=np.load(rest_filename)['roi']
    cond=roi[condition_mask]
    roi_train=np.append(roi[0:150],roi[450:750],axis=0)#roi[0:150]#
    y_train=np.append(labelgen[0:150],labelgen[450:750],axis=0)#labelgen[0:150]#
    roi_test=roi[300:450]
    y_test=labelgen[300:450]
    
    #gr=GraphTransformer(rest=rest, coords=coords, kind='functional',
    #                 method='covariance',spars=0.1)
    gr=GraphTransformer(rest=rest, coords=coords, kind='geometric',
                     method='distance',spars=0,geo_alpha=0.0001)
    gr.fit(cond)
    GW=gr.G.W.toarray()
    #data=gr.transform(cond)
    pipeline_graph_anova = Pipeline([('graph',gr),('anova', feature_selection), ('scale', scaler),('classif_name', svm)])
    
    
    
    from sklearn.grid_search import GridSearchCV
    
    param = [
      {'graph__kind': ['geometric'], 'graph__method':['distance'],'graph__spars':[0.],'graph__geo_alpha':[0.00015]},
      {'graph__kind': ['functional'], 'graph__method':['covariance','correlation'],'graph__spars':[0.1,0.5]},
      {'graph__kind': ['mixed'], 'graph__method':['covariance'],'graph__spars':[15]}, 
     ]
    grid = GridSearchCV(pipeline_graph_anova, param_grid=param, verbose=1)
    #nested_cv_scores = cross_val_score(grid, cond, y,cv=cv)
    #print("Nested CV score: %.4f" % np.mean(nested_cv_scores))
    
    grid.fit(roi_train, y_train)
    print(grid.best_params_)
    prediction = grid.predict(roi_test)      
    result.append(accuracy_score(prediction,y_test))
                   
#    pipeline_graph_anova.fit(roi_train, y_train)
#    prediction = pipeline_graph_anova.predict(roi_test)      
#    print accuracy_score(prediction,y_test)