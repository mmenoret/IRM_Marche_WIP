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
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
#wdir='Z:/Python/Scripts'
from sklearn.metrics import accuracy_score
names='af','br','pj','rs','wl','be','ds','ea','fj','gc','gv','hc','hn','lbc','lc','lp','mc','my','pf'#'ba'
smt='w'
fold_g = 'F:/IRM_Marche/'
blocks=np.loadtxt(fold_g+'block.txt','int')
label=np.loadtxt(fold_g+'label.txt','S12')
labelgen=np.loadtxt(fold_g+'label_reststim.txt','S12')
   

coords=get_masker_coord()
k=100
feature_selection = SelectKBest(f_classif, k=k)
scaler = preprocessing.StandardScaler()
svm= SVC(C=1., kernel="linear")  

roi=np.zeros([0,187])
rest=np.zeros([0,187])
y=np.array([])
ygen=np.array([])
block=np.array([])
for n in names:
    sim_filename=fold_g+'nii_data/mni/dictroi_'+smt+'_'+n+'.npz'
    rest_filename=  fold_g+'nii_data/mni/dictrest_'+smt+'_'+n+'.npz'  
    tmproi=np.load(sim_filename)['roi']
    tmprest=np.load(rest_filename)['roi']
    roi=np.concatenate((roi,tmproi))
    rest=np.concatenate((rest,tmprest))
    y=np.append(y,label)
    block=np.append(block,blocks) 
    ygen=np.append(ygen,labelgen)

#

mask = np.logical_not((y == b'restimag', y == b'imag'))
train_mask=mask[0]==mask[1]
train_mask= np.logical_or(y == b'restconf', y == b'conf')
test_mask = np.logical_or(y == b'restimag', y == b'imag')
roi_train=roi[train_mask]
y_train=ygen[train_mask]
roi_test=roi[test_mask]
y_test=ygen[test_mask]

#condition_mask = np.logical_or(y == b'conf', y == b'des')
#
#cond=roi[condition_mask]
#y=y[condition_mask]
#block=block[condition_mask]
#cv = LeaveOneLabelOut(block/2) 

# CE QUI NE VA PAS - CALCULER POUR CHAQUE SUJET SES VALEUR DE CONNECTIVITE PUIS
# FAIRE MOYENNE
gr=GraphTransformer(rest=rest, coords=coords, kind='functional',
                 method='correlation',spars=0.5)
#gr=GraphTransformer(rest=rest, coords=coords, kind='geometric',
#                 method='distance',spars=0,geo_alpha=0.0001)
#gr.fit(cond)
#GW=gr.G.W.toarray()
##data=gr.transform(cond)
pipeline_graph_anova = Pipeline([('graph',gr),('anova', feature_selection), ('scale', scaler),('classif_name', svm)])
#
#
#
from sklearn.grid_search import GridSearchCV
#
param = [
  {'graph__kind': ['geometric'], 'graph__method':['distance'],'graph__spars':[0.,15],'graph__geo_alpha':[0.0001],'anova__k':[100]},
  {'graph__kind': ['functional'], 'graph__method':['correlation'],'graph__spars':[0.4,0.6,0.7],'anova__k':[100]},
  {'graph__kind': ['mixed'], 'graph__method':['correlation'],'graph__spars':[10,15,30],'anova__k':[100]}, 
 # {'graph__kind':[None],'anova__k':[100,300,'all']}
  ]
#grid = GridSearchCV(pipeline_graph_anova, param_grid=param, verbose=1)
#nested_cv_scores = cross_val_score(grid, cond, y,cv=cv)
#print(grid.best_params_)
#print("Nested CV score: %.4f" % np.mean(nested_cv_scores))

#grid.fit(roi_train, y_train)
#print(grid.best_params_)
#prediction = grid.predict(roi_test)      
#print('nested CV', accuracy_score(prediction,y_test))
               
#pipeline_graph_anova.fit(roi_train, y_train)
#prediction = pipeline_graph_anova.predict(roi_test)      
#print('Graph', accuracy_score(prediction,y_test))
#
pipeline = Pipeline([ ('scale', scaler),('classif_name', svm)])
pipeline.fit(roi_train, y_train)
prediction = pipeline.predict(roi_test)      
print('All', accuracy_score(prediction,y_test))

pipeline_anova = Pipeline([ ('anova', feature_selection),('scale', scaler),('classif_name', svm)])
#pipeline_anova.fit(roi_train, y_train)
#prediction = pipeline_anova.predict(roi_test)      
#print('Anova', accuracy_score(prediction,y_test))
#
pca = PCA(n_components=k,svd_solver = 'full')
pipeline_pca = Pipeline([ ('pca', pca),('scale', scaler),('classif_name', svm)])
#pipeline_pca.fit(roi_train, y_train)
#prediction = pipeline_pca.predict(roi_test)      
#print('PCA', accuracy_score(prediction,y_test))
#
ica=FastICA(n_components=k)
pipeline_ica = Pipeline([ ('ica', ica),('scale', scaler),('classif_name', svm)])
#pipeline_ica.fit(roi_train, y_train)
#prediction = pipeline_ica.predict(roi_test)      
#print('ICA', accuracy_score(prediction,y_test))
#
#scores = cross_val_score(pipeline_graph_anova, cond, y,cv=cv)
#print("Graph: %.4f" % np.mean(scores))
#scores = cross_val_score(pipeline, cond, y,cv=cv)
#print("ALL: %.4f" % np.mean(scores))
#scores = cross_val_score(pipeline_anova, cond, y,cv=cv)
#print("ANOVA: %.4f" % np.mean(scores))
#scores = cross_val_score(pipeline_pca, cond, y,cv=cv)
#print("PCA: %.4f" % np.mean(scores))
#scores = cross_val_score(pipeline_ica, cond, y,cv=cv)
#print("ICA: %.4f" % np.mean(scores))

