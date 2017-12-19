# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:32:01 2017

@author: mmenoret
"""
import nibabel as nib 
from nilearn.plotting import find_xyz_cut_coords
from nilearn.image import math_img
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from sklearn.externals.joblib import Memory

def get_masker_coord(filename):

    if 'BASC' in filename:    
        basc = datasets.fetch_atlas_basc_multiscale_2015(version='sym')['scale444']
        filename=basc
        nib_basc444 = nib.load(basc)
        labels_data = nib_basc444.get_data()  
    else:       
        nib_parcel = nib.load(filename)
        labels_data = nib_parcel.get_data()   
    #fetch all possible label values 
    all_labels = np.unique(labels_data)
    # remove the 0. value which correspond to voxels out of ROIs
    all_labels = all_labels[1:]
#    bari_labels = np.zeros((all_labels.shape[0],3))
#    ## go through all labels 
#    for i,curlabel in enumerate(all_labels):
#        vox_in_label = np.stack(np.argwhere(labels_data == curlabel))
#        bari_labels[i] = vox_in_label.mean(axis=0)
#        
    allcoords=[]
    for i,curlabel in enumerate(all_labels):
        img_curlab = math_img(formula="img==%d"%curlabel,img=filename)
        allcoords.append(find_xyz_cut_coords(img_curlab))
    allcoords=np.array(allcoords)
    return  allcoords  

#################################################
import numpy as np
from gsplearn.GSPTransform import GraphTransformer
from gsplearn.GSPPlot import plot_selectedregions
from sklearn.pipeline import Pipeline   
from sklearn.svm import SVC
from sklearn import preprocessing 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import LeaveOneLabelOut, cross_val_score, permutation_test_score
from sklearn.grid_search import GridSearchCV

#wdir='Z:/Python/Scripts'
from sklearn.metrics import accuracy_score
names={'ap':'ALLAIN',
       'as':'ANDRE',
       'bh':'BEAUGE',
       'bi':'BROCHARD',
       'cmp':'CAPERAN',
       'cas':'CASINTHIE',
       'cs':'CHOIMET',
       'cb':'CHOLOUX',
       'gm':'GAUTIER',
       'gn':'GAUTIERNATH',
       'gbn':'GOURDONBELLARD',
       'mv':'MAROT',
       'ms':'MICHEL',
       'pm':'PAILLEY',
       'pc':'PAPON',
       'ph':'PESNEL',
       'pa':'PETIT',
       'pv':'PIC',
       'pom':'PONTIE',
       'rdc':'RODRIGUEZDIAZ',
       'ti':'TESSON',
       'vs':'VANWATERLOO',
       }
#names='ap','as','bh','bi','cmp','cas','cs','cb','gm','gn','gbn','mv','ms','pm','pc','ph','pa','pv','pom','rdc','ti','vs'
smt='ss'
fold_g = 'F:/IRM_Marche/'
blocks_i=np.loadtxt(fold_g+'block_main.txt','int')
label_i=np.loadtxt(fold_g+'label_main.txt','S12')


coords=get_masker_coord('BASC')
from nilearn.datasets import load_mni152_brain_mask
basc = datasets.fetch_atlas_basc_multiscale_2015(version='asym')['scale444']
brainmask = load_mni152_brain_mask()
masker = NiftiLabelsMasker(labels_img = basc, mask_img = brainmask, 
                           memory_level=1, verbose=0,
                           detrend=True, standardize=False,  
                           high_pass=0.01,t_r=2.28,
                           resampling_target='labels'
                           )
masker.fit()

scaler = preprocessing.StandardScaler()
svm= SVC(C=1., kernel="linear")  


## INDIVIDUAL ANALYSIS
#index=[]
#for x in range(label.shape[0]):
#    if label[x,0]!=label[x-1,0]:
#        index.append(x)
#    elif label[x,0]!=label[x-2,0]:
#        index.append(x)
#        
#label=np.delete(label,index,0)
#
#blocks=np.delete(blocks,index,0)
#condition_cat = np.logical_or(label[:,1] == b'foot', label[:,1] == b'hand')
#condition_out=np.logical_not(label[:,2]== b'des')
#condition_mask= condition_cat==condition_out
#y=label[condition_mask]
#block=blocks[condition_mask]
#cv = LeaveOneLabelOut(block)    
#train_mask= y[:,0]==b'stim'
#test_mask= y[:,0]==b'imag'
#y_train=y[train_mask,1]
#y_test=y[test_mask,1]
#    
#
#
#result=[]
#for n in names:
#    sim_filename=fold_g+'mni4060/roi_'+smt+'_'+n+'.npz'
#    rest_filename=  fold_g+'mni4060/roirest_'+smt+'_'+n+'.npz'       
#    
#    roi=np.load(sim_filename)['roi']
#    roi=np.delete(roi,index,0)
#    
#    rest=np.load(rest_filename)['roi']
#    cond=roi[condition_mask]
#
#    roi_train=cond[train_mask]#   
#    roi_test=cond[test_mask]
#    
#    
#    #gr=GraphTransformer(rest=rest, coords=coords, kind='functional',
#    #                 method='covariance',spars=0.1)
##    gr=GraphTransformer(rest=rest, coords=coords, kind='geometric',
##                     method='distance',spars=0,geo_alpha=0.0001)
##    gr.fit(cond)
##    GW=gr.G.W.toarray()
#    #data=gr.transform(cond)
##    pipeline_graph_anova = Pipeline([('graph',gr),('anova', feature_selection), ('scale', scaler),('classif_name', svm)])
#    pipeline_anova = Pipeline([('anova', feature_selection), ('scale', scaler),('classif_name', svm)])
#    
#    pipeline_anova.fit(roi_train, y_train)
#    prediction = pipeline_anova.predict(roi_test)  
#    result.append(accuracy_score(prediction,y_test))
##    
##    
##    param = [
##      {'graph__kind': ['geometric'], 'graph__method':['distance'],'graph__spars':[0.],'graph__geo_alpha':[0.00015]},
##      {'graph__kind': ['functional'], 'graph__method':['covariance','correlation'],'graph__spars':[0.1,0.5]},
##      {'graph__kind': ['mixed'], 'graph__method':['covariance'],'graph__spars':[15]}, 
##     ]
##    grid = GridSearchCV(pipeline_graph_anova, param_grid=param, verbose=1)
##    #nested_cv_scores = cross_val_score(grid, cond, y,cv=cv)
##    #print("Nested CV score: %.4f" % np.mean(nested_cv_scores))
##    
##    grid.fit(roi_train, y_train)
##    print(grid.best_params_)
##    prediction = grid.predict(roi_test)      
##    result.append(accuracy_score(prediction,y_test))
#                   
###    pipeline_graph_anova.fit(roi_train, y_train)
###    prediction = pipeline_graph_anova.predict(roi_test)      
## #   print accuracy_score(prediction,y_test)

###########################
# ALL SUJ

roi=np.zeros([0,444])
rest=np.zeros([0,444])
label=np.zeros([0,3])
blocks=np.array([])
for n in names:
    sim_filename=fold_g+'mni4060/asymroi_'+smt+'_'+n+'.npz'
    rest_filename=  fold_g+'mni4060/asymroirest_'+smt+'_'+n+'.npz'  
    tmproi=np.load(sim_filename)['roi']
    tmprest=np.load(rest_filename)['roi']
    roi=np.concatenate((roi,tmproi))
    rest=np.concatenate((rest,tmprest))
    label=np.append(label,label_i,axis=0)
    blocks=np.append(blocks,blocks_i) 
    
index=[]
for x in range(label.shape[0]):
    if label[x,0]!=label[x-1,0]:
        index.append(x)
    elif label[x,0]!=label[x-2,0]:
        index.append(x)
        
label=np.delete(label,index,0)
blocks=np.delete(blocks,index,0)
roi=np.delete(roi,index,0)

condition_mask = np.logical_or(label[:,2] == b'imp', label[:,2] == b'des')

y=label[condition_mask,2]
block=blocks[condition_mask]
cond=roi[condition_mask]


k=60
feature_selection = SelectKBest(f_classif, k=k)
    
pipeline_anova = Pipeline([('anova', feature_selection), ('scale', scaler),('classif_name', svm)])
pipeline = Pipeline([('scale', scaler),('classif_name', svm)])
grid = GridSearchCV(pipeline_anova, param_grid={'anova__k':[20,60,100,200]}, verbose=1)


gr=GraphTransformer(rest=rest, coords=coords, kind='mixed',
                     method='correlation',spars=0.5,geo_alpha=0.00015)
param = [
      {'graph__kind': ['geometric'], 'graph__method':['distance'],'graph__spars':[0.,0.5],'graph__geo_alpha':[0.00015]},
      {'graph__kind': ['functional'], 'graph__method':['covariance','correlation'],'graph__spars':[0.1,0.3,0.5,0.7],'anova__k':[10,30,60,100,200]},
      {'graph__kind': ['mixed'], 'graph__method':['covariance','correlation'],'graph__spars':[0.3,0.5,0.7]}, 
     ]
pipeline_graph_anova = Pipeline([('graph',gr),('anova', feature_selection), ('scale', scaler),('classif_name', svm)])
grid_graph = GridSearchCV(pipeline_graph_anova, param_grid=param, verbose=1)
#nested_cv_scores = cross_val_score(grid, cond, y,cv=cv)
#print("Nested CV score: %.4f" % np.mean(nested_cv_scores))


########################
# Cat IMP/DES CROSS VALIDATION STIM

cv = LeaveOneLabelOut(block)
score_cv = cross_val_score(pipeline_anova, cond, y,cv=cv)
null_score_cv= permutation_test_score(pipeline_anova, cond, y,cv=cv)#weights=pipeline_anova.named_steps['classif_name'].coef_
#plot_selectedregions(pipeline_anova,masker,weights=weights,anova_name='anova')
