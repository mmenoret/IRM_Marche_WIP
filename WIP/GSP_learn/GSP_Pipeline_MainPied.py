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



from nilearn.datasets import load_mni152_brain_mask
basc = datasets.fetch_atlas_basc_multiscale_2015(version='asym')['scale444']
brainmask = load_mni152_brain_mask()
#masker = NiftiLabelsMasker(labels_img = basc, mask_img = brainmask, 
#                           memory_level=1, verbose=0,
#                           detrend=True, standardize=False,  
#                           high_pass=0.01,t_r=2.28,
#                           resampling_target='labels'
#                           )
#masker.fit()

scaler = preprocessing.StandardScaler()
svm= SVC(C=1., kernel="linear")  
k=60
feature_selection = SelectKBest(f_classif, k=k)
#
## INDIVIDUAL ANALYSIS
#index=[]
#for x in range(label_i.shape[0]):
#    if label_i[x,0]!=label_i[x-1,0]:
#        index.append(x)
#    elif label_i[x,0]!=label_i[x-2,0]:
#        index.append(x)
#        
#label_i=np.delete(label_i,index,0)
#
#blocks_i=np.delete(blocks_i,index,0)
#condition_cat = np.logical_or(label_i[:,1] == b'foot', label_i[:,1] == b'hand')
#condition_out=np.logical_not(label_i[:,2]== b'des')
#condition_mask= condition_cat==condition_out
#y=label_i[condition_mask]
#block=blocks_i[condition_mask]
#cv = LeaveOneLabelOut(block)    
#train_mask= y[:,0]==b'stim'
#test_mask= y[:,0]==b'imag'
#y_train=y[train_mask,1]
#y_test=y[test_mask,1]
#block=block[train_mask]    
#
##coords=get_masker_coord('BASC')
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
##    gr=GraphTransformer(rest=rest, coords=coords, kind='functional',
##                     method='covariance',spars=0.3)
###    gr=GraphTransformer(rest=rest, coords=coords, kind='geometric',
###                     method='distance',spars=0,geo_alpha=0.0001)
##    gr.fit(cond)
##    GW=gr.G.W.toarray()
##    data=gr.transform(cond)
##    pipeline_graph_anova = Pipeline([('graph',gr),('anova', feature_selection), ('scale', scaler),('classif_name', svm)])
#    pipeline_anova = Pipeline([('anova', feature_selection), ('scale', scaler),('classif_name', svm)])
#    
#    pipeline_anova.fit(roi_train, y_train)
#    prediction = pipeline_anova.predict(roi_test)  
#    result=accuracy_score(prediction,y_test)
##    print(n,'anova',result)
#
### Permutation Null Score
#    nb_p=100
#    if result>0.6:
#        null_result=np.zeros(nb_p)
#        for i in range(nb_p):
#            y_train_random=np.random.permutation(y_train)
#            pipeline_anova.fit(roi_train, y_train_random)
#            prediction = pipeline_anova.predict(roi_test) 
#            null_result[i]=accuracy_score(prediction,y_test)
#
#        sign=(null_result>=result).sum()/nb_p
#        print(n,'anova',result,sign)
#    else: print(n,'anova',result)
#    ## Cat IMP/DES CROSS VALIDATION STIM
##
#    cv = LeaveOneLabelOut(block)
#    score_cv = cross_val_score(pipeline_anova, roi_train, y_train,cv=cv)   
#    null_score_cv= permutation_test_score(pipeline_anova, roi_train, y_train,cv=cv)
#    print(n,'anova stimCV', score_cv.mean(),null_score_cv[2])
#    #weights=pipeline_anova.named_steps['classif_name'].coef_
###plot_selectedregions(pipeline_anova,masker,weights=weights,anova_name='anova')
#
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
##    pipeline_graph_anova.fit(roi_train, y_train)
##    prediction = pipeline_graph_anova.predict(roi_test)      
##    print(n,'graph',accuracy_score(prediction,y_test))

############################
# ALL SUJ

roi=np.zeros([0,444])
rest=np.zeros([0,444])
label=np.zeros([0,3])
blocks=np.array([])
for n in names:
    sim_filename=fold_g+'mni4060/roi_'+smt+'_'+n+'.npz'
    rest_filename=  fold_g+'mni4060/roirest_'+smt+'_'+n+'.npz'  
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

condition_cat = np.logical_or(label[:,1] == b'foot', label[:,1] == b'hand')
condition_out=np.logical_not(label[:,2]== b'des')
condition_mask= condition_cat==condition_out

y=label[condition_mask]
block=blocks[condition_mask]
cond=roi[condition_mask]

cv = LeaveOneLabelOut(block) 
   
train_mask= y[:,0]==b'stim'
test_mask= y[:,0]==b'imag'
y_train=y[train_mask,1]
y_test=y[test_mask,1]
roi_train=cond[train_mask]#   
roi_test=cond[test_mask]
block=block[test_mask]

#k=60
feature_selection = SelectKBest(f_classif, k=k)
    
pipeline_anova = Pipeline([('anova', feature_selection), ('scale', scaler),('classif_name', svm)])
pipeline = Pipeline([('scale', scaler),('classif_name', svm)])
    
pipeline_anova.fit(roi_train, y_train)
prediction = pipeline_anova.predict(roi_test)  
print('anova',accuracy_score(prediction,y_test))
#result=accuracy_score(prediction,y_test)
#
#grid = GridSearchCV(pipeline_anova, param_grid={'anova__k':[20,40,60,80,100,200]}, verbose=1)
##weights=pipeline_anova.named_steps['classif_name'].coef_
##plot_selectedregions(pipeline_anova,masker,weights=weights,anova_name='anova')
#grid.fit(roi_train, y_train)
#print(grid.best_params_)
#prediction = grid.predict(roi_test)  
#print('anova GS', accuracy_score(prediction,y_test))
#
#pipeline.fit(roi_train, y_train)
#prediction = pipeline.predict(roi_test)  
#print('all', accuracy_score(prediction,y_test))
#
#
#
#gr=GraphTransformer(rest=rest, coords=coords, kind='mixed',
#                     method='correlation',spars=0.5,geo_alpha=0.00015)
##gr.fit(cond)
#
#param = [
#      {'graph__kind': ['geometric'], 'graph__method':['distance'],'graph__spars':[0.,0.5],'graph__geo_alpha':[0.00015]},
#      {'graph__kind': ['functional'], 'graph__method':['covariance','correlation'],'graph__spars':[0.1,0.3,0.5,0.7],'anova__k':[10,30,60,100,200]},
#      {'graph__kind': ['mixed'], 'graph__method':['covariance','correlation'],'graph__spars':[0.3,0.5,0.7]}, 
#     ]
#pipeline_graph_anova = Pipeline([('graph',gr),('anova', feature_selection), ('scale', scaler),('classif_name', svm)])
#grid = GridSearchCV(pipeline_graph_anova, param_grid=param, verbose=1)
##nested_cv_scores = cross_val_score(grid, cond, y,cv=cv)
##print("Nested CV score: %.4f" % np.mean(nested_cv_scores))
#
#grid.fit(roi_train, y_train)
#print(grid.best_params_)
#prediction = grid.predict(roi_test)      
#print('gsp GS',accuracy_score(prediction,y_test))
##
##pipeline_graph_anova.fit(roi_train, y_train)
##prediction = pipeline_graph_anova.predict(roi_test)      
##print('gsp',accuracy_score(prediction,y_test))
#
#
#
################################################################
#################### TEST NIVEAU CHANCE #########################
#### Permutation Null Score
##nb_p=10000
##
##null_result=np.zeros(nb_p)
##for i in range(nb_p):
##    y_train_random=np.random.permutation(y_train)
##    pipeline_anova.fit(roi_train, y_train_random)
##    prediction = pipeline_anova.predict(roi_test) 
##    null_result[i]=accuracy_score(prediction,y_test)
##
##sign=(null_result>=result).sum()/nb_p
##print(sign)
#
#### Permutation block
##nb_p=1000
##null_result=np.zeros(nb_p)
##for i in range(nb_p):
##    # shuffle block number
##    session=np.zeros(y_train.shape)
##    d=0
##    ses_bool=y_train==b'foot'
##    for x in range(ses_bool.size):
##        if ses_bool[x]!=ses_bool[x-1]:
##            d=d+1
##        session[x]=d
##        
##    x=np.unique(session)
##    xperm=np.random.permutation(x)
##    y_train_random=np.zeros(y_train.size,dtype='S12')
##    for z in range(x.size):
##        y_train_random[session==xperm[z]]=y_train[session==x[z]]
##        
##    pipeline_anova.fit(roi_train, y_train_random)
##    prediction = pipeline_anova.predict(roi_test) 
##    null_result[i]=accuracy_score(prediction,y_test)
##
##sign=(null_result>=result).sum()/nb_p
##print(sign)
#
### Permutation block
#nb_p=1000
#null_result=np.zeros(nb_p)
#
#ncond=['hand','foot']
#for i in range(nb_p):
#    y_train_random=np.zeros((0,1),dtype='S12')
#    # shuffle block number
#    for suj in range(22):
#        xncond=np.random.permutation(ncond)
#        suj_train_random=np.append(np.full(57,xncond[0],dtype='S12'),np.full(57,xncond[1],dtype='S12'))
#        y_train_random=np.append(y_train_random,suj_train_random)
#
#        
#    pipeline_anova.fit(roi_train, y_train_random)
#    prediction = pipeline_anova.predict(roi_test) 
#    null_result[i]=accuracy_score(prediction,y_test)
#
#sign=(null_result>=result).sum()/nb_p
#print(sign)
#
#########################
# Cat IMP/DES CROSS VALIDATION STIM

cv = LeaveOneLabelOut(block)
score_cv = cross_val_score(pipeline_anova, roi_train, y_train,cv=cv)
print('STIM only CV',score_cv.mean())
#null_score_cv= permutation_test_score(pipeline_anova, cond, y,cv=cv)#weights=pipeline_anova.named_steps['classif_name'].coef_
##plot_selectedregions(pipeline_anova,masker,weights=weights,anova_name='anova')
