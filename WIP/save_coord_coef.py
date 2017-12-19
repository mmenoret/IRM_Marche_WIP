# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:14:52 2017

@author: mmenoret
"""
# Prepare ploting
import numpy as np
from nilearn.datasets import load_mni152_brain_mask
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib 
from nilearn.plotting import find_xyz_cut_coords
from nilearn.image import math_img
from nilearn import image
import pandas as pd
from nibabel.affines import apply_affine
import numpy.linalg as npl

basc = datasets.fetch_atlas_basc_multiscale_2015(version='asym')['scale444']
brainmask = load_mni152_brain_mask()
masker = NiftiLabelsMasker(labels_img = basc, mask_img = brainmask, 
                           memory_level=1, verbose=0,
                           detrend=False, standardize=False,  
                           t_r=2.28,
                           resampling_target='labels'
                           )
masker.fit()
nib_parcel = nib.load(basc)
labels_data = nib_parcel.get_data()         
#fetch all possible label values 
all_labels = np.unique(labels_data)
# remove the 0. value which correspond to voxels out of ROIs
all_labels = all_labels[1:]
allcoords=[]
basc_vox=[]
for i,curlabel in enumerate(all_labels):
# Methods nilearn
    img_curlab = math_img(formula="img==%d"%curlabel,img=basc)
    allcoords.append(find_xyz_cut_coords(img_curlab))
    basc_vox.append((labels_data==curlabel).sum())
# Method moyenne
#    vox_in_label = np.stack(np.argwhere(labels_data == curlabel))
#    allcoords.append(vox_in_label.mean(axis=0))
allcoords=np.array(allcoords)
basc_vox=np.array(basc_vox)
affine_return = npl.inv(nib_parcel.affine)
#scancoords=apply_affine(affine_return, allcoords)
#scancoords=np.array(scancoords,dtype=int)
from nilearn.datasets import fetch_atlas_aal
aal_atlas=fetch_atlas_aal(version='SPM12')
aal_label=np.array(aal_atlas.labels,dtype='S20')
aal_indices=np.array(aal_atlas.indices,dtype='f4')
aal_map=aal_atlas.maps

#aal_map='F:/IRM_Marche/masquesROI/aal_MNI_V4.nii'
#aal_map='F:/IRM_Marche/masquesROI/aal.nii.gz'
from nilearn.plotting import plot_roi
from nilearn import input_data
#plot_roi(basc)
#plot_roi(aal_map)

fold='F:/IRM_Marche/result/'
filelist=['compHANDFOOT_trainIMP_coefPOS',
          'compHANDFOOT_trainIMAG_coePOS',
          'compHANDFOOT_trainIMP_coefNEG',
          'compHANDFOOT_trainIMAG_coefNEG',
          'compHANDFOOT_COMMON_coefPOS',
          'compHANDFOOT_COMMON_coefNEG',
          'compIMAGSTIM_trainHAND_coefPOS',
          'compIMAGSTIM_trainFOOT_coePOS',
          'compIMAGSTIM_trainHAND_coefNEG',
          'compIMAGSTIM_trainFOOT_coefNEG',
          'compIMAGSTIM_COMMON_coefPOS',
          'compIMAGSTIM_COMMON_coefNEG',
          ]
writer = pd.ExcelWriter(fold+'result_f.xlsx')
for name in sorted(filelist):
    filename=fold+name+'_th90.nii'
    coef = masker.transform(filename)
 #   data=np.zeros(image.mean_img(filename).shape)
 #   for i,(x,y,z) in enumerate(scancoords):
 #       data[x,y,z]=coef[:,i]
 #   voximg=image.new_img_like(filename,data)
 #   voximg.to_filename(fold+name+'_vox.nii')
    mask=(coef[0]!=0).T
    basc_n=all_labels[mask]
    basc_n=basc_n.reshape((basc_n.shape[0],1))
    n_vox=basc_vox[mask]
    n_vox=n_vox.reshape((n_vox.shape[0],1))
    coord=np.round(allcoords[mask])
    spheres_masker = input_data.NiftiSpheresMasker(seeds=coord)
    aal_value = spheres_masker.fit_transform(aal_map)
    label=[]
    for index in aal_value.T:
        if index == 0:
#            label.append('none')
            label.append(np.array((['none']),dtype=str))
          
        else:
            label.append(aal_label[aal_indices==int(index)])
#            label.append(aal_label[int(index)-1])

    label=np.array(label)

 #   coord_scan=scancoords[mask]
    coef_f=coef.T[mask]
    final=np.hstack((basc_n,n_vox,label,coord,coef_f))
    data=pd.DataFrame(final,columns=('BASC region','Region size (in voxel)','Anatomical region','X','Y','Z','SVM coefficient'),dtype=str)   
    data.to_excel(writer,sheet_name=name,index=False)
writer.close()