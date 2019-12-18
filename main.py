# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:00:18 2019

@author: Julien
"""

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def load (file_name):
    img = nib.load(file_name)
    img_data = img.get_data()
    return img_data
    
def view4D(file_name,axe,b_plot=False,time=False):
    img_data = load(file_name)
    img_processed = []
    if not time :
        time = round(len(img_data[0][0][0])/2)
        
    if axe =='transversal':
        for i in range(len(img_data[0][0])):
            slice = img_data[:, :, i,time]
            img_processed.append(slice)
            if b_plot : 
                plot_slice(i,slice)
    elif axe =='coronal' or axe=='frontal':
        for i in range(len(img_data[0])):
            slice = img_data[:, i, :,time]
            img_processed.append(slice)
            if b_plot : 
                plot_slice(i,slice)
    elif axe =='sagital' or axe=='median':
        for i in range(len(img_data)):
            slice = img_data[i, :, :,time]
            img_processed.append(slice)
            if b_plot : 
                plot_slice(i,slice)
                
    return img_processed

def plot_slice(i,slice,subtitle=""):
    plt.figure(i)
    plt.imshow(slice.T, cmap="gray", origin="lower")
    plt.show
    plt.suptitle(subtitle)

data= load('Data/fmri.nii/fmri.nii')
np_imgs = view4D('Data/fmri.nii/fmri.nii','transversal', False)
# =============================================================================
# np_img = np_imgs[0]
# plot_slice(1,np_img,"Image source")
# =============================================================================

# =============================================================================
# img=nib.load('Data/fmri.nii/fmri.nii')
# img_data = img.get_data()
# img_header = img.header
# print(img_header["pixdim"])
# =============================================================================

def fft_voxel(img,x,y,z,plot=False):
    img_data = img.get_data()
    img_header = img.header
    voxel=img_data[x][y][z][:]
    #f = np.fft.fft(voxel)
    if plot :
        t = np.arange(85)
        # affichage du signal
        plt.subplot(211)
        plt.plot(t,voxel)
        
        # calcul de A
        A = np.fft.fft(voxel)
        # visualisation de A
        plt.subplot(212)
        plt.plot(np.real(A))
        plt.ylabel("partie reelle")
        
        plt.show()
    return

#fft_voxel(img,40,40,30,True)

def eliminate_non_brain(img_data):
    for t in range(0,85):
        for i in range(len(img_data[0][0])):
            slice = img_data[:, :, i,t]
            median = np.median(slice)
            indexes = slice<=median
            img_data[:, :, i,t][indexes]=0
    return

eliminate_non_brain(img_data)