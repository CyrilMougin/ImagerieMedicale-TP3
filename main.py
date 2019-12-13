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
    
def view(file_name,axe,b_plot=False):
    img_data = load(file_name)
    img_processed = []
    if axe =='transversal':
        for i in range(len(img_data[0][0])):
            slice = img_data[:, :, i]
            img_processed.append(slice)
            if b_plot : 
                plot_slice(i,slice)
    elif axe =='coronal' or axe=='frontal':
        for i in range(len(img_data[0])):
            slice = img_data[:, i, :]
            img_processed.append(slice)
            if b_plot : 
                plot_slice(i,slice)
    elif axe =='sagital' or axe=='median':
        for i in range(len(img_data)):
            slice = img_data[i, :, :]
            img_processed.append(slice)
            if b_plot : 
                plot_slice(i,slice)
                
    return img_processed

def plot_slice(i,slice,subtitle=""):
    plt.figure(i)
    plt.imshow(slice.T, cmap="gray", origin="lower")
    plt.show
    plt.suptitle(subtitle)

np_imgs = view('Data_MiseEnForme/IRM/Brain/flair.nii','transversal', False)
np_img = np_imgs[10]
plot_slice(np_img,"Image source")
    