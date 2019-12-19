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
    
def view4D(img_data,axe,b_plot=False,time=False):
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

# =============================================================================
# data= load('Data/fmri.nii/fmri.nii')
# np_imgs = view4D('Data/fmri.nii/fmri.nii','transversal', False)
# np_img = np_imgs[0]
# plot_slice(1,np_img,"Image source")
# =============================================================================

img=nib.load('Data/fmri.nii/fmri.nii')
img_data = img.get_data()
img_header = img.header

def eliminate_non_brain(img_data):
    for t in range(0,85):
        for i in range(len(img_data[0][0])):
            slice = img_data[:, :, i,t]
            median = np.median(slice)
            indexes = slice<=median
            img_data[:, :, i,t][indexes]=0
    return img_data


img_data = eliminate_non_brain(img_data)
acq_num=85

# Select a random voxel by getting one random x- and y-coorinate
x_voxel = np.random.randint(64)
y_voxel = np.random.randint(64)
z_voxel = np.random.randint(50)

# Create the plot
fig, ax = plt.subplots(1,1,figsize=(15, 5))
ax.plot(img_data[x_voxel, y_voxel,z_voxel,:], lw=3)
ax.set_xlim(0, acq_num-1)
ax.set_xlabel('time [volumes]', fontsize=20)
ax.set_ylabel('signal strength', fontsize=20)
ax.set_title("Courbe d'intensité d'un voxel aléatoire", fontsize=25)
ax.tick_params(labelsize=12)
plt.show()

# Average all volumes
mean_data = img_data.mean(axis=3)

# Create the design matrix
constant = np.ones(acq_num)
with open("Data/ideal.txt") as f:
    content = f.readlines()
predicted_response= [float(x.strip()) for x in content] 

design_matrix = np.array((constant, predicted_response))

fig.subplots_adjust(wspace=0, hspace=0.5)
plt.show()

img_correlation=np.zeros((64,64,50))
# Calculate the correlation coefficients - for each voxel
c_max=0
x_max=0
y_max=0
z_max=0
for x in range(64):
    for y in range(64):
        for z in range(50):
            real_response=img_data[x,y,z,:]
            c = np.corrcoef(predicted_response, real_response)[1:,0]
            if np.abs(c[0]) > 0.2:     
                img_correlation[x,y,z]=c[0]
            else:
                img_correlation[x,y,z]=np.nan
            if np.abs(c[0]) > c_max :
                c_max=c
                x_max=x
                y_max=y
                z_max=z

# Find the voxel with the highest correlation coefficient
strongest_correlated = img_data[x_max,y_max,z_max,:]

# Define the min-max scaling function
def scale(data):
    return (data - data.min()) / (data.max() - data.min())

# Scale the voxel with the highest correlation
strongest_correlated_scaled = scale(img_data[x_max,y_max,z_max,:])

# Create the plots
fig, (ax) = plt.subplots(1,1,figsize=(15, 5))
ax.plot(strongest_correlated_scaled, label='voxel timecourse', lw=3)
ax.plot(design_matrix[1,:], label='design matrix', lw=3)
ax.set_xlim(0, acq_num-1)
ax.set_ylim(-0.25, 1.5)
ax.set_xlabel('time [volumes]', fontsize=20)
ax.set_ylabel('scaled response', fontsize=20)
ax.tick_params(labelsize=12)
ax.legend()
plt.show()

# affichage slice par slice des correlations
for i in range(len(img_data[0][0])):
    fig, ax = plt.subplots(1,1,figsize=(18, 6))
    ax.imshow(mean_data[:,:,i], cmap='gray')
    ax.imshow(img_correlation[:,:,i], cmap='afmhot')
    ax.set_title('thresholded map (overlay)', fontsize=25)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.show()
    
# Look at activation for 2 different voxels:
signal1 = img_data[x_max, y_max, z_max, :]
signal2 = img_data[0, 0, 0, :]
time = np.arange(0, 340, 4)
plt.plot(time, signal1, 'b')
plt.plot(time, signal2, 'r')
plt.legend(['Activation', 'No activation'], loc='best')
plt.show()
# Compute fft for a voxel:
# input signal is 85 long, 2**7 = 128 seems to be enough
freq_signal1 = np.squeeze(np.fft.fft(signal1, n=2**7))
freq_signal2 = np.squeeze(np.fft.fft(signal2, n=2**7))
timestep = 3.864
freq = np.squeeze(np.fft.fftfreq(2**7, d=timestep))
plt.plot(freq[:63], np.absolute(freq_signal1[:63]), 'b')
plt.plot(freq[:63], np.absolute(freq_signal2[:63]), 'r')
plt.xlim((0, 0.13))
plt.ylim((-100, 500))
plt.legend(['Activation', 'No activation'], loc='best')
plt.xticks(np.arange(0, max(freq[:63]), 0.02))
plt.show()
